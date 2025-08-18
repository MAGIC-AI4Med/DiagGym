# -*- coding: utf-8 -*-
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ======== OpenAI API Setup ========
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")

client = OpenAI(api_key=API_KEY)

def workflow(messages, model="gpt-4o", temperature=0.0, max_tokens=1024):
    """
    Call the OpenAI API to process a conversation.

    Args:
        messages (list): List of dicts with role/content, e.g.:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        model (str): Model name (default: gpt-4o).
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens to generate.

    Returns:
        str: Model's response text.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


# ======== Prompt Building ========
def load_instruction(txt_path):
    """Load instruction template from a text file."""
    with open(txt_path, encoding='utf-8') as fp:
        return fp.read()


def build_accuracy_fullchain_prompt(pred_diag, gt_diag):
    """
    Build the prompt for evaluating diagnosis accuracy.

    Args:
        pred_diag (str): Predicted diagnosis.
        gt_diag (str): Ground truth diagnosis.

    Returns:
        str: Formatted evaluation prompt.
    """
    instruction = load_instruction("instructions/accuracy.txt")
    return instruction.format(pred_diag=pred_diag, gt_diag=gt_diag)


# ======== Multithreaded Processing ========
def process_single_prompt(prompt_data, max_retry=5):
    """
    Process a single prompt for GPT evaluation (supports retries).

    Args:
        prompt_data (tuple): (item_idx, exam_idx, task_type, prompt)
        max_retry (int): Maximum retry attempts.

    Returns:
        tuple: ((item_idx, exam_idx, task_type), result)
    """
    item_idx, exam_idx, task_type, prompt = prompt_data

    for i in range(max_retry):
        try:
            result = workflow([{"role": "user", "content": prompt}])
            return (item_idx, exam_idx, task_type), result
        except Exception as e:
            if i == max_retry - 1:
                print(f"[ERROR] item {item_idx} exam {exam_idx} task {task_type}: {e}")
                return (item_idx, exam_idx, task_type), f"ERROR: {e}"
            time.sleep(2 + i * 2)


def send_gpt_multithread(prompts, max_workers=4, max_retry=5):
    """
    Send multiple prompts to GPT using multithreading.

    Args:
        prompts (list): List of (item_idx, exam_idx, task_type, prompt) tuples.
        max_workers (int): Number of threads.
        max_retry (int): Maximum retry attempts.

    Returns:
        dict: Mapping (item_idx, exam_idx, task_type) -> result.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {
            executor.submit(process_single_prompt, prompt_data, max_retry): prompt_data
            for prompt_data in prompts
        }

        for future in tqdm(as_completed(future_to_prompt), total=len(prompts), desc="Processing"):
            try:
                key, result = future.result()
                results[key] = result
            except Exception as e:
                item_idx, exam_idx, task_type, _ = future_to_prompt[future]
                print(f"[THREAD ERROR] item {item_idx} exam {exam_idx} task {task_type}: {e}")
                results[(item_idx, exam_idx, task_type)] = f"THREAD ERROR: {e}"
    return results


# ======== Accuracy Evaluation ========
def evaluate_accuracy_multithread(json_filepath, save_filepath, max_workers=4, max_retry=5):
    """
    Evaluate diagnosis accuracy using GPT in a multithreaded manner.

    Args:
        json_filepath (str): Path to generated diagnosis JSON.
        save_filepath (str): Path to save evaluation results.
        max_workers (int): Number of threads.
        max_retry (int): Maximum retry attempts.
    """
    with open('testset_with_raw_data.json', 'r') as fp:
        testset = json.load(fp)

    # Mapping for ground truth
    gt_mapping = {item['note_id']: item['final_diagnosis'] for item in testset}

    with open(json_filepath, 'r') as f:
        json_data = json.load(f)

    eval_results = {}
    prompts = []

    for item_idx, item in enumerate(json_data):
        note_id = item['note_id']
        last_msg = item['generated_diagnosis']
        gt = item.get('ground_truth', gt_mapping.get(note_id))

        try:
            if 'Diagnosis:' in last_msg:
                pred = last_msg.split('Diagnosis:')[1].split('Reason:')[0].strip()
            else:
                pred = last_msg.split('Reason:')[0].strip()

            prompt = build_accuracy_fullchain_prompt(pred, gt)
            prompts.append((item_idx, note_id, "accuracy", prompt))
        except Exception:
            continue

    print(f"Using {max_workers} threads to process {len(prompts)} prompts...")
    results = send_gpt_multithread(prompts, max_workers=max_workers, max_retry=max_retry)

    for (_, note_id, _), result in results.items():
        eval_results[note_id] = result

    with open(save_filepath, 'w') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    print(f"Accuracy evaluation saved to {save_filepath}")


# ======== Dataset Loader ========
def load_dataset(data_path='testset_with_raw_data.json'):
    """
    Load dataset from JSON and prepare exams_dict for each case.

    Args:
        data_path (str): Path to dataset JSON.

    Returns:
        list: Loaded dataset with 'exams_dict' field added.
    """
    with open(data_path, 'r') as fp:
        testset_data = json.load(fp)

    for idx, item in enumerate(testset_data):
        events = item['raw_data']['events']
        exams_dict = {event['data']['label']: event['data']['lab_data'] for event in events}

        for label, value in item.get('key_pertinent_results_dict', {}).items():
            if label not in exams_dict:
                exams_dict[label] = value

        testset_data[idx]['exams_dict'] = exams_dict
        testset_data[idx]['text'] = item['raw_data']['text']

    return testset_data


# ======== Main ========
if __name__ == "__main__":
    filenames = []
    existing_results = os.listdir('metric_results')

    for filename in os.listdir('outputs'):
        if (
            f'final_diag_acc_{filename}' not in existing_results
            and 'tmp' not in filename
            and filename.startswith('final_diagnose')
        ):
            filenames.append(filename)

    print(f"Files to process: {filenames}")

    for filename in filenames:
        evaluate_accuracy_multithread(
            json_filepath=os.path.join('outputs', filename),
            save_filepath=os.path.join('metric_results', f'final_diag_acc_{filename}'),
            max_workers=64,
            max_retry=500
        )