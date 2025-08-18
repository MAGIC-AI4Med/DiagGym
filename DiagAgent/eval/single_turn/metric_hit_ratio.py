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
        messages (list): List of dicts with role/content.
        model (str): Model name.
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


def build_examination_comparison_prompt(pred_exam, gt_exam_list):
    """
    Build examination comparison prompt.

    Args:
        pred_exam (str): Predicted examination name.
        gt_exam_list (list): List of ground truth examination names.

    Returns:
        str: Formatted prompt.
    """
    instruction = load_instruction("instructions/hit_ratio.txt")
    gt_exam_str = ", ".join(gt_exam_list)
    return instruction.format(pred_exam=pred_exam, gt_exam=gt_exam_str)


def extract_examination_name(model_output_text):
    """
    Extract examination name from model output.

    Args:
        model_output_text (str): Model output string.

    Returns:
        str: Extracted examination name.
    """
    if model_output_text is None:
        raise ValueError("Model output is None")

    model_output_list = model_output_text.split('\n')
    if len(model_output_list) < 2:
        raise ValueError("Invalid response format. Expected at least 2 lines.")

    request_text = model_output_list[1]
    if 'needed:' in request_text:
        return request_text.split('needed:')[1].replace('.', '').strip()
    elif 'performed:' in request_text:
        return request_text.split('performed:')[1].replace('.', '').strip()
    else:
        raise ValueError("Invalid response format. Expected 'needed:' or 'performed:'")


# ======== Dataset Loading ========
def load_dataset(data_path='testset_with_raw_data.json'):
    """
    Load dataset from JSON and prepare exams_dict for each case.

    Returns:
        list: Dataset with 'exams_dict' and 'text'.
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


def build_key_exams_mapping(testset_data):
    """
    Build mapping of note_id -> list of key exams.

    Returns:
        dict: Mapping note_id to list of key exams.
    """
    return {item['note_id']: list(item.get('exams_dict', {}).keys()) for item in testset_data}


# ======== Multithreaded Processing ========
def process_single_prompt(prompt_data, max_retry=5):
    """
    Process a single GPT prompt with retries.

    Args:
        prompt_data (tuple): (item_idx, turn_idx, task_type, prompt)
        max_retry (int): Max retry attempts.

    Returns:
        tuple: ((item_idx, turn_idx, task_type), result)
    """
    item_idx, turn_idx, task_type, prompt = prompt_data

    for i in range(max_retry):
        try:
            result = workflow([{"role": "user", "content": prompt}])
            return (item_idx, turn_idx, task_type), result
        except Exception as e:
            if i == max_retry - 1:
                print(f"[ERROR] item {item_idx} turn {turn_idx} task {task_type}: {e}")
                return (item_idx, turn_idx, task_type), f"ERROR: {e}"
            time.sleep(2 + i * 2)


def send_gpt_multithread(prompts, max_workers=4, max_retry=5):
    """
    Send multiple prompts to GPT with multithreading.

    Args:
        prompts (list): List of (item_idx, turn_idx, task_type, prompt)
        max_workers (int): Number of threads.
        max_retry (int): Maximum retry attempts.

    Returns:
        dict: Mapping (item_idx, turn_idx, task_type) -> result.
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
                item_idx, turn_idx, task_type, _ = future_to_prompt[future]
                print(f"[THREAD ERROR] item {item_idx} turn {turn_idx} task {task_type}: {e}")
                results[(item_idx, turn_idx, task_type)] = f"THREAD ERROR: {e}"
    return results


# ======== Evaluation ========
def evaluate_examination_comparison_multithread(json_filepath, save_filepath, key_exams_mapping, max_workers=4, max_retry=5):
    """
    Evaluate examination name accuracy.

    Args:
        json_filepath (str): Path to generated responses JSON.
        save_filepath (str): Path to save evaluation results.
        key_exams_mapping (dict): Mapping note_id -> list of key exams.
    """
    with open(json_filepath, 'r') as f:
        json_data = json.load(f)

    eval_results = {}
    prompts = []

    for result_item in json_data:
        item_idx = result_item['item_idx']
        turn_idx = result_item['turn_idx']
        note_id = result_item['note_id']
        generated_response = result_item['generated_response']
        is_final_turn = False

        if is_final_turn:
            continue

        try:
            pred_examination = extract_examination_name(generated_response)
            if note_id in key_exams_mapping and key_exams_mapping[note_id]:
                prompt = build_examination_comparison_prompt(pred_examination, key_exams_mapping[note_id])
                prompts.append((item_idx, turn_idx, "examination_comparison", prompt))
            else:
                eval_results[f"item_{item_idx}_turn_{turn_idx}"] = {
                    "item_idx": item_idx,
                    "turn_idx": turn_idx,
                    "note_id": note_id,
                    "predicted_exam": pred_examination,
                    "key_exams": key_exams_mapping.get(note_id, []),
                    "examination_comparison_result": "unfound - no key exams"
                }
        except Exception as e:
            eval_results[f"item_{item_idx}_turn_{turn_idx}"] = {
                "item_idx": item_idx,
                "turn_idx": turn_idx,
                "note_id": note_id,
                "predicted_exam": "extraction_failed",
                "key_exams": key_exams_mapping.get(note_id, []),
                "examination_comparison_result": "unfound - extraction failed"
            }

    print(f"Using {max_workers} threads for {len(prompts)} prompts...")
    print(f"{len(eval_results)} items skipped due to missing exams.")

    if prompts:
        results = send_gpt_multithread(prompts, max_workers=max_workers, max_retry=max_retry)
        for (item_idx, turn_idx, _), result in results.items():
            note_id = next((item['note_id'] for item in json_data if item['item_idx'] == item_idx and item['turn_idx'] == turn_idx), f"unknown_{item_idx}")
            try:
                predicted_exam = extract_examination_name(
                    next(item['generated_response'] for item in json_data if item['item_idx'] == item_idx and item['turn_idx'] == turn_idx)
                )
            except:
                predicted_exam = "extraction_failed"

            eval_results[f"item_{item_idx}_turn_{turn_idx}"] = {
                "item_idx": item_idx,
                "turn_idx": turn_idx,
                "note_id": note_id,
                "predicted_exam": predicted_exam,
                "key_exams": key_exams_mapping.get(note_id, []),
                "examination_comparison_result": result
            }

    with open(save_filepath, 'w') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {save_filepath}, total {len(eval_results)} entries.")


# ======== Main ========
if __name__ == "__main__":
    print("Loading dataset and building key exams mapping...")
    testset_data = load_dataset('testset_with_raw_data.json')
    key_exams_mapping = build_key_exams_mapping(testset_data)
    print(f"Built mapping for {len(key_exams_mapping)} cases.")

    os.makedirs('metric_results_v2', exist_ok=True)

    filenames = []
    exists_names = os.listdir('metric_results_v2')
    for filename in os.listdir('outputs_v2'):
        if f'exam_comparison_v3_{filename}' not in exists_names and 'tmp' not in filename and filename.startswith('process_exam'):
            filenames.append(filename)

    print(f"Found {len(filenames)} files to process.")

    for filename in filenames:
        print(f"Processing {filename}...")
        try:
            evaluate_examination_comparison_multithread(
                json_filepath=os.path.join('outputs_v2', filename),
                save_filepath=os.path.join('metric_results_v2', f'exam_comparison_v3_{filename}'),
                key_exams_mapping=key_exams_mapping,
                max_workers=128,
                max_retry=5000
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("All files processed!")