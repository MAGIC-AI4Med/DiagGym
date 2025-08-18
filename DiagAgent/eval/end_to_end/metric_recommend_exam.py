# -*- coding: utf-8 -*-
import os
import json
import time
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ======== OpenAI API Setup ========
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")

client = OpenAI(api_key=API_KEY)

def workflow(messages, model="gpt-4o", temperature=0.0, max_tokens=512):
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


# ======== Prompt Builders (unchanged) ========
def build_precision_prompt(key_exam_names, recommended_exam_names):
    """Build prompt for precision calculation (do not modify)."""
    prompt = f"""
Please determine how many of the recommended exams appear in the key exam list. Note that even if the expressions are different, if they refer to the same examination, they should be considered as matches.

Key exam list: {key_exam_names}
Recommended exam list: {recommended_exam_names}

Please analyze each item in the recommended exam list and determine if it has a corresponding item in the key exam list (even with different expressions).

Please only output the number of matches as an integer. For example: 3
"""
    return prompt

def build_recall_prompt(key_exam_names, recommended_exam_names):
    """Build prompt for recall calculation (do not modify)."""
    prompt = f"""
Please determine how many of the key exams appear in the recommended exam list. Note that even if the expressions are different, if they refer to the same examination, they should be considered as matches.

Key exam list: {key_exam_names}
Recommended exam list: {recommended_exam_names}

Please analyze each item in the key exam list and determine if it has a corresponding item in the recommended exam list (even with different expressions).

Please only output the number of matches as an integer. For example: 2
"""
    return prompt


# ======== Exam Extraction ========
def extract_examination_name(model_output_text):
    """Extract examination name from model output string."""
    model_output_list = model_output_text.split('\n')
    if len(model_output_list) < 2:
        return None
    request_text = model_output_list[1]
    if 'needed:' in request_text:
        return request_text.split('needed:')[1].replace('.', '').strip()
    elif 'performed:' in request_text:
        return request_text.split('performed:')[1].replace('.', '').strip()
    return None

def extract_all_recommended_exams(messages):
    """
    Extract all recommended exams from assistant messages.
    Returns a list of exam names.
    """
    recommended_exams = []
    for message in messages:
        if message.get('role') == 'assistant':
            try:
                exam_name = extract_examination_name(message['content'])
                if exam_name:
                    if ',' in exam_name:
                        exams = [exam.strip() for exam in exam_name.split(',') if exam.strip()]
                        recommended_exams.extend(exams)
                    else:
                        recommended_exams.append(exam_name)
            except Exception:
                continue
    return recommended_exams


# ======== Multithreaded Processing ========
def process_single_prompt(prompt_data, max_retry=5):
    """
    Process a single GPT prompt with retries.

    Args:
        prompt_data (tuple): (item_idx, exam_idx, task_type, prompt)
        max_retry (int): Maximum retry attempts.

    Returns:
        tuple: ((item_idx, exam_idx, task_type), result)
    """
    item_idx, exam_idx, task_type, prompt = prompt_data

    for i in range(max_retry):
        try:
            result_text = workflow([{"role": "user", "content": prompt}])
            numbers = re.findall(r'\d+', result_text.strip())
            result_value = int(numbers[0]) if numbers else 0
            return (item_idx, exam_idx, task_type), result_value
        except Exception as e:
            if i == max_retry - 1:
                print(f"[ERROR] item {item_idx} exam {exam_idx} task {task_type}: {e}")
                return (item_idx, exam_idx, task_type), f"ERROR: {e}"
            time.sleep(2 + i * 2)


def send_gpt_multithread(prompts, max_workers=4, max_retry=5):
    """
    Process multiple prompts in parallel using multithreading.

    Args:
        prompts (list): List of (item_idx, exam_idx, task_type, prompt)
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


# ======== Precision & Recall Evaluation ========
def evaluate_precision_recall_multithread(json_filepath, save_filepath, max_workers=4, max_retry=5):
    """
    Evaluate precision and recall for exam recommendations.
    """
    with open(json_filepath, 'r') as f:
        json_data = json.load(f)

    eval_results = {}
    prompts = []

    for item_idx, item in enumerate(json_data):
        note_id = item['note_id']
        messages = item['messages']
        key_exam_names = item.get('key_exam_names', [])

        if key_exam_names:
            recommended_exam_names = extract_all_recommended_exams(messages)
            if recommended_exam_names:
                precision_prompt = build_precision_prompt(key_exam_names, recommended_exam_names)
                prompts.append((item_idx, note_id, "precision", precision_prompt))

                recall_prompt = build_recall_prompt(key_exam_names, recommended_exam_names)
                prompts.append((item_idx, note_id, "recall", recall_prompt))

    print(f"Using {max_workers} threads to process {len(prompts)} prompts...")

    results = send_gpt_multithread(prompts, max_workers=max_workers, max_retry=max_retry)

    for (item_idx, note_id, task_type), result in results.items():
        if note_id not in eval_results:
            eval_results[note_id] = {}
        eval_results[note_id][task_type] = result

    # Compute precision and recall scores
    for note_id in eval_results:
        if 'precision' in eval_results[note_id] and 'recall' in eval_results[note_id]:
            item = next(item for item in json_data if item['note_id'] == note_id)
            key_exam_names = item.get('key_exam_names', [])
            recommended_exam_names = extract_all_recommended_exams(item['messages'])

            precision_matches = eval_results[note_id]['precision']
            recall_matches = eval_results[note_id]['recall']

            if isinstance(precision_matches, int) and len(recommended_exam_names) > 0:
                eval_results[note_id]['precision_score'] = precision_matches / len(recommended_exam_names)
                eval_results[note_id]['precision_matches'] = precision_matches
                eval_results[note_id]['recommended_count'] = len(recommended_exam_names)
                eval_results[note_id]['recommended_exams'] = recommended_exam_names

            if isinstance(recall_matches, int) and len(key_exam_names) > 0:
                eval_results[note_id]['recall_score'] = recall_matches / len(key_exam_names)
                eval_results[note_id]['recall_matches'] = recall_matches
                eval_results[note_id]['key_count'] = len(key_exam_names)
                eval_results[note_id]['key_exams'] = key_exam_names

    with open(save_filepath, 'w') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    print(f"Precision and Recall evaluation saved to {save_filepath}")

    # Overall statistics
    precision_scores = [v['precision_score'] for v in eval_results.values() if 'precision_score' in v]
    recall_scores = [v['recall_score'] for v in eval_results.values() if 'recall_score' in v]

    print(f"\n=== Overall Statistics ===")
    print(f"Total evaluated cases: {len(eval_results)}")
    if precision_scores:
        avg_precision = sum(precision_scores) / len(precision_scores)
        print(f"Average Precision: {avg_precision:.4f} ({len(precision_scores)} cases)")
    if recall_scores:
        avg_recall = sum(recall_scores) / len(recall_scores)
        print(f"Average Recall: {avg_recall:.4f} ({len(recall_scores)} cases)")
    if precision_scores and recall_scores and (avg_precision + avg_recall) > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        print(f"F1 Score: {f1_score:.4f}")


# ======== Main ========
if __name__ == "__main__":
    filenames = []
    exists_names = os.listdir('metric_results')
    for filename in os.listdir('outputs'):
        if f'precision_recall_{filename}' not in exists_names and 'tmp' not in filename:
            filenames.append(filename)

    for filename in filenames:
        evaluate_precision_recall_multithread(
            json_filepath=os.path.join('outputs', filename),
            save_filepath=os.path.join('metric_results', f'precision_recall_{filename}'),
            max_workers=64,
            max_retry=500
        )