# -*- coding: utf-8 -*-
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Read API credentials from environment variables
API_KEY = os.getenv("API_KEY", "YOUR_API_KEY_HERE")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def workflow_gpt4o(messages, model="gpt-4o", temperature=0.0, max_tokens=2048):
    """
    Send messages to GPT-4o and return the generated content.
    Using a workflow-like function instead of async.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def get_gpt4_result(prompt, model="gpt-4o", temperature=0.0):
    """Helper function to send a single prompt to GPT-4o."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant for medical text evaluation."},
        {"role": "user", "content": prompt}
    ]
    return workflow_gpt4o(messages, model=model, temperature=temperature)


def load_instruction(txt_path):
    """Load instruction template from a text file."""
    with open(txt_path, encoding='utf-8') as fp:
        return fp.read()


def build_similarity_prompt(exam_name, ground_truth, prediction):
    """Build prompt for similarity evaluation."""
    instruction = load_instruction("instructions/similarity_evaluation.txt")
    return instruction.format(
        exam_name=exam_name,
        ground_truth=ground_truth,
        prediction=prediction
    )


def build_fidelity_fullchain_prompt(case_summary, predicted_chain, ground_truth_chain):
    """Build prompt for fullchain fidelity evaluation."""
    instruction = load_instruction("instructions/fidelity_fullchain_evaluation.txt")
    predicted_chain_text = ""
    for i, (exam_name, result) in enumerate(predicted_chain, 1):
        predicted_chain_text += f"{i}. **{exam_name}**: {result}\n"
    ground_truth_chain_text = ""
    for i, (exam_name, result) in enumerate(ground_truth_chain, 1):
        ground_truth_chain_text += f"{i}. **{exam_name}**: {result}\n"
    return instruction.format(
        case_summary=case_summary,
        predicted_chain=predicted_chain_text,
        ground_truth_chain=ground_truth_chain_text
    )


def gpt4_task(args):
    """Single task wrapper for multithreading."""
    item_idx, exam_idx, task_type, prompt, max_retry, model = args
    for i in range(max_retry):
        try:
            result = get_gpt4_result(prompt, model=model)
            return (item_idx, exam_idx, task_type, result)
        except Exception as e:
            if i == max_retry - 1:
                print(f"[ERROR] item {item_idx} exam {exam_idx} task {task_type}: {e}")
                return (item_idx, exam_idx, task_type, f"ERROR: {e}")
            time.sleep(1)


def evaluate_similarity_async(json_filepath, save_filepath, model="gpt-4o", temp=0.0, max_retry=5, max_workers=32):
    """Evaluate similarity for each prediction asynchronously."""
    with open(json_filepath, 'r') as f:
        json_data = json.load(f)
    samples = json_data['results'] if 'results' in json_data else json_data

    prompts = []
    for item_idx, item in enumerate(samples):
        preds = item.get("predictions", [])
        for exam_idx, exam in enumerate(preds):
            exam_name = exam['exam_name']
            prediction = exam['prediction']
            ground_truth = exam['ground_truth']
            prompt = build_similarity_prompt(exam_name, ground_truth, prediction)
            prompts.append((item_idx, exam_idx, "similarity", prompt, max_retry, model))

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(gpt4_task, args) for args in prompts]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Similarity Evaluating"):
            item_idx, exam_idx, task_type, result = f.result()
            results[(item_idx, exam_idx, task_type)] = result

    for (item_idx, exam_idx, task_type), result in results.items():
        if task_type == "similarity":
            samples[item_idx]['predictions'][exam_idx]['gpt_eval_similarity'] = result

    with open(save_filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
    print(f"Similarity evaluation saved to {save_filepath}")


def evaluate_fidelity_fullchain_async(json_filepath, save_filepath, model="gpt-4o", temp=0.0, max_retry=5, max_workers=32):
    """Evaluate fullchain fidelity asynchronously."""
    with open(json_filepath, 'r') as f:
        json_data = json.load(f)
    samples = json_data['results'] if 'results' in json_data else json_data

    prompts = []
    for item_idx, item in enumerate(samples):
        context = item['context']
        preds = item.get("predictions", [])
        predicted_chain = [(p['exam_name'], p['prediction']) for p in preds]
        ground_truth_chain = [(p['exam_name'], p['ground_truth']) for p in preds]
        prompt = build_fidelity_fullchain_prompt(context, predicted_chain, ground_truth_chain)
        prompts.append((item_idx, -1, "fidelity_fullchain", prompt, max_retry, model))

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(gpt4_task, args) for args in prompts]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Fullchain Evaluating"):
            item_idx, exam_idx, task_type, result = f.result()
            results[(item_idx, exam_idx, task_type)] = result

    for (item_idx, exam_idx, task_type), result in results.items():
        if task_type == "fidelity_fullchain":
            samples[item_idx]['gpt_eval_fidelity_fullchain'] = result

    with open(save_filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
    print(f"Fidelity fullchain evaluation saved to {save_filepath}")


if __name__ == "__main__":
    results_dir = 'inference_code/results'
    metric_dir = 'inference_code/metric_results'
    os.makedirs(metric_dir, exist_ok=True)

    filenames = [filename for filename in os.listdir(results_dir) if 'gpt4o' in filename]

    for filename in filenames:
        if 'stepwise' in filename:
            evaluate_similarity_async(
                json_filepath=os.path.join(results_dir, filename),
                save_filepath=os.path.join(metric_dir, 'similarity_' + filename),
                model="gpt-4o",
                temp=0.0,
                max_retry=5,
                max_workers=32
            )
    for filename in filenames:
        if 'fullchain' in filename:
            evaluate_fidelity_fullchain_async(
                json_filepath=os.path.join(results_dir, filename),
                save_filepath=os.path.join(metric_dir, 'passrate_' + filename),
                model="gpt-4o",
                temp=0.0,
                max_retry=5,
                max_workers=32
            )