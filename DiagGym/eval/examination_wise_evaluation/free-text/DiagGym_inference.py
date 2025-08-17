import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from openai import OpenAI
import random

# ============================================
# Global Configuration
# ============================================
random.seed(42)
SEP = "<SEP>"
stop_tokens = [SEP, "<endoftext>"]

API_KEY = "YOUR_API_KEY"  # Replace with your OpenAI API key
API_BASE_URL = "http://localhost:8079/v1"  # Replace if using a different API endpoint


# ============================================
# Utility Functions
# ============================================
def load_data(filename):
    """
    Load dataset from a JSON file.
    Args:
        filename (str): Path to JSON file.
    Returns:
        dict: Loaded dataset.
    """
    print(f"[INFO] Loading data from: {filename}")
    with open(filename, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    print(f"[INFO] Loaded {len(data)} cases.")
    return data


def extract_top5_radiology_exams(filename):
    """
    Extract the top 5 most frequent radiology exams from the dataset.
    Args:
        filename (str): Path to the dataset JSON file.
    Returns:
        list: Top 5 radiology exam names.
    """
    print("[INFO] Extracting Top 5 Frequent Radiology Exam Names...")

    json_data = load_data(filename)
    radiology_counter = Counter()

    for note_id in json_data.keys():
        for event in json_data[note_id]['events']:
            if event['source'] == 'radiology':
                exam_name = event['data'].get('label', 'Unknown').strip()
                radiology_counter[exam_name] += 1

    top_5_radiology = radiology_counter.most_common(5)

    print("[INFO] Top 5 Most Frequent Radiology Exams:")
    top_5_names = []
    for rank, (exam_name, count) in enumerate(top_5_radiology, 1):
        print(f"  {rank}. {exam_name}: {count}")
        top_5_names.append(exam_name)

    return top_5_names


# ============================================
# EHR Generator Class
# ============================================
class EHRGenerator:
    """
    Wrapper for generating EHR outputs using a language model.
    """
    def __init__(self, model_name_or_path, api_key=API_KEY, api_base=API_BASE_URL) -> None:
        self.model_name_or_path = model_name_or_path
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def inference(self, context, exam_name):
        """
        Perform inference using only context and exam_name (no past events).
        Implements a retry mechanism for robustness.
        """
        input_prompt = context + "Exam name:\n" + exam_name + "\nExam results:\n"

        for i in range(3):  # Retry up to 3 times
            try:
                response = self.client.completions.create(
                    model=self.model_name_or_path,
                    prompt=input_prompt,
                    max_tokens=8192,
                    temperature=1.0,
                    stop=stop_tokens
                )
                generated_text = response.choices[0].text.strip()
                return generated_text
            except Exception as e:
                print(f"[ERROR] API request failed, retrying... Error: {e}")
                if i == 2:
                    return None
                time.sleep(1)
        return None


def process_single_inference(args):
    """
    Process a single inference task for radiology generation.
    """
    sample, exam_name, model_name_or_path = args
    ehr_generator = EHRGenerator(model_name_or_path)
    result = ehr_generator.inference(sample['context'], exam_name)

    return {
        'exam_name': exam_name,
        'case_id': sample['case_id'],
        'context': sample['context'],
        'prediction': result,
        'original_result': sample.get('original_result', '')
    }


# ============================================
# Radiology Generator Class
# ============================================
class RadiologyGenerator:
    """
    Class for generating radiology exam results for top N frequent exams.
    """
    def __init__(self, model_name_or_path="EHRGenerator"):
        self.model_name_or_path = model_name_or_path
        print(f"[INFO] RadiologyGenerator initialized with model: {model_name_or_path}")

    def collect_exam_specific_cases(self, dataset_path, top5_exam_names):
        """
        Collect cases for each target radiology exam.
        Args:
            dataset_path (str): Path to dataset file.
            top5_exam_names (list): List of top 5 exam names.
        Returns:
            dict: Mapping {exam_name: [case_samples]}.
        """
        print("[INFO] Collecting cases for each target exam...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        exam_cases = defaultdict(list)
        for case_id, case_data in data.items():
            if "text" not in case_data or "discharge_diagnosis" not in case_data or "events" not in case_data:
                continue

            discharge_diagnosis = case_data["discharge_diagnosis"]
            discharge_note = case_data["text"]
            context = discharge_note.split('Physical Exam:')[0].strip() + "\n\nFinal Diagnosis:\n" + discharge_diagnosis
            context += "\nThe following summarizes the results from the patient's medical examination:\n"

            case_radiology_exams = {}
            for event in case_data["events"]:
                if event['source'] == 'radiology':
                    exam_name = event['data'].get('label', 'Unknown').strip()
                    if exam_name in top5_exam_names:
                        original_result = event['data'].get('lab_data', '')
                        case_radiology_exams[exam_name] = original_result

            for exam_name in case_radiology_exams:
                exam_cases[exam_name].append({
                    'case_id': case_id,
                    'context': context,
                    'original_result': case_radiology_exams[exam_name]
                })

        print("[INFO] Exam case counts:")
        total_cases = sum(len(exam_cases[exam]) for exam in top5_exam_names)
        for exam_name in top5_exam_names:
            print(f"  {exam_name}: {len(exam_cases[exam_name])} cases")
        print(f"[INFO] Total: {total_cases} case-exam pairs")

        return dict(exam_cases)

    def run_radiology_generation(self, dataset_path, top5_exam_names):
        """
        Generate results for top radiology exams using the model.
        """
        print("[INFO] Starting radiology generation...")
        print(f"[INFO] Target exams: {top5_exam_names}")

        exam_cases = self.collect_exam_specific_cases(dataset_path, top5_exam_names)

        generation_tasks = []
        for exam_name, cases in exam_cases.items():
            for case_sample in cases:
                generation_tasks.append((case_sample, exam_name, self.model_name_or_path))

        print(f"[INFO] Total {len(generation_tasks)} generation tasks prepared.")

        if not generation_tasks:
            print("[WARNING] No generation tasks found.")
            return {}

        generation_results = defaultdict(list)
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_task = {executor.submit(process_single_inference, task_args): task_args
                              for task_args in generation_tasks}
            for future in tqdm(as_completed(future_to_task), total=len(generation_tasks), desc="Radiology Generation"):
                result = future.result()
                if result and result['prediction']:
                    generation_results[result['exam_name']].append({
                        'case_id': result['case_id'],
                        'context': result['context'],
                        'prediction': result['prediction'],
                        'original_result': result['original_result']
                    })

        total_generated = sum(len(results) for results in generation_results.values())
        print(f"[INFO] Completed {total_generated} successful generations.")

        for exam_name in top5_exam_names:
            original_count = len(exam_cases.get(exam_name, []))
            generated_count = len(generation_results.get(exam_name, []))
            success_rate = (generated_count / original_count * 100) if original_count > 0 else 0
            print(f"  {exam_name}: {generated_count}/{original_count} ({success_rate:.1f}%)")

        return dict(generation_results)

    def save_generation_results(self, generation_results, top5_exam_names):
        """
        Save generated results to a JSON file.
        """
        os.makedirs("radiology_generation_outputs", exist_ok=True)
        generation_data = {
            'model_name': self.model_name_or_path,
            'top5_exam_names': top5_exam_names,
            'total_generations': sum(len(results) for results in generation_results.values()),
            'exam_count': len(generation_results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'generation_settings': {
                'past_events': 'empty',
                'temperature': 1.0,
                'max_tokens': 8192,
                'method': 'case_specific'
            },
            'results': generation_results
        }
        full_results_path = "radiology_generation_outputs/EHRGenerator.json"
        with open(full_results_path, 'w', encoding='utf-8') as f:
            json.dump(generation_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Full generation results saved to: {full_results_path}")
        return full_results_path


# ============================================
# Main Execution
# ============================================
def main():
    """
    Main function for top radiology exam analysis and generation.
    """
    print("[INFO] Starting Top 5 Radiology Exam Analysis and Generation...")

    test_file = "test_data.json"  # Replace with your dataset path
    if not os.path.exists(test_file):
        print(f"[ERROR] Test file not found: {test_file}")
        return

    top_5_names = extract_top5_radiology_exams(test_file)
    if not top_5_names:
        print("[ERROR] No radiology exams found, exiting.")
        return

    generator = RadiologyGenerator()
    generation_results = generator.run_radiology_generation(test_file, top_5_names)
    results_path = generator.save_generation_results(generation_results, top_5_names)

    print("[INFO] Task completed.")
    print(f"[INFO] Output file: {results_path}")
    return top_5_names, generation_results


if __name__ == '__main__':
    top_5_radiology_names, results = main()