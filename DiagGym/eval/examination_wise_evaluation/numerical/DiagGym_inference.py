import json
import numpy as np
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import random
import os
import time
from pathlib import Path

# ==============================
# Global Configuration
# ==============================
random.seed(42)

SEP = "<SEP>"
stop_tokens = [SEP, "<endoftext>"]

API_KEY = "YOUR_API_KEY"  # Replace with your OpenAI API key
API_BASE_URL = "http://localhost:8079/v1"  # Replace if using a different API endpoint


# ==============================
# Utility Functions
# ==============================
def load_all_numberic_results():
    """
    Load reference ranges for all numeric lab events from a JSON file.
    Returns:
        dict: Mapping of event name to a list of reference sub-events.
    """
    events = {}
    filepath = "reference_range.json"
    with open(filepath, 'r') as fp:
        json_data = json.load(fp)

    for event_name in json_data.keys():
        events[event_name] = []
        reference_ranges = json_data[event_name].split('\n\n')
        for reference_range in reference_ranges:
            reference_range = reference_range.split('\n')[0].replace('Sub Lab Events: ', '').strip()
            events[event_name].append(reference_range)
    return events


# ==============================
# EHR Generator Class
# ==============================
class EHRGenerator:
    """
    Wrapper for generating EHR text outputs using the specified language model.
    """
    def __init__(self, model_name_or_path, api_key=API_KEY, api_base=API_BASE_URL) -> None:
        self.model_name_or_path = model_name_or_path
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def inference(self, context, past_events_list, exam_name):
        """
        Perform inference given a context and exam name.
        Implements a retry mechanism for robustness.
        """
        if len(past_events_list) == 0:
            input_prompt = context + "Exam name:\n" + exam_name + "\nExam results:\n"
        else:
            input_prompt = context + SEP.join(past_events_list) + SEP + "Exam name:\n" + exam_name + "\nExam results:\n"
        
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
                print(f"API request failed, retrying... Error: {e}")
                if i == 2:
                    return None
        return None


# ==============================
# Data Organization
# ==============================
def organize_events_by_time(events):
    """
    Organize events by their timestamp.
    Args:
        events (list): List of event dictionaries.
    Returns:
        list: List of grouped events sorted by time.
    """
    time_grouped = {}
    for event in events:
        time = event["time"]
        if time not in time_grouped:
            time_grouped[time] = []
        time_grouped[time].append((event['data']['label'], event['data']['lab_data']))
    
    sorted_times = sorted(time_grouped.keys())
    return [time_grouped[t] for t in sorted_times]


def process_single_inference_ehrgen(args):
    """
    Process a single inference task for EHR generation.
    """
    sample, event_name, target_subevents, original_idx, model_name_or_path = args
    ehr_generator = EHRGenerator(model_name_or_path)
    
    result = ehr_generator.inference(sample['context'], [], event_name)
    
    return {
        'original_idx': original_idx,
        'event_name': event_name,
        'case_id': sample['case_id'],
        'prediction': result
    }


# ==============================
# Diversity Analyzer
# ==============================
class DiversityAnalyzer:
    """
    Class for analyzing diversity of numeric values in EHR data.
    """
    def __init__(self, model_name_or_path="EHRGenerator"):
        self.model_name_or_path = model_name_or_path
        self.target_events = load_all_numberic_results()
        print(f"Loaded {len(self.target_events)} lab events.")

    def extract_numeric_value_from_text(self, text, target_subevent):
        """
        Extract numeric value for a given subevent from text.
        """
        if not text or not target_subevent:
            return None
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if target_subevent.lower() in line.lower():
                numeric_match = re.search(r'Numeric Value:\s*([+-]?\d*\.?\d+)', line)
                if numeric_match:
                    try:
                        return float(numeric_match.group(1))
                    except ValueError:
                        continue
        return None

    def extract_ground_truth_values(self, dataset_path):
        """
        Extract numeric ground truth values from dataset.
        """
        print("Extracting ground truth values...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        ground_truth_values = defaultdict(lambda: defaultdict(list))
        for case_id, case_data in tqdm(data.items(), desc="Scanning ground truth"):
            if "events" not in case_data:
                continue
            for event in case_data["events"]:
                if event['source'] != 'lab':
                    continue
                event_name = event['data'].get('label')
                if event_name not in self.target_events:
                    continue
                lab_data = event['data'].get('lab_data', '')
                for target_subevent in self.target_events[event_name]:
                    value = self.extract_numeric_value_from_text(lab_data, target_subevent)
                    if value is not None:
                        ground_truth_values[event_name][target_subevent].append(value)
        
        total_values = sum(len(values) for event_values in ground_truth_values.values() 
                           for values in event_values.values())
        print(f"Extracted {total_values} ground truth values.")
        return dict(ground_truth_values)

    def run_model_inference(self, dataset_path, sample_limit_per_event=50):
        """
        Run EHRGenerator inference for all target events using multithreading.
        """
        print("Running EHRGenerator inference...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        event_samples = defaultdict(list)
        for case_id, case_data in data.items():
            if "events" not in case_data or "reformat_physical_exam" not in case_data:
                continue
            case_events = set()
            for event in case_data["events"]:
                if event['source'] == 'lab':
                    event_name = event['data'].get('label')
                    if event_name in self.target_events:
                        case_events.add(event_name)
            for event_name in case_events:
                if len(event_samples[event_name]) < sample_limit_per_event:
                    discharge_diagnosis = case_data["discharge_diagnosis"]
                    discharge_note = case_data["text"]
                    context = discharge_note.split('Physical Exam:')[0].strip() + \
                              "\n\nFinal Diagnosis:\n" + discharge_diagnosis + \
                              "\nThe following summarizes the results from the patient's medical examination:\n"
                    event_samples[event_name].append({
                        'case_id': case_id,
                        'context': context
                    })
        
        all_tasks = []
        task_idx = 0
        for event_name, samples in event_samples.items():
            target_subevents = self.target_events[event_name]
            for sample in samples:
                all_tasks.append((sample, event_name, target_subevents, task_idx, self.model_name_or_path))
                task_idx += 1
        
        inference_results = defaultdict(list)
        completed_tasks = [None] * len(all_tasks)
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_task = {executor.submit(process_single_inference_ehrgen, task_args): task_args
                              for task_args in all_tasks}
            for future in tqdm(as_completed(future_to_task), total=len(all_tasks), desc="EHRGenerator Inference"):
                result = future.result()
                if result and result['prediction']:
                    completed_tasks[result['original_idx']] = result
                    inference_results[result['event_name']].append({
                        'case_id': result['case_id'],
                        'prediction': result['prediction']
                    })
        
        total_predictions = sum(len(predictions) for predictions in inference_results.values())
        print(f"Completed {total_predictions} valid inferences.")
        return dict(inference_results)

    def extract_predicted_values(self, inference_results):
        """
        Extract numeric values from model predictions.
        """
        print("Extracting predicted values...")
        predicted_values = defaultdict(lambda: defaultdict(list))
        for event_name, predictions in inference_results.items():
            for pred_data in predictions:
                prediction_text = pred_data['prediction']
                for target_subevent in self.target_events[event_name]:
                    value = self.extract_numeric_value_from_text(prediction_text, target_subevent)
                    if value is not None:
                        predicted_values[event_name][target_subevent].append(value)
        
        total_values = sum(len(values) for event_values in predicted_values.values() 
                           for values in event_values.values())
        print(f"Extracted {total_values} predicted values.")
        return dict(predicted_values)

    def calculate_normalized_variance(self, values_dict, data_type="Data"):
        """
        Calculate normalized variance for numeric values.
        """
        print(f"Calculating normalized variance for {data_type}...")
        normalized_variances = {}
        all_norm_vars = []
        for event_name, subevents in values_dict.items():
            event_norm_vars = {}
            for subevent, values in subevents.items():
                if len(values) >= 3:
                    mean_val = np.mean(values)
                    var_val = np.var(values, ddof=1)
                    if mean_val != 0:
                        norm_var = var_val / (mean_val ** 2)
                        event_norm_vars[subevent] = {
                            'count': len(values),
                            'mean': mean_val,
                            'variance': var_val,
                            'normalized_variance': norm_var,
                            'std': np.std(values, ddof=1),
                            'cv': np.std(values, ddof=1) / mean_val
                        }
                        all_norm_vars.append(norm_var)
            if event_norm_vars:
                normalized_variances[event_name] = event_norm_vars
        overall_norm_var = np.mean(all_norm_vars) if all_norm_vars else 0
        print(f"{data_type} overall normalized variance: {overall_norm_var:.6f}")
        return normalized_variances, overall_norm_var

    def save_inference_results(self, inference_results):
        """
        Save raw inference results to file.
        """
        os.makedirs("inference_outputs", exist_ok=True)
        inference_data = {
            'model_name': self.model_name_or_path,
            'total_predictions': sum(len(predictions) for predictions in inference_results.values()),
            'event_count': len(inference_results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': inference_results
        }
        inference_file_path = "inference_outputs/EHRGenerator_inference_results.json"
        with open(inference_file_path, 'w') as f:
            json.dump(inference_data, f, indent=4, ensure_ascii=False)
        print(f"Inference results saved to: {inference_file_path}")
        return inference_file_path

    def save_results(self, gt_norm_vars, pred_norm_vars, gt_overall, pred_overall):
        """
        Save diversity analysis results to file.
        """
        os.makedirs("diversity_results", exist_ok=True)
        results = {
            'ground_truth': {
                'overall_normalized_variance': gt_overall,
                'event_details': gt_norm_vars
            },
            'predictions': {
                'overall_normalized_variance': pred_overall,
                'event_details': pred_norm_vars
            },
            'summary': {
                'gt_overall_nv': gt_overall,
                'pred_overall_nv': pred_overall,
                'diversity_ratio': pred_overall / gt_overall if gt_overall > 0 else None
            },
            'model_name': self.model_name_or_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        results_file_path = "diversity_results/EHRGenerator_analysis.json"
        with open(results_file_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Diversity analysis results saved to: {results_file_path}")
        return results_file_path

    def run_analysis(self, dataset_path, sample_limit_per_event=30):
        """
        Run complete diversity analysis pipeline.
        """
        print("Starting diversity analysis...")
        ground_truth_values = self.extract_ground_truth_values(dataset_path)
        gt_norm_vars, gt_overall = self.calculate_normalized_variance(ground_truth_values, "Ground Truth")
        inference_results = self.run_model_inference(dataset_path, sample_limit_per_event)
        inference_file_path = self.save_inference_results(inference_results)
        predicted_values = self.extract_predicted_values(inference_results)
        pred_norm_vars, pred_overall = self.calculate_normalized_variance(predicted_values, "Predictions")
        results_file_path = self.save_results(gt_norm_vars, pred_norm_vars, gt_overall, pred_overall)
        print(f"Files saved:\n Inference: {inference_file_path}\n Analysis: {results_file_path}")
        return gt_overall, pred_overall


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    analyzer = DiversityAnalyzer()
    dataset_path = "test_data.json"  # Change to your dataset path
    gt_nv, pred_nv = analyzer.run_analysis(dataset_path, sample_limit_per_event=5000)
    print("Analysis complete.")
    print(f"Ground truth overall normalized variance: {gt_nv:.6f}")
    print(f"Predictions overall normalized variance: {pred_nv:.6f}")