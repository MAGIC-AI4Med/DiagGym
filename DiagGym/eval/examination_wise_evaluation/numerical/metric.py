# -*- coding: utf-8 -*-
import json
import numpy as np
import re
import os
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import time


def load_all_numeric_results(reference_range_path):
    """
    Load all lab events with numeric values and their subevents.

    Args:
        reference_range_path (str): Path to reference_range.json.

    Returns:
        dict: {event_name: [subevent_names]}
    """
    events = {}
    with open(reference_range_path, 'r') as fp:
        json_data = json.load(fp)

    for event_name, ref_data in json_data.items():
        events[event_name] = []
        reference_ranges = ref_data.split('\n\n')
        for reference_range in reference_ranges:
            reference_range = reference_range.split('\n')[0].replace('Sub Lab Events: ', '').strip()
            events[event_name].append(reference_range)
    return events


def extract_units(trainset_path):
    """
    Extract units for each subevent from the training dataset.

    Args:
        trainset_path (str): Path to training dataset JSON.

    Returns:
        tuple: (event_units dict, missing_units set)
    """
    print("[INFO] Extracting units from training dataset...")
    with open(trainset_path, 'r') as f:
        data = json.load(f)

    event_units = defaultdict(lambda: defaultdict(set))
    for _, case_data in tqdm(data.items(), desc="Scanning units"):
        if "events" not in case_data:
            continue
        for event in case_data["events"]:
            if event['source'] != 'lab':
                continue
            event_name = event['data'].get('label')
            if not event_name:
                continue
            lab_data = event['data'].get('lab_data', '')
            for line in lab_data.split('\n'):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'^([^:]+):\s*.*?Units:\s*([^;]+)', line)
                if match:
                    subevent_name = match.group(1).strip()
                    unit = match.group(2).strip()
                    if subevent_name and unit:
                        event_units[event_name][subevent_name].add(unit)

    final_units = {}
    missing_units = set()
    for event_name, subevents in event_units.items():
        final_units[event_name] = {}
        for subevent_name, units_set in subevents.items():
            if units_set and len(units_set) == 1:
                final_units[event_name][subevent_name] = list(units_set)[0]
            else:
                missing_units.add(subevent_name)

    print(f"[INFO] Extracted units for {len(final_units)} events.")
    total_subevents = sum(len(subevents) for subevents in final_units.values())
    print(f"[INFO] Total subevents with unit info: {total_subevents}")
    print(f"[WARNING] Missing/inconsistent units for {len(missing_units)} subevents.")
    return final_units, missing_units


class InferenceResultsAnalyzer:
    """
    Analyzer for comparing ground truth and model inference results
    for numeric lab event subevents, computing diversity and distribution metrics.
    """
    def __init__(self,
                 inference_outputs_dir="inference_outputs",
                 min_count_threshold=1000,
                 reference_range_path="reference_range.json",
                 trainset_path="train_data.json"):
        self.inference_outputs_dir = inference_outputs_dir
        self.min_count_threshold = min_count_threshold
        self.target_events = load_all_numeric_results(reference_range_path)
        self.event_units, self.missing_units = extract_units(trainset_path)
        print(f"[INFO] Loaded {len(self.target_events)} target lab events.")
        print(f"[INFO] Min count threshold: {min_count_threshold}")
        print(f"[WARNING] Missing units for {len(self.missing_units)} subevents.")

    def should_skip_subevent(self, event_name, subevent_name):
        """Return True if subevent should be skipped due to missing unit info."""
        return (subevent_name in self.missing_units or
                event_name not in self.event_units or
                subevent_name not in self.event_units[event_name])

    def extract_numeric_value_from_output(self, text, target_subevent, is_original_label=False, event_name=None):
        """Extract numeric value for a subevent from text if it has unit info."""
        if not text or not target_subevent:
            return None
        if event_name and self.should_skip_subevent(event_name, target_subevent):
            return None
        lines = text.split('\n')
        if not is_original_label:
            for line in lines:
                if target_subevent.lower() in line.lower():
                    numeric_match = re.search(r'Numeric Value:\s*([+-]?\d*\.?\d+)', line)
                    if numeric_match:
                        try:
                            return float(numeric_match.group(1))
                        except ValueError:
                            continue
        else:
            for line in lines:
                if line.lower().startswith('value'):
                    numeric_match = re.search(r'Numeric Value:\s*([+-]?\d*\.?\d+)', line)
                    if numeric_match:
                        try:
                            return float(numeric_match.group(1))
                        except ValueError:
                            continue
        return None

    def load_all_inference_results(self):
        """Load all model inference results from directory."""
        all_models_results = {}
        if not os.path.exists(self.inference_outputs_dir):
            print(f"[ERROR] Directory not found: {self.inference_outputs_dir}")
            return all_models_results
        for filename in os.listdir(self.inference_outputs_dir):
            if filename.endswith('_inference_results.json'):
                model_name = filename.replace('_inference_results.json', '')
                filepath = os.path.join(self.inference_outputs_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                all_models_results[model_name] = data['results']
                total_predictions = sum(len(predictions) for predictions in data['results'].values())
                print(f"[INFO] Loaded {total_predictions} predictions for model {model_name}")
        print(f"[INFO] Total models loaded: {len(all_models_results)}")
        return all_models_results

    def extract_ground_truth_values(self, dataset_path):
        """Extract numeric ground truth values for subevents with unit info."""
        print("[INFO] Extracting ground truth values...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        ground_truth_values = defaultdict(lambda: defaultdict(list))
        for _, case_data in tqdm(data.items(), desc="Scanning ground truth"):
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
                    if self.should_skip_subevent(event_name, target_subevent):
                        continue
                    value = self.extract_numeric_value_from_output(lab_data, target_subevent, False, event_name)
                    if value is not None:
                        ground_truth_values[event_name][target_subevent].append(value)
        total_values = sum(len(v) for ev in ground_truth_values.values() for v in ev.values())
        print(f"[INFO] Extracted {total_values} ground truth numeric values.")
        return dict(ground_truth_values)

    def extract_predicted_values_from_inference(self, inference_results, model_name):
        """Extract numeric predicted values for subevents with unit info."""
        print(f"[INFO] Extracting predicted values for model {model_name}...")
        predicted_values = defaultdict(lambda: defaultdict(list))
        for event_name, predictions in inference_results.items():
            if event_name not in self.target_events:
                continue
            for pred_data in predictions:
                prediction_text = pred_data['prediction']
                if not prediction_text:
                    continue
                for target_subevent in self.target_events[event_name]:
                    if self.should_skip_subevent(event_name, target_subevent):
                        continue
                    value = self.extract_numeric_value_from_output(
                        prediction_text,
                        target_subevent,
                        (event_name == target_subevent) and (model_name == "EHRGenerator"),
                        event_name
                    )
                    if value is not None:
                        predicted_values[event_name][target_subevent].append(value)
        total_values = sum(len(v) for ev in predicted_values.values() for v in ev.values())
        print(f"[INFO] Extracted {total_values} predicted numeric values.")
        return dict(predicted_values)

    def calculate_diversity_metrics(self, values_dict, data_type="Data"):
        """Calculate normalized variance as diversity metric."""
        print(f"[INFO] Calculating diversity metrics for {data_type}...")
        diversity_metrics = {}
        all_norm_vars = []
        for event_name, subevents in values_dict.items():
            event_metrics = {}
            for subevent, values in subevents.items():
                if len(values) >= 3:
                    mean_val = np.mean(values)
                    var_val = np.var(values, ddof=1)
                    if mean_val != 0:
                        norm_var = var_val / (mean_val ** 2)
                        event_metrics[subevent] = {
                            'count': len(values),
                            'mean': mean_val,
                            'variance': var_val,
                            'normalized_variance': norm_var,
                            'std': np.std(values, ddof=1),
                            'cv': np.std(values, ddof=1) / mean_val
                        }
                        all_norm_vars.append(norm_var)
            if event_metrics:
                diversity_metrics[event_name] = event_metrics
        overall_diversity = np.mean(all_norm_vars) if all_norm_vars else 0
        print(f"[INFO] Overall diversity for {data_type}: {overall_diversity:.6f}")
        return diversity_metrics, overall_diversity

    def calculate_wasserstein_distance(self, real_values, pred_values, min_samples=30):
        """
        Calculate Wasserstein distance for each subevent after standardizing by GT stats.
        Returns mean distance and detailed stats.
        """
        print("[INFO] Calculating Wasserstein distances...")
        subevent_distances = []
        subevent_details = {}
        for event_name, subevents in real_values.items():
            if event_name not in pred_values:
                continue
            for subevent, real_vals in subevents.items():
                if subevent not in pred_values[event_name]:
                    continue
                pred_vals = pred_values[event_name][subevent]
                if len(real_vals) < min_samples or len(pred_vals) < min_samples:
                    continue
                real_array = np.array(real_vals)
                pred_array = np.array(pred_vals)
                gt_mean = np.mean(real_array)
                gt_std = np.std(real_array, ddof=1)
                if gt_std <= 0:
                    continue
                real_std = (real_array - gt_mean) / gt_std
                pred_std = (pred_array - gt_mean) / gt_std
                distance = wasserstein_distance(real_std, pred_std)
                subevent_distances.append(distance)
                subevent_details[f"{event_name}_{subevent}"] = {
                    'wasserstein_distance': float(distance),
                    'real_count': len(real_vals),
                    'pred_count': len(pred_vals)
                }
        overall_distance = np.mean(subevent_distances) if subevent_distances else float('inf')
        print(f"[INFO] Mean Wasserstein distance: {overall_distance:.6f}")
        return overall_distance, {'subevent_details': subevent_details}

    def run_comprehensive_analysis(self, dataset_path):
        """
        Run complete analysis pipeline: load data, extract values, compute metrics.
        """
        print("[INFO] Running comprehensive analysis...")
        all_models_inference = self.load_all_inference_results()
        if not all_models_inference:
            print("[ERROR] No inference results found.")
            return
        ground_truth_values = self.extract_ground_truth_values(dataset_path)
        all_models_predicted_values = {
            model_name: self.extract_predicted_values_from_inference(results, model_name)
            for model_name, results in all_models_inference.items()
        }
        filtered_gt = ground_truth_values  # Filtering by valid subevents can be added here
        gt_div_metrics, gt_overall_div = self.calculate_diversity_metrics(filtered_gt, "Ground Truth")
        results = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'ground_truth': {'overall_diversity': gt_overall_div, 'detailed_stats': gt_div_metrics},
            'models': {}
        }
        for model_name, pred_values in all_models_predicted_values.items():
            pred_div_metrics, pred_overall_div = self.calculate_diversity_metrics(pred_values, f"{model_name} Predictions")
            wass_dist, wass_details = self.calculate_wasserstein_distance(filtered_gt, pred_values)
            results['models'][model_name] = {
                'diversity_metrics': {'overall': pred_overall_div, 'detailed_stats': pred_div_metrics},
                'distribution_distance': wass_details,
                'diversity_ratio': pred_overall_div / gt_overall_div if gt_overall_div > 0 else 0
            }
        return results


if __name__ == "__main__":
    analyzer = InferenceResultsAnalyzer(
        inference_outputs_dir="inference_outputs",
        min_count_threshold=500,
        reference_range_path="reference_range.json",
        trainset_path="train_data.json"
    )
    dataset_path = "train_data.json"
    results = analyzer.run_comprehensive_analysis(dataset_path)
    print("[INFO] Analysis complete.")