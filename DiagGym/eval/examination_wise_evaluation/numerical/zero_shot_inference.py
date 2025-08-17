#!/usr/bin/env python3
"""
EHR Diversity Analyzer

A tool for analyzing the diversity of medical examination predictions from various LLM models.
Compares predicted numerical values against ground truth data using normalized variance metrics.
"""

import json
import numpy as np
import re
import os
import time
import argparse
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI


class BaseEHRGenerator:
    """Base class for EHR generation models."""
    
    SYSTEM_PROMPT_TEMPLATE = """You are an expert medical AI assistant specialized in predicting medical examination results. Your task is to analyze patient information and predict numerical values for specific laboratory tests.

CRITICAL FORMATTING REQUIREMENTS:
- You must provide numerical values for each requested sub-test
- Format each result as: "Sub-test Name: Numeric Value: [number] Units: [unit]"
- Use the exact units specified for each sub-test
- If multiple sub-tests are requested, provide each on a separate line
- Use realistic medical values appropriate for the patient's condition
- Be precise with numbers (use decimals when appropriate)

For the examination "{exam_name}", you need to provide values for these specific measurements: {subevents_text}

Example format:
Hemoglobin: Numeric Value: 12.5 Units: g/dL
White Blood Cell Count: Numeric Value: 7200 Units: cells/μL
Platelet Count: Numeric Value: 250000 Units: cells/μL"""
    
    def __init__(self, model_name: str, units_dict: Optional[Dict] = None):
        """
        Initialize the base generator.
        
        Args:
            model_name: Name of the model
            units_dict: Dictionary mapping events to sub-events and their units
        """
        self.model_name = model_name
        self.units_dict = units_dict or {}
    
    def get_subevent_unit(self, event_name: str, subevent_name: str) -> Optional[str]:
        """
        Get unit for a specific sub-event.
        
        Args:
            event_name: Name of the main event
            subevent_name: Name of the sub-event
            
        Returns:
            Unit string or None if not found
        """
        # Direct lookup
        if event_name in self.units_dict and subevent_name in self.units_dict[event_name]:
            return self.units_dict[event_name][subevent_name]
        
        # Fuzzy matching for partial matches
        if event_name in self.units_dict:
            for stored_subevent, unit in self.units_dict[event_name].items():
                if (subevent_name.lower() in stored_subevent.lower() or 
                    stored_subevent.lower() in subevent_name.lower()):
                    return unit
        
        return None
    
    def build_prompt(self, 
                    context: str, 
                    past_events_list: List[str],
                    exam_name: str, 
                    target_subevents: List[str]) -> Tuple[str, str]:
        """
        Build system and user prompts for the model.
        
        Args:
            context: Patient context
            past_events_list: List of past examination results
            exam_name: Name of current examination
            target_subevents: List of sub-events to predict
            
        Returns:
            Tuple of (system_prompt, user_query)
        """
        # Prepare sub-events with units
        subevents_with_units = []
        missing_units_count = 0
        
        for subevent in target_subevents:
            unit = self.get_subevent_unit(exam_name, subevent)
            if unit:
                subevents_with_units.append(f"{subevent} (Unit: {unit})")
            else:
                subevents_with_units.append(f"{subevent} (Unit: unknown)")
                missing_units_count += 1
        
        if missing_units_count > 0:
            print(f"Warning: {exam_name} has {missing_units_count} sub-events without units")
        
        subevents_text = ", ".join(subevents_with_units) if subevents_with_units else "main value"
        
        # Build system prompt
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            exam_name=exam_name,
            subevents_text=subevents_text
        )
        
        # Build user query
        past_events_text = ""
        if past_events_list:
            past_events_text = f"\n\nPast Examination Results:\n{chr(10).join(past_events_list)}"
        
        user_query = f"""Patient Case Summary:
{context}
{past_events_text}

Current Examination to Predict: {exam_name}

Required Sub-tests with Units: {subevents_text}

Please predict realistic numerical values for each sub-test listed above. Format each result exactly as specified in the system instructions and use the correct units:"""

        return system_prompt, user_query
    
    def inference(self, 
                 context: str, 
                 past_events_list: List[str],
                 exam_name: str, 
                 target_subevents: List[str]) -> Optional[str]:
        """
        Run inference (to be implemented by subclasses).
        
        Args:
            context: Patient context
            past_events_list: List of past examination results
            exam_name: Name of current examination
            target_subevents: List of sub-events to predict
            
        Returns:
            Generated text or None if failed
        """
        raise NotImplementedError("Subclasses must implement inference method")


class OpenAICompatibleGenerator(BaseEHRGenerator):
    """Generator for OpenAI-compatible APIs (Qwen, MedGemma)."""
    
    def __init__(self, 
                 api_key: str, 
                 api_base: str, 
                 model_name: str, 
                 units_dict: Optional[Dict] = None):
        """
        Initialize OpenAI-compatible generator.
        
        Args:
            api_key: API key for authentication
            api_base: Base URL for the API
            model_name: Name of the model
            units_dict: Dictionary of units
        """
        super().__init__(model_name, units_dict)
        self.client = OpenAI(api_key=api_key, base_url=api_base)
    
    def inference(self, 
                 context: str, 
                 past_events_list: List[str],
                 exam_name: str, 
                 target_subevents: List[str]) -> Optional[str]:
        """Run inference using OpenAI-compatible API."""
        system_prompt, user_query = self.build_prompt(
            context, past_events_list, exam_name, target_subevents
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    max_tokens=16324,
                    temperature=1.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None


class CustomAPIGenerator(BaseEHRGenerator):
    """Generator for custom API endpoints (DeepSeek)."""
    
    def __init__(self, 
                 api_key: str, 
                 api_url: str,
                 model_name: str,
                 units_dict: Optional[Dict] = None):
        """
        Initialize custom API generator.
        
        Args:
            api_key: API key for authentication
            api_url: Full URL for the API endpoint
            model_name: Name of the model
            units_dict: Dictionary of units
        """
        super().__init__(model_name, units_dict)
        self.url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    
    def inference(self, 
                 context: str, 
                 past_events_list: List[str],
                 exam_name: str, 
                 target_subevents: List[str]) -> Optional[str]:
        """Run inference using custom API."""
        system_prompt, user_query = self.build_prompt(
            context, past_events_list, exam_name, target_subevents
        )
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            "temperature": 1.0,
            "max_tokens": 16324,
        }

        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=json.dumps(data),
                    timeout=90,
                )
                response.raise_for_status()
                response_data = response.json()
                return response_data['choices'][0]['message']['content']
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return None
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        return None


class DiversityAnalyzer:
    """Main analyzer class for diversity analysis."""
    
    MODEL_CONFIGS = {
        "deepseek": {
            "model_name": "DeepSeek-V3",
            "display_name": "DeepSeek-V3",
            "api_url": "https://api.deepseek.com/v1/chat/completions",
            "type": "custom"
        },
        "qwen7b": {
            "model_name": "qwen2_5_7B",
            "display_name": "Qwen2.5-7B-Instruct",
            "type": "openai"
        },
        "qwen72b": {
            "model_name": "qwen2_5_72B",
            "display_name": "Qwen2.5-72B-Instruct",
            "type": "openai"
        },
        "medgemma": {
            "model_name": "medgemma",
            "display_name": "MedGemma-27B",
            "type": "openai"
        }
    }
    
    def __init__(self, 
                 model_type: str, 
                 api_config: Dict, 
                 reference_file: str, 
                 train_file: str):
        """
        Initialize the diversity analyzer.
        
        Args:
            model_type: Type of model to use
            api_config: API configuration dictionary
            reference_file: Path to reference range file
            train_file: Path to training dataset
        """
        self.model_type = model_type
        self.reference_file = reference_file
        self.train_file = train_file
        
        # Validate model type
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Extract units information
        self.units_dict = self._extract_units_from_dataset(train_file)
        
        # Initialize generator
        self.generator, self.model_name = self._create_generator(model_type, api_config)
        
        # Load target events
        self.target_events = self._load_numeric_events()
        print(f"Loaded {len(self.target_events)} examination events")
        
        # Check unit coverage
        self._check_unit_coverage()
    
    def _create_generator(self, 
                         model_type: str, 
                         api_config: Dict) -> Tuple[BaseEHRGenerator, str]:
        """
        Create appropriate generator based on model type.
        
        Args:
            model_type: Type of model
            api_config: API configuration
            
        Returns:
            Tuple of (generator, display_name)
        """
        config = self.MODEL_CONFIGS[model_type]
        
        if config["type"] == "custom":
            generator = CustomAPIGenerator(
                api_config["api_key"],
                config["api_url"],
                config["model_name"],
                self.units_dict
            )
        else:  # openai type
            if "api_base" not in api_config:
                raise ValueError(f"api_base is required for {model_type}")
            
            generator = OpenAICompatibleGenerator(
                api_config["api_key"],
                api_config["api_base"],
                config["model_name"],
                self.units_dict
            )
        
        return generator, config["display_name"]
    
    def _load_numeric_events(self) -> Dict[str, List[str]]:
        """Load all events with numeric values and their sub-events."""
        events = {}
        
        with open(self.reference_file, 'r') as fp:
            json_data = json.load(fp)
        
        for event_name in json_data.keys():
            events[event_name] = []
            reference_ranges = json_data[event_name].split('\n\n')
            
            for reference_range in reference_ranges:
                subevent = reference_range.split('\n')[0].replace('Sub Lab Events: ', '').strip()
                if subevent:
                    events[event_name].append(subevent)
        
        return events
    
    def _extract_units_from_dataset(self, dataset_path: str) -> Dict[str, Dict[str, str]]:
        """Extract units for each sub-event from dataset."""
        print("Extracting unit information from training set...")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        event_units = defaultdict(lambda: defaultdict(set))
        
        # Scan all cases for unit information
        for case_id, case_data in tqdm(data.items(), desc="Scanning for units"):
            if "events" not in case_data:
                continue
            
            for event in case_data.get("events", []):
                if event.get('source') != 'lab':
                    continue
                
                event_name = event.get('data', {}).get('label')
                if not event_name:
                    continue
                
                lab_data = event.get('data', {}).get('lab_data', '')
                
                # Extract units from lab data
                for line in lab_data.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Pattern: "SubEvent: ... Units: unit_value"
                    match = re.match(r'^([^:]+):\s*.*?Units:\s*([^;]+)', line)
                    if match:
                        subevent_name = match.group(1).strip()
                        unit = match.group(2).strip()
                        if subevent_name and unit:
                            event_units[event_name][subevent_name].add(unit)
        
        # Consolidate units (use most common or single value)
        final_units = {}
        for event_name, subevents in event_units.items():
            final_units[event_name] = {}
            for subevent_name, units_set in subevents.items():
                if units_set and len(units_set) == 1:
                    final_units[event_name][subevent_name] = list(units_set)[0]
                elif units_set:
                    # If multiple units found, could implement voting or take first
                    final_units[event_name][subevent_name] = list(units_set)[0]
        
        total_subevents = sum(len(subevents) for subevents in final_units.values())
        print(f"Extracted units for {len(final_units)} events ({total_subevents} sub-events)")
        
        return final_units
    
    def _check_unit_coverage(self) -> None:
        """Check unit coverage for target events."""
        print("Checking unit coverage...")
        
        total_subevents = 0
        covered_subevents = 0
        
        for event_name, subevents in self.target_events.items():
            for subevent in subevents:
                total_subevents += 1
                unit = self.generator.get_subevent_unit(event_name, subevent)
                if unit:
                    covered_subevents += 1
        
        coverage_rate = (covered_subevents / total_subevents * 100 
                        if total_subevents > 0 else 0)
        print(f"Unit coverage: {covered_subevents}/{total_subevents} ({coverage_rate:.1f}%)")
    
    def extract_numeric_value(self, text: str, target_subevent: str) -> Optional[float]:
        """
        Extract numeric value from model output.
        
        Args:
            text: Model output text
            target_subevent: Name of the sub-event to extract
            
        Returns:
            Extracted numeric value or None
        """
        if not text or not target_subevent:
            return None
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line contains the target sub-event
            if target_subevent.lower() in line.lower():
                # Extract numeric value
                numeric_match = re.search(r'Numeric Value:\s*([+-]?\d*\.?\d+)', line)
                if numeric_match:
                    try:
                        return float(numeric_match.group(1))
                    except ValueError:
                        continue
        
        return None
    
    def extract_ground_truth_values(self, dataset_path: str) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract ground truth values from dataset.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            Dictionary of ground truth values
        """
        print("Extracting ground truth values...")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        ground_truth_values = defaultdict(lambda: defaultdict(list))
        
        for case_id, case_data in tqdm(data.items(), desc="Scanning ground truth"):
            if "events" not in case_data:
                continue
            
            for event in case_data.get("events", []):
                if event.get('source') != 'lab':
                    continue
                
                event_name = event.get('data', {}).get('label')
                if event_name not in self.target_events:
                    continue
                
                lab_data = event.get('data', {}).get('lab_data', '')
                
                # Extract values for each target sub-event
                for target_subevent in self.target_events[event_name]:
                    value = self.extract_numeric_value(lab_data, target_subevent)
                    if value is not None:
                        ground_truth_values[event_name][target_subevent].append(value)
        
        total_values = sum(
            len(values) 
            for event_values in ground_truth_values.values() 
            for values in event_values.values()
        )
        print(f"Extracted {total_values} ground truth values")
        
        return dict(ground_truth_values)
    
    def run_model_inference(self, 
                          dataset_path: str, 
                          sample_limit_per_event: int = 30,
                          max_workers: int = 16) -> Dict[str, List[Dict]]:
        """
        Run model inference on dataset.
        
        Args:
            dataset_path: Path to dataset file
            sample_limit_per_event: Maximum samples per event type
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary of inference results
        """
        print(f"Starting {self.model_name} inference...")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Collect samples containing target events
        event_samples = defaultdict(list)
        
        for case_id, case_data in data.items():
            if "events" not in case_data:
                continue
            
            # Find all lab events in this case
            case_events = set()
            for event in case_data.get("events", []):
                if event.get('source') == 'lab':
                    event_name = event.get('data', {}).get('label')
                    if event_name in self.target_events:
                        case_events.add(event_name)
            
            # Add this case to each relevant event type
            for event_name in case_events:
                if len(event_samples[event_name]) < sample_limit_per_event:
                    # Build context
                    discharge_diagnosis = case_data.get("discharge_diagnosis", "")
                    discharge_note = case_data.get("text", "")
                    
                    if "Physical Exam:" in discharge_note:
                        context = discharge_note.split('Physical Exam:')[0].strip()
                    else:
                        context = discharge_note
                    
                    context += "\n\nFinal Diagnosis:\n" + discharge_diagnosis
                    
                    event_samples[event_name].append({
                        'case_id': case_id,
                        'context': context
                    })
        
        print(f"Prepared {len(event_samples)} event types for inference")
        
        # Prepare all inference tasks
        all_tasks = []
        for event_name, samples in event_samples.items():
            target_subevents = self.target_events[event_name]
            for sample in samples:
                all_tasks.append((sample, event_name, target_subevents))
        
        print(f"Total inference tasks: {len(all_tasks)}")
        
        # Execute inference with thread pool
        inference_results = defaultdict(list)
        
        def process_task(args: Tuple) -> Dict[str, Any]:
            """Process a single inference task."""
            sample, event_name, target_subevents = args
            result = self.generator.inference(
                sample['context'], [], event_name, target_subevents
            )
            return {
                'event_name': event_name,
                'case_id': sample['case_id'],
                'prediction': result
            }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_task, task) for task in all_tasks]
            
            for future in tqdm(as_completed(futures), 
                             total=len(all_tasks),
                             desc=f"Running {self.model_name} inference"):
                try:
                    result = future.result()
                    if result and result['prediction']:
                        inference_results[result['event_name']].append({
                            'case_id': result['case_id'],
                            'prediction': result['prediction']
                        })
                except Exception as e:
                    print(f"Task failed: {e}")
        
        total_predictions = sum(len(predictions) for predictions in inference_results.values())
        print(f"Completed {total_predictions} valid predictions")
        
        return dict(inference_results)
    
    def extract_predicted_values(self, 
                                inference_results: Dict) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract predicted values from inference results.
        
        Args:
            inference_results: Dictionary of inference results
            
        Returns:
            Dictionary of predicted values
        """
        print(f"Extracting {self.model_name} predicted values...")
        
        predicted_values = defaultdict(lambda: defaultdict(list))
        
        for event_name, predictions in inference_results.items():
            for pred_data in predictions:
                prediction_text = pred_data['prediction']
                
                # Extract values for each target sub-event
                for target_subevent in self.target_events[event_name]:
                    value = self.extract_numeric_value(prediction_text, target_subevent)
                    if value is not None:
                        predicted_values[event_name][target_subevent].append(value)
        
        total_values = sum(
            len(values) 
            for event_values in predicted_values.values() 
            for values in event_values.values()
        )
        print(f"Extracted {total_values} predicted values")
        
        return dict(predicted_values)
    
    def calculate_normalized_variance(self, 
                                     values_dict: Dict[str, Dict[str, List[float]]],
                                     data_type: str = "data") -> Tuple[Dict, float]:
        """
        Calculate normalized variance for values.
        
        Args:
            values_dict: Dictionary of values
            data_type: Type of data (for logging)
            
        Returns:
            Tuple of (detailed variances, overall variance)
        """
        print(f"Calculating {data_type} normalized variance...")
        
        normalized_variances = {}
        all_norm_vars = []
        
        for event_name, subevents in values_dict.items():
            event_norm_vars = {}
            
            for subevent, values in subevents.items():
                if len(values) >= 3:  # Need at least 3 values for meaningful variance
                    mean_val = np.mean(values)
                    var_val = np.var(values, ddof=1)  # Sample variance
                    
                    if mean_val != 0:
                        norm_var = var_val / (mean_val ** 2)
                        event_norm_vars[subevent] = {
                            'count': len(values),
                            'mean': float(mean_val),
                            'variance': float(var_val),
                            'normalized_variance': float(norm_var),
                            'std': float(np.std(values, ddof=1)),
                            'cv': float(np.std(values, ddof=1) / mean_val)  # Coefficient of variation
                        }
                        all_norm_vars.append(norm_var)
            
            if event_norm_vars:
                normalized_variances[event_name] = event_norm_vars
        
        overall_norm_var = float(np.mean(all_norm_vars)) if all_norm_vars else 0.0
        print(f"{data_type} overall normalized variance: {overall_norm_var:.6f}")
        
        return normalized_variances, overall_norm_var
    
    def save_results(self, 
                    inference_results: Dict,
                    gt_norm_vars: Dict,
                    pred_norm_vars: Dict,
                    gt_overall: float,
                    pred_overall: float,
                    output_dir: str = "results") -> None:
        """
        Save analysis results to files.
        
        Args:
            inference_results: Raw inference results
            gt_norm_vars: Ground truth normalized variances
            pred_norm_vars: Predicted normalized variances
            gt_overall: Overall ground truth variance
            pred_overall: Overall predicted variance
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save inference results
        inference_data = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'total_predictions': sum(len(predictions) for predictions in inference_results.values()),
            'event_count': len(inference_results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': inference_results
        }
        
        inference_file = os.path.join(output_dir, f"{self.model_type}_inference.json")
        with open(inference_file, 'w') as f:
            json.dump(inference_data, f, indent=2)
        
        # Save analysis results
        analysis_data = {
            'model': self.model_name,
            'model_type': self.model_type,
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
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        analysis_file = os.path.join(output_dir, f"{self.model_type}_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        # Print results summary
        print(f"\nResults saved to {output_dir}/")
        print(f"  - Inference results: {inference_file}")
        print(f"  - Analysis results: {analysis_file}")
        
        self._print_summary(gt_overall, pred_overall)
    
    def _print_summary(self, gt_overall: float, pred_overall: float) -> None:
        """Print analysis summary."""
        print(f"\n{'='*60}")
        print(f"{self.model_name} Diversity Analysis Summary")
        print(f"{'='*60}")
        print(f"Ground truth normalized variance: {gt_overall:.6f}")
        print(f"{self.model_name} predicted normalized variance: {pred_overall:.6f}")
        
        if gt_overall > 0:
            ratio = pred_overall / gt_overall
            print(f"Diversity ratio: {ratio:.3f}x")
            
            if ratio < 0.5:
                print("Status: Low diversity (model predictions are too uniform)")
            elif ratio < 0.8:
                print("Status: Moderate diversity")
            elif ratio < 1.2:
                print("Status: Good diversity (close to ground truth)")
            else:
                print("Status: High diversity (more varied than ground truth)")
        
        print(f"{'='*60}")
    
    def run_analysis(self, 
                    dataset_path: str,
                    sample_limit_per_event: int = 20,
                    max_workers: int = 16,
                    output_dir: str = "results") -> Tuple[float, float]:
        """
        Run complete diversity analysis.
        
        Args:
            dataset_path: Path to dataset file
            sample_limit_per_event: Maximum samples per event type
            max_workers: Maximum number of parallel workers
            output_dir: Output directory path
            
        Returns:
            Tuple of (ground_truth_variance, predicted_variance)
        """
        print(f"\nStarting {self.model_name} diversity analysis...")
        print(f"Dataset: {dataset_path}")
        print(f"Sample limit per event: {sample_limit_per_event}")
        print(f"Max workers: {max_workers}")
        
        # Step 1: Extract ground truth values
        ground_truth_values = self.extract_ground_truth_values(dataset_path)
        gt_norm_vars, gt_overall = self.calculate_normalized_variance(
            ground_truth_values, "ground truth"
        )
        
        # Step 2: Run model inference
        inference_results = self.run_model_inference(
            dataset_path, sample_limit_per_event, max_workers
        )
        
        # Step 3: Extract predicted values
        predicted_values = self.extract_predicted_values(inference_results)
        pred_norm_vars, pred_overall = self.calculate_normalized_variance(
            predicted_values, f"{self.model_name}"
        )
        
        # Step 4: Save results
        self.save_results(
            inference_results, gt_norm_vars, pred_norm_vars,
            gt_overall, pred_overall, output_dir
        )
        
        return gt_overall, pred_overall


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If validation fails
    """
    # Check if files exist
    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset file not found: {args.dataset}")
    
    if not os.path.exists(args.reference):
        raise ValueError(f"Reference file not found: {args.reference}")
    
    if not os.path.exists(args.train):
        raise ValueError(f"Training file not found: {args.train}")
    
    # Check API configuration
    if args.model in ['qwen7b', 'qwen72b', 'medgemma'] and not args.api_base:
        raise ValueError(f"--api-base is required for {args.model}")
    
    # Validate numeric arguments
    if args.samples_per_event <= 0:
        raise ValueError("--samples-per-event must be positive")
    
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EHR Diversity Analyzer - Analyze diversity of medical examination predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze using DeepSeek model
  %(prog)s --model deepseek --dataset test.json --reference ref.json --train train.json \\
           --api-key YOUR_KEY

  # Analyze using local Qwen model
  %(prog)s --model qwen72b --dataset test.json --reference ref.json --train train.json \\
           --api-key dummy --api-base http://localhost:8081/v1

  # Analyze with custom parameters
  %(prog)s --model medgemma --dataset test.json --reference ref.json --train train.json \\
           --api-key YOUR_KEY --api-base http://localhost:8084/v1 \\
           --samples-per-event 30 --max-workers 32 --output-dir custom_results

Notes:
  - The reference file should contain numeric event definitions
  - The train file is used to extract unit information
  - Results are saved in JSON format for further analysis
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=['deepseek', 'qwen7b', 'qwen72b', 'medgemma'],
        help='Model type to use for predictions'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to test dataset JSON file'
    )
    
    parser.add_argument(
        '--reference', 
        type=str, 
        required=True,
        help='Path to reference range JSON file containing numeric events'
    )
    
    parser.add_argument(
        '--train', 
        type=str, 
        required=True,
        help='Path to training dataset for extracting units'
    )
    
    parser.add_argument(
        '--api-key', 
        type=str, 
        required=True,
        help='API key for the model'
    )
    
    # Optional arguments
    parser.add_argument(
        '--api-base', 
        type=str, 
        default=None,
        help='API base URL (required for OpenAI-compatible models)'
    )
    
    parser.add_argument(
        '--samples-per-event', 
        type=int, 
        default=20,
        help='Maximum samples per event type (default: 20)'
    )
    
    parser.add_argument(
        '--max-workers', 
        type=int, 
        default=16,
        help='Maximum number of parallel workers (default: 16)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='diversity_results',
        help='Output directory for results (default: diversity_results)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Prepare API configuration
        api_config = {"api_key": args.api_key}
        if args.api_base:
            api_config["api_base"] = args.api_base
        
        # Initialize analyzer
        print(f"\nInitializing {args.model} analyzer...")
        analyzer = DiversityAnalyzer(
            model_type=args.model,
            api_config=api_config,
            reference_file=args.reference,
            train_file=args.train
        )
        
        # Run analysis
        start_time = time.time()
        gt_nv, pred_nv = analyzer.run_analysis(
            dataset_path=args.dataset,
            sample_limit_per_event=args.samples_per_event,
            max_workers=args.max_workers,
            output_dir=args.output_dir
        )
        
        # Print final summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Analysis Complete!")
        print(f"{'='*60}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Ground truth normalized variance: {gt_nv:.6f}")
        print(f"Predicted normalized variance: {pred_nv:.6f}")
        
        if gt_nv > 0:
            ratio = pred_nv / gt_nv
            print(f"Diversity ratio: {ratio:.3f}x")
            
            # Provide interpretation
            if ratio < 0.5:
                print("\nInterpretation: The model generates overly uniform predictions.")
                print("Recommendation: Consider adjusting temperature or prompting strategy.")
            elif ratio < 0.8:
                print("\nInterpretation: The model shows moderate diversity in predictions.")
                print("Recommendation: Slightly increase generation diversity if needed.")
            elif ratio < 1.2:
                print("\nInterpretation: The model's diversity closely matches ground truth.")
                print("Recommendation: Current settings are well-calibrated.")
            else:
                print("\nInterpretation: The model generates more varied predictions than ground truth.")
                print("Recommendation: Consider reducing temperature if consistency is needed.")
        
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()