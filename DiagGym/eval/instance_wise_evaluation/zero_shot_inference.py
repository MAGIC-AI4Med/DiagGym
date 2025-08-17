#!/usr/bin/env python3
"""
EHR Medical Examination Result Predictor

A tool for predicting medical examination results based on patient case summaries
and past events using various LLM APIs.
"""

import json
import argparse
import requests
import time
import random
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
from openai import OpenAI

# Random seed for reproducibility
random.seed(42)

# Separator token for concatenating past events
SEP = "<SEP>"

# System prompt for medical prediction
SYSTEM_PROMPT = """You are an expert medical AI assistant specialized in predicting medical examination results based on patient case summaries and past events. Your task is to analyze the provided patient information and predict the most likely results for a specific medical examination.

Instructions:
1. Carefully analyze the patient case summary, including diagnosis, symptoms, and clinical presentation
2. Consider all past examination results and their implications
3. Based on the medical context, predict realistic and clinically appropriate results for the requested examination
4. Provide only the examination results without additional explanation or reasoning
5. Format your response as concise, medically accurate examination findings
6. If multiple measurements or findings are typical for the exam, include all relevant components
7. Ensure your predictions are consistent with the patient's overall clinical picture

Remember: Your predictions should be realistic, medically sound, and consistent with the patient's condition."""


class EHRGenerator:
    """EHR text generator using various LLM APIs."""

    def __init__(self, model_name: str, api_key: str, api_base: str, api_type: str = "openai") -> None:
        """
        Initialize the EHR generator.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            api_base: Base URL for the API
            api_type: Type of API ("openai" or "custom")
        """
        self.model_name = model_name
        self.api_type = api_type
        
        if api_type == "openai":
            self.client = OpenAI(api_key=api_key, base_url=api_base)
        else:
            self.api_url = api_base
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

    def inference(self, context: str, past_events_list: List[str], exam_name: str) -> Optional[str]:
        """
        Generate prediction for a single medical exam.
        
        Args:
            context: Patient case summary and diagnosis
            past_events_list: List of past examination results
            exam_name: Name of the current examination to predict
            
        Returns:
            Predicted examination results or None if failed
        """
        past_events_text = f"\n\nPast Examination Results:\n{SEP.join(past_events_list)}" if past_events_list else ""
        
        query = f"""Patient Case Summary:
{context}
{past_events_text}

Current Examination to Predict:
Exam name: {exam_name}

Please predict the most likely results for this examination based on the patient's clinical information and past results. Provide only the examination results:"""

        if self.api_type == "openai":
            return self._inference_openai(query)
        else:
            return self._inference_custom(query)

    def _inference_openai(self, query: str) -> Optional[str]:
        """Make inference using OpenAI-compatible API."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=16384,
                    temperature=1.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return None
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(2 ** attempt)
        return None

    def _inference_custom(self, query: str) -> Optional[str]:
        """Make inference using custom API endpoint."""
        max_retries = 5
        base_delay = 2
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": 1.0,
            "max_tokens": 16384,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=json.dumps(data),
                    timeout=90,
                )
                response.raise_for_status()
                response_data = response.json()
                return response_data['choices'][0]['message']['content']
            except (requests.exceptions.RequestException, ConnectionError) as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    return None
                delay = base_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                time.sleep(delay)
        return None


def organize_events_by_time(events: List[Dict]) -> List[List[tuple]]:
    """
    Group events by their timestamp and return in chronological order.
    
    Args:
        events: List of event dictionaries with time and data fields
        
    Returns:
        List of grouped events sorted by time
    """
    time_grouped = {}
    for event in events:
        event_time = event["time"]
        if event_time not in time_grouped:
            time_grouped[event_time] = []
        time_grouped[event_time].append((event['data']['label'], event['data']['lab_data']))

    sorted_times = sorted(time_grouped.keys())
    return [time_grouped[t] for t in sorted_times]


def process_single_sample(args: tuple, mode: str) -> Dict[str, Any]:
    """
    Process a single EHR sample.
    
    Args:
        args: Tuple containing (index, item data, EHR generator instance)
        mode: Processing mode ("stepwise" or "fullchain")
        
    Returns:
        Dictionary containing processing results
    """
    idx, item, ehr_generator = args
    context = item['context'] + "\nThe following summarizes the results from the patient's medical examination:\n"
    events = item['events']

    sample_result = {
        'note_id': item['note_id'],
        'context': item['context'],
        'predictions': [],
        'original_index': idx,
        'event_times': []
    }

    past_events_list = []
    for event_group_idx, event_group in enumerate(events):
        for event_idx, event in enumerate(event_group):
            event_name, event_result = event
            start_time = time.time()

            current_event = {
                'group_idx': event_group_idx,
                'event_idx': event_idx,
                'exam_name': event_name,
                'ground_truth': event_result,
            }
            
            resp = ehr_generator.inference(context, past_events_list, event_name)
            current_event['prediction'] = resp

            elapsed = time.time() - start_time
            current_event['processing_time'] = elapsed
            sample_result['event_times'].append(elapsed)
            sample_result['predictions'].append(current_event)

            if mode == "stepwise":
                past_events_list.append(f"Exam name:\n{event_name}\nExam results:\n{event_result}")
            else:
                past_events_list.append(f"Exam name:\n{event_name}\nExam results:\n{resp}")

    return sample_result


def run_prediction(
    data_filepath: str,
    model_name: str,
    api_key: str,
    api_base: str,
    api_type: str = "openai",
    mode: str = "stepwise",
    num_threads: int = 4,
    output_dir: str = "results"
) -> None:
    """
    Run EHR prediction on dataset.
    
    Args:
        data_filepath: Path to input JSON data file
        model_name: Name of the model to use
        api_key: API key for authentication
        api_base: Base URL for the API
        api_type: Type of API ("openai" or "custom")
        mode: Processing mode ("stepwise" or "fullchain")
        num_threads: Number of parallel threads
        output_dir: Directory for output files
    """
    print(f"Loading data from {data_filepath}...")
    with open(data_filepath, 'r') as f:
        json_data = json.load(f)

    data = []
    for note_id, note_data in json_data.items():
        if "reformat_physical_exam" not in note_data:
            continue
            
        discharge_diagnosis = note_data["discharge_diagnosis"]
        discharge_note = note_data["text"]

        generator_context = (
            discharge_note.split('Physical Exam:')[0].strip() + 
            "\n\nFinal Diagnosis:\n" + discharge_diagnosis
        )

        events = [[]]
        for item in note_data["reformat_physical_exam"]:
            events[0].append((item['exam_name'], item['exam_results']))
        events.extend(organize_events_by_time(note_data["events"]))

        data.append({
            'note_id': note_id,
            'context': generator_context,
            'events': events
        })

    print(f"Processing {len(data)} samples with {num_threads} threads...")
    
    all_results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        tasks = []
        for idx, item in enumerate(data):
            ehr_generator = EHRGenerator(model_name, api_key, api_base, api_type)
            task = executor.submit(process_single_sample, (idx, item, ehr_generator), mode)
            tasks.append(task)

        for task in tqdm(as_completed(tasks), total=len(tasks), desc=f"Processing ({mode} mode)"):
            result = task.result()
            original_idx = result['original_index']
            del result['original_index']
            all_results[original_idx] = result

    # Calculate statistics
    all_event_times = [t for result in all_results for t in result['event_times']]
    for result in all_results:
        del result['event_times']

    avg_event_time = sum(all_event_times) / len(all_event_times) if all_event_times else 0

    final_result = {
        'model': model_name,
        'mode': mode,
        'average_process_event_time_consumption': avg_event_time,
        'total_events_processed': len(all_event_times),
        'results': all_results
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{mode}_{model_name.replace('/', '_')}_results.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Average event processing time: {avg_event_time:.2f} seconds")
    print(f"Total events processed: {len(all_event_times)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EHR Medical Examination Result Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using OpenAI-compatible API
  %(prog)s data.json --model gpt-4 --api-key YOUR_KEY --api-base https://api.openai.com/v1

  # Using custom API endpoint
  %(prog)s data.json --model DeepSeek-V3 --api-key YOUR_KEY --api-base https://api.example.com/v1/chat/completions --api-type custom

  # Process in fullchain mode with more threads
  %(prog)s data.json --model medgemma --mode fullchain --threads 8
        """
    )
    
    parser.add_argument('input_file', help='Path to input JSON data file')
    parser.add_argument('--model', required=True, help='Model name to use for predictions')
    parser.add_argument('--api-key', required=True, help='API key for authentication')
    parser.add_argument('--api-base', required=True, help='Base URL for the API')
    parser.add_argument('--api-type', choices=['openai', 'custom'], default='openai',
                        help='Type of API endpoint (default: openai)')
    parser.add_argument('--mode', choices=['stepwise', 'fullchain'], default='stepwise',
                        help='Processing mode (default: stepwise)')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    
    run_prediction(
        data_filepath=args.input_file,
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        api_type=args.api_type,
        mode=args.mode,
        num_threads=args.threads,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()