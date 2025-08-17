import json
from openai import OpenAI
from typing import Optional
import random
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Global random seed for reproducibility
random.seed(42)

# Separator token for concatenating past events
SEP = "<SEP>"
stop_tokens = [SEP, "<endoftext>"]

# API credentials should be provided via environment variables or config file
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8079/v1")


class EHRGenerator:
    """EHR text generator using OpenAI API."""

    def __init__(self, model_name_or_path: str, api_key: str = API_KEY, api_base: str = API_BASE) -> None:
        self.model_name_or_path = model_name_or_path
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def inference(self, context: str, past_events_list: list, exam_name: str) -> Optional[str]:
        """Generate exam results based on context and past events."""
        if len(past_events_list) == 0:
            input_prompt = context + "Exam name:\n" + exam_name + "\nExam results:\n"
        else:
            input_prompt = context + SEP.join(past_events_list) + SEP + "Exam name:\n" + exam_name + "\nExam results:\n"

        for i in range(3):  # Retry mechanism
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


def organize_events_by_time(events: list) -> list:
    """Group events by time and return them in chronological order."""
    time_grouped = {}
    for event in events:
        event_time = event["time"]
        if event_time not in time_grouped:
            time_grouped[event_time] = []
        time_grouped[event_time].append((event['data']['label'], event['data']['lab_data']))

    sorted_times = sorted(time_grouped.keys())
    result = [time_grouped[time] for time in sorted_times]
    return result


def process_single_sample_stepwise(args):
    """Process a single sample in stepwise mode."""
    idx, item, ehr_generator = args
    note_id = item['note_id']
    context = item['context'] + "\nThe following summarizes the results from the patient's medical examination:\n"
    events = item['events']

    sample_result = {
        'note_id': note_id,
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

            end_time = time.time()
            event_time = end_time - start_time
            current_event['processing_time'] = event_time
            sample_result['event_times'].append(event_time)

            sample_result['predictions'].append(current_event)
            past_events_list.append(f"Exam name:\n{event_name}\nExam results:\n{event_result}")

    return sample_result


def process_single_sample_fullchain(args):
    """Process a single sample in fullchain mode."""
    idx, item, ehr_generator = args
    note_id = item['note_id']
    context = item['context'] + "\nThe following summarizes the results from the patient's medical examination:\n"
    events = item['events']

    sample_result = {
        'note_id': note_id,
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

            end_time = time.time()
            event_time = end_time - start_time
            current_event['processing_time'] = event_time
            sample_result['event_times'].append(event_time)

            sample_result['predictions'].append(current_event)
            past_events_list.append(f"Exam name:\n{event_name}\nExam results:\n{resp}")

    return sample_result


def run_prediction(data_filepath: str, mode: str = "stepwise"):
    """Run EHR prediction in either stepwise or fullchain mode."""
    model_name_or_path = "EHRGenerator"

    with open(data_filepath, 'r') as f:
        json_data = json.load(f)

    data = []
    for note_id, note_data in json_data.items():
        if "reformat_physical_exam" not in note_data:
            continue
        discharge_diagnosis = note_data["discharge_diagnosis"]
        discharge_note = note_data["text"]

        generator_context = discharge_note.split('Physical Exam:')[0].strip() + "\n\nFinal Diagnosis:\n" + discharge_diagnosis

        events = [[]]
        for item in note_data["reformat_physical_exam"]:
            events[0].append((item['exam_name'], item['exam_results']))
        events.extend(organize_events_by_time(note_data["events"]))
        data.append({'note_id': note_id, 'context': generator_context, 'events': events})

    print(f"Data length: {len(data)}")

    num_threads = 8
    all_results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        tasks = []
        for idx, item in enumerate(data):
            ehr_generator = EHRGenerator(model_name_or_path)
            if mode == "stepwise":
                task = executor.submit(process_single_sample_stepwise, (idx, item, ehr_generator))
            else:
                task = executor.submit(process_single_sample_fullchain, (idx, item, ehr_generator))
            tasks.append(task)

        for task in tqdm(as_completed(tasks), total=len(tasks), desc=f"Processing EHR samples ({mode})"):
            result = task.result()
            original_idx = result['original_index']
            del result['original_index']
            all_results[original_idx] = result

    all_event_times = []
    for result in all_results:
        all_event_times.extend(result['event_times'])
        del result['event_times']

    average_event_time = sum(all_event_times) / len(all_event_times) if all_event_times else 0

    final_result = {
        'average_process_event_time_consumption': average_event_time,
        'total_events_processed': len(all_event_times),
        'results': all_results
    }

    os.makedirs("results", exist_ok=True)
    output_filename = f"{mode}_EHRGenerator_results.json"
    with open(os.path.join("results", output_filename), 'w') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage (set your own paths)
    run_prediction("data/test_data.json", mode="stepwise")
    run_prediction("data/test_data.json", mode="fullchain")