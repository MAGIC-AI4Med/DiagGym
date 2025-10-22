"""
Process medical diagnose dataset into Verl RLHF format.
"""

import argparse
import os
import json

import datasets
from datasets import Dataset
from datasets.tasks.language_modeling import dataclass
from verl.utils.hdfs_io import copy, makedirs


INSTRUCTION = '''You are a medical AI assistant. Help the doctor with diagnosis by analyzing patient information, suggesting relevant tests, and providing a final diagnosis when sufficient information is available.

RESPONSE FORMAT:

If more information is needed:
```
Current diagnosis: [your diagnosis according to the information provided]
Based on the patient's initial presentation, the following investigation(s) should be performed: [one additional test]
Reason: [reason for the test]
```

If sufficient information exists for diagnosis:
```
The available information is sufficient to make a diagnosis. 

Diagnosis: [Diagnosis result]
Reason: [Diagnosis reason]
```'''


def load_diagnose_dataset(diaggym_ref_data_path, diagagent_data_path):
    """Load diagnose dataset from JSON file."""
    # get mapping table (Patient Profile)
    with open(diaggym_ref_data_path, 'r') as f:
        json_data = json.load(f)
    mapping_dict = {}
    for note_id in json_data.keys():
        if "reformat_physical_exam" not in json_data[note_id]:
            continue
        discharge_diagnosis = json_data[note_id]["discharge_diagnosis"]
        discharge_note = json_data[note_id]["text"]
        generator_context = discharge_note.split('Physical Exam:')[0].strip() + "\n\nFinal Diagnosis:\n" + discharge_diagnosis
        mapping_dict[note_id] = generator_context


    with open(diagagent_data_path, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)
    
    data = []
    for note_id in json_data.keys():
        item = json_data[note_id]
        if note_id not in mapping_dict.keys():
            continue
        if 'recommended_exam_names' not in item.keys():
            continue
        data.append({
            'question': item['case_summary'],
            'answer': item['final_diagnosis'],
            'text': mapping_dict[note_id],
            'key_exam_names' : item['recommended_exam_names']
        })
        
    # data = data[:5000]
    return Dataset.from_list(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir")
    parser.add_argument("--diaggym_ref_data_path", help="Path to diaggym reference JSON file")# providing patient profile for diaggym to predict examination results 
    parser.add_argument("--diagagent_train_data_path", help="Path to training data JSON file")
    parser.add_argument("--diaagent_test_data_path", help="Path to test data JSON file")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "diagnose_dataset"

    # Load your custom datasets
    train_dataset = load_diagnose_dataset(args.diaggym_ref_data_path, args.diagagent_train_data_path)
    test_dataset = load_diagnose_dataset(args.diaggym_ref_data_path, args.diaagent_test_data_path)

    # Add necessary fields to each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            answer = example.pop("answer")
            ehr_text = example.pop("text")
            key_exam_names = example.pop("key_exam_names")
            assert len(ehr_text) > 1

            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": question}
                ],
                "ability": "diagnose",
                "reward_model": {"style": "LLM", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                    "text" : ehr_text,
                    "key_exam_names" : key_exam_names,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Create directory if it doesn't exist
    os.makedirs(os.path.expanduser(args.local_dir), exist_ok=True)
    
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(len(train_dataset))
    print(len(test_dataset))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)