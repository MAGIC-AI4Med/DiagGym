import json
import copy
import random
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import random

IGNORE_INDEX = -100
SEP = '<SEP>'

def organize_events_by_time(events, ref_range_dict):
    time_grouped = {}
    
    for event in events:
        time = event["time"]
        if time not in time_grouped:
            time_grouped[time] = []
        ref_range = ref_range_dict.get(event['data']['label'], None)
        time_grouped[time].append({
            'exam_name': event['data']['label'],
            'exam_results':event['data']['lab_data'],
            'ref_range': ref_range
        })
    
    sorted_times = sorted(time_grouped.keys())
    
    result = []
    for time in sorted_times:
        result.append(time_grouped[time])
    
    return result
class EHRGeneratorDataset(Dataset):
    def __init__(self, tokenizer, data_path, ref_range_json_path=None, use_ref_range_posibility=0.5):
        self.tokenizer = tokenizer
        self.data_path = data_path
        if self.data_path is None:
            self.use_ref_range_posibility = 0
            self.ref_range_dict = {}
        else:
            self.use_ref_range_posibility = use_ref_range_posibility
            with open(ref_range_json_path, 'r', encoding='utf-8') as fp:
                self.ref_range_dict = json.load(fp)

        self.raw_data_list = []
        
        self.template_with_ref = "Exam name:\n{}\nReference Range:\n{}\nExam results:\n{}" + SEP
        self.template_without_ref = "Exam name:\n{}\nExam results:\n{}" + SEP
        
        self._pregenerate_random_numbers()
        
        with open(self.data_path, 'r') as f:
            json_data = json.load(f)
            
        for note_id in json_data.keys():
            if "reformat_physical_exam" not in json_data[note_id]:
                continue
                
            discharge_diagnosis = json_data[note_id]["discharge_diagnosis"]
            discharge_note = json_data[note_id]["text"]
            
            generator_context = discharge_note.split('Physical Exam:')[0].strip() + "\n\nFinal Diagnosis:\n" + discharge_diagnosis
            
            events = [[]]
            for item in json_data[note_id]["reformat_physical_exam"]:
                events[0].append({
                    'exam_name': item['exam_name'], 
                    'exam_results': item['exam_results'], 
                    'ref_range': None
                })
                
            events.extend(organize_events_by_time(json_data[note_id]["events"], self.ref_range_dict))
            
            self.raw_data_list.append({
                'generator_context': generator_context,
                'events': events
            })
    
    def _pregenerate_random_numbers(self, pool_size=1000000):
        self.random_pool = [random.random() for _ in range(pool_size)]
        self.random_idx = 0
    
    def _get_random(self):
        if self.random_idx >= len(self.random_pool):
            self.random_idx = 0
        result = self.random_pool[self.random_idx]
        self.random_idx += 1
        return result
    
    
    def __len__(self):
        return len(self.raw_data_list)
    
    def __getitem__(self, idx):
        data = self.raw_data_list[idx]
        generator_context = data['generator_context']
        events = data['events']
        
        events_parts = []
        
        for i, event_group in enumerate(events):
            indices = list(range(len(event_group)))
            random.shuffle(indices)
            
            for idx_in_group in indices:
                event = event_group[idx_in_group]
                
                if event['ref_range'] and self._get_random() < self.use_ref_range_posibility:
                    events_parts.append(self.template_with_ref.format(
                        event['exam_name'], 
                        event['ref_range'], 
                        event['exam_results']
                    ))
                else:
                    events_parts.append(self.template_without_ref.format(
                        event['exam_name'], 
                        event['exam_results']
                    ))
        
        events_str = ''.join(events_parts)
        
        return self.preprocess(generator_context, events_str)
            
    def preprocess(
        self,
        input_prompt: str,
        gt_output: str,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        source = f"{input_prompt}\nThe following summarizes the results from the patient's medical examination:\n"
        example = f"{input_prompt}\nThe following summarizes the results from the patient's medical examination:\n{gt_output}<|endoftext|>"
        example_tokenized = self.tokenizer(
            example,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=False,
        )
        source_tokenized = self.tokenizer(
                source,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                truncation=False,
            )
        
        input_ids = example_tokenized["input_ids"][0]
        attention_mask = example_tokenized["attention_mask"][0]
        source_input_ids =  source_tokenized["input_ids"][0]
        labels = copy.deepcopy(input_ids)
        labels[:len(source_input_ids)] = IGNORE_INDEX
        if len(input_ids) > self.tokenizer.model_max_length:
            input_ids = input_ids[-self.tokenizer.model_max_length:]
            labels = labels[-self.tokenizer.model_max_length:] 
            attention_mask = attention_mask[-self.tokenizer.model_max_length:]
        return dict(input_ids=input_ids, labels=labels, attention_mask = attention_mask)


    
    