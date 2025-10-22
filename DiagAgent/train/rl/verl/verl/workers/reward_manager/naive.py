# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import concurrent.futures
import threading
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


def process_item(item_idx, data, tokenizer, compute_score, reward_fn_key):
    """Process a single data item in parallel"""
    data_item = data[item_idx]  # DataProtoItem

    prompt_ids = data_item.batch["prompts"]
    prompt_length = prompt_ids.shape[-1]
    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch["responses"]
    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    # decode
    prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    data_source = data_item.non_tensor_batch[reward_fn_key]
    key_exam_names = data_item.non_tensor_batch['key_exam_names']
    examination_history = data_item.non_tensor_batch['examination_histories']
    turn_count = data_item.non_tensor_batch['turn_counts']

    examination_history = [item.split('Exam results:')[0].split('Exam name:')[1].strip() for item in examination_history]

    # extra_info = data_item.non_tensor_batch.get("extra_info", None)

    score_dict = compute_score(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info={
            'key_exam_names' : key_exam_names,
            'examination_history' : examination_history,
            'turn_count' : turn_count,
        },
    )

    

    return {
        "idx": item_idx,
        "valid_response_length": valid_response_length,
        "reward": score_dict,
        "prompt_str": prompt_str,
        "response_str": response_str,
        "ground_truth": ground_truth,
        "data_source": data_source
    }


class NaiveRewardManager:
    """The threaded reward manager."""
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", num_threads=16) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self.num_threads = num_threads
        # 用于打印的锁，防止多线程打印混乱
        self.print_lock = threading.Lock()

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        # 收集所有结果的列表，按索引存储确保顺序
        results = [None] * len(data)
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(
                    process_item, 
                    i, 
                    data, 
                    self.tokenizer, 
                    self.compute_score, 
                    self.reward_fn_key
                ): i for i in range(len(data))
            }
            
            # 处理完成的任务结果
            for future in concurrent.futures.as_completed(future_to_idx):
                result = future.result()
                # 存储结果到对应的索引位置，保证顺序
                results[result["idx"]] = result
        
        # 按原始顺序处理结果
        for result in results:
            i = result["idx"]
            valid_response_length = result["valid_response_length"]
            score = result["reward"]
            
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            data_source = result["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # # 使用锁确保打印不会混乱
                # with self.print_lock:
                #     print("[prompt]", result["prompt_str"])
                #     print("[response]", result["response_str"])
                #     print("[ground_truth]", result["ground_truth"])
                #     if isinstance(score, dict):
                #         for key, value in score.items():
                #             print(f"[{key}]", value)
                #     else:
                #         print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor