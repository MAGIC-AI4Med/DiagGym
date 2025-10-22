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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
import time

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp
                )
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
        )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=True,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
    
        # print(f"Size of responses: {response.size()}")
        # print(f"Size of input_ids: {seq.size()}")
        # print(f"Size of attention_mask: {attention_mask.size()}")
        # print(f"Size of position_ids: {position_ids.size()}")
        # print(f"batch_size: {batch_size}")
        # print("non_tensor_batch:\n", non_tensor_batch)

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


from typing import List, Union
import torch
from openai import OpenAI
from vllm import SamplingParams
from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def get_response_mask_last_eos(
    response_id: torch.Tensor, 
    eos_token: Union[int, List[int]] = 2, 
    dtype=torch.int64
):

    if isinstance(eos_token, int):
        eos_token = [eos_token]
    eos_mask = torch.zeros_like(response_id, dtype=torch.bool)
    for token in eos_token:
        eos_mask = eos_mask | (response_id == token)
    
    idx = torch.arange(response_id.size(1), device=response_id.device)
    eos_idx = eos_mask * idx + (~eos_mask) * -1
    last_eos_pos = eos_idx.max(dim=1).values
    mask = idx.unsqueeze(0) <= last_eos_pos.unsqueeze(1)
    return mask.to(dtype)


def extract_examination_name(model_output_text):
    model_output_list = model_output_text.split('\n')
    request_text = model_output_list[1]
    examination_name = ""
    if 'needed:' in request_text:
        examination_name = request_text.split('needed:')[1].replace('.', '').strip()
    elif 'performed:' in request_text:
        examination_name = request_text.split('performed:')[1].replace('.', '').strip()
    else:
        raise ValueError("Invalid response format. Expected 'needed:' or 'performed:' in the text.")
    return examination_name


import random
class vLLMRolloutWithMultiTurn(vLLMRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.max_turns = 12
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        self.assistant_input_ids = self.tokenizer.encode(f"{self.im_start}assistant\n")
        
        self.SEP = "<SEP>"
        self.stop_tokens = [self.SEP, "<endoftext>"]

    def call_openai_diaggym(self, input_prompt):
        """调用OpenAI API获取响应"""
        openai_api_key = "1234"

        openai_api_base_ls = [
            "http://33.147.221.21:8079/v1",
            "http://33.148.43.85:8079/v1"
        ]

        openai_api_base = random.choice(openai_api_base_ls)
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=3000
        )
        for i in range(3):
            try:
                response = client.completions.create(
                    model="EHRGenerator",
                    prompt=input_prompt,
                    max_tokens=512,
                    temperature=1.0,
                    stop=self.stop_tokens
                )
                return response.choices[0].text.strip()
            except:
                return None
        
    def request_examination_result(self, ehr_text, examination_name, past_events_list=None):
        if past_events_list is None:
            past_events_list = []
            
        context = ehr_text + "\nThe following summarizes the results from the patient's medical examination:\n"
        
        if len(past_events_list) == 0:
            input_prompt = context + "Exam name:\n" + examination_name + "\nExam results:\n"
        else:
            input_prompt = context + self.SEP.join(past_events_list) + self.SEP + "Exam name:\n" + examination_name + "\nExam results:\n"

        for i in range(3):
            try:
                resp = self.call_openai_diaggym(input_prompt)
                return resp
            except Exception as e:
                print(f"OpenAI call failed: {str(e)}")
                resp = None
        return resp
        
    
    def check_if_stop(self, text):
        if "The available information is sufficient to make a diagnosis." in text:
            return True
        else:
            return False

    def extract_messages(self, input_text):
        """Extract message list from Qwen2.5 format input text"""
        messages = []
        # Splitting text using <|im_start|> and <|im_end|>
        parts = input_text.split(self.im_start)
        
        for part in parts:
            if not part.strip():
                continue
                
            if self.im_end in part:
                role_content = part.split('\n', 1)
                if len(role_content) == 2:
                    role = role_content[0].strip()
                    content = role_content[1].split(self.im_end, 1)[0].strip()
                    
                    messages.append({"role": role, "content": content})
        
        return messages

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']


        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.pad_token_id

        batch_size = ori_input_ids.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        
        ehr_texts = non_tensor_batch['ehr_text'].tolist()


        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1 
            }
        
        with self.update_sampling_params(**kwargs):
            curr_inputs = []
            curr_ehr_texts = []
            for i, input_ids in enumerate(idx_list):
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
                    curr_ehr_texts.append(ehr_texts[i] if i < len(ehr_texts) else None)
            init_inputs = [ids.copy() for ids in curr_inputs]
            
            active_indices = list(range(len(curr_inputs)))
            
            turn_counts = [0] * len(curr_inputs)
            
            message_histories = []
            for idx in range(len(curr_inputs)):
                input_text = self.tokenizer.decode(curr_inputs[idx])
                messages = self.extract_messages(input_text.replace('<|im_start|>assistant\n', ''))
                message_histories.append(messages)
            
            result_mask_list = [[] for _ in range(len(curr_inputs))]
            
            examination_histories = [[] for _ in range(len(curr_inputs))]

            while active_indices:
                assistant_indices = []
                for idx in active_indices:
                    if turn_counts[idx] < self.max_turns and message_histories[idx][-1]['role'] == 'user':
                        assistant_indices.append(idx)
                
                if not assistant_indices:
                    break 
                
                # Prepare the current input for each active sample
                vllm_inputs = []
                for idx in assistant_indices:
                    current_messages = message_histories[idx]
                    current_input_text = self.tokenizer.apply_chat_template(current_messages, add_generation_prompt=True, tokenize=False)
                    current_input_ids = self.tokenizer.encode(current_input_text)
                    vllm_inputs.append({"prompt_token_ids": current_input_ids})
                
                # Batch generate assistant replies
                with self.update_sampling_params(n=1, max_tokens=512):
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                
                # Processing the generated assistant response
                new_user_indices = []  # Need to generate sample index of user replies
                examination_requests = []  # Request recommended examination results
                examination_ehrs = []  # corresponding EHR Text
                
                for i, idx in enumerate(assistant_indices):
                    output_ids = outputs[i].outputs[0].token_ids
                    
                    # Decode the generated text ids
                    output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    new_message = {'role':'assistant', 'content': output_text}
                    
                    # add assistant message to history
                    message_histories[idx].append(new_message)
                    
                    # Calculate the token ID replied by the assistant and set the mask
                    if turn_counts[idx] == 0:
                        assistant_response_ids = self.tokenizer.encode(
                            f"{new_message['content']}{self.im_end}"
                        )
                        result_mask_list[idx].extend([1] * len(assistant_response_ids))                        
                        curr_inputs[idx].extend(assistant_response_ids)
                    else:
                        assistant_response_ids = self.tokenizer.encode(
                            f"{self.im_start}assistant\n{new_message['content']}{self.im_end}"
                        )
                        result_mask_list[idx].extend([0] * len(self.assistant_input_ids))
                        result_mask_list[idx].extend([1] * (len(assistant_response_ids) - len(self.assistant_input_ids)))
                        curr_inputs[idx].extend(assistant_response_ids)
                    
                    # Check whether the termination condition is met
                    if self.check_if_stop(new_message['content']):
                        # Mark this sample completed
                        turn_counts[idx] = self.max_turns 
                    else:
                        # try to extract the recommended examination name
                        try:
                            examination_name = extract_examination_name(new_message['content'])
                            new_user_indices.append(idx)
                            examination_requests.append(examination_name)
                            examination_ehrs.append(curr_ehr_texts[idx])
                        except:
                            # Unable to extract inspection name, marking completed
                            turn_counts[idx] = self.max_turns

                if new_user_indices:
                    examination_results = []
                    request_params = []
                    for i, idx in enumerate(new_user_indices):
                        ehr_text = examination_ehrs[i]
                        exam_name = examination_requests[i]
                        past_events = examination_histories[idx].copy()
                        request_params.append((ehr_text, exam_name, past_events))
                    with ThreadPoolExecutor(max_workers=64) as executor:
                        # Directly map in order, the result order is consistent with the input
                        examination_results = list(
                            executor.map(
                                lambda args: self.request_examination_result(*args),
                                request_params,
                            )
                        )

                    # Process the examination results and add them as user messages
                    for i, idx in enumerate(new_user_indices):
                        examination_name = examination_requests[i]
                        examination_result = examination_results[i]
                        history_entry = f"Exam name:\n{examination_name}\nExam results:\n{examination_result}"
                        examination_histories[idx].append(history_entry)
                
                        # Create User Response
                        user_content = f"Here is the test result: {examination_result}"
                        user_message = {"role": "user", "content": user_content}
                        
                        # Calculate the token of the user's reply and add a mask (marked as 0 and excluded from loss calculation)
                        user_response_ids = self.tokenizer.encode(
                            f"{self.im_start}user\n{user_content}{self.im_end}"
                        )
                        result_mask_list[idx].extend([0] * len(user_response_ids))
                        
                        # Update input sequence and message history
                        curr_inputs[idx].extend(user_response_ids)
                        message_histories[idx].append(user_message)
                        
                        turn_counts[idx] += 1
                
                # Update active index: remove completed samples
                active_indices = [idx for idx in active_indices if turn_counts[idx] < self.max_turns]
                
                # Check the sequence length. If any sample exceeds the maximum length, truncate it in advance and complete it.
                for idx in active_indices[:]:
                    input_len = len(init_inputs[idx])
                    output_len = len(curr_inputs[idx]) - input_len
                    
                    if output_len >= self.config.response_length:
                        # Exceeds the maximum length and needs to be truncated
                        truncation_point = input_len + self.config.response_length
                        curr_inputs[idx] = curr_inputs[idx][:truncation_point]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                        active_indices.remove(idx)
                        turn_counts[idx] = self.max_turns


            # Final processing: Ensure that all samples meet the length requirements
            response_list = []
            result_mask_list_padded = []
            
            for idx in range(len(curr_inputs)):
                input_len = len(init_inputs[idx])
                output_ids = curr_inputs[idx][input_len:]
                result_mask = result_mask_list[idx]
                
                # If the maximum length is exceeded, truncate
                if len(output_ids) > self.config.response_length:
                    output_ids = output_ids[:self.config.response_length]
                    result_mask = result_mask[:self.config.response_length]
                
                # If the maximum length is less than the specified value, padding
                if len(output_ids) < self.config.response_length:
                    pad_length = self.config.response_length - len(output_ids)
                    output_ids = output_ids + [pad_token_id] * pad_length
                    result_mask = result_mask + [0] * pad_length
                
                response_list.append(output_ids)
                result_mask_list_padded.append(result_mask)
            
            # Convert the list to a 2D tensor
            response = pad_2d_list_to_length(response_list, pad_token_id, max_length=self.config.response_length).to(ori_input_ids.device)
            result_mask = torch.tensor(result_mask_list_padded, device=ori_input_ids.device)

        # If generating multiple samples, repeat input 
        if self.sampling_params.n > 1 and do_sample:
            ori_input_ids = _repeat_interleave(ori_input_ids, self.sampling_params.n)
            attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
            position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            batch_size = batch_size * self.sampling_params.n
        
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
            
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        response_attention_mask = get_response_mask_last_eos(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Construct loss mask: calculate loss only for the part generated by the model, ignoring the part returned by the local model
        loss_mask = result_mask * response_attention_mask

        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # complete seq
            'attention_mask': attention_mask,
            'loss_mask': loss_mask, 
            'position_ids': position_ids
        }, batch_size=batch_size)

        non_tensor_batch = {
            'examination_histories': np.array(examination_histories, dtype=object),
            'turn_counts': np.array([min(count, self.max_turns) for count in turn_counts]),
        }

        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
