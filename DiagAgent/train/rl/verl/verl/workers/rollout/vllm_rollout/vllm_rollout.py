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
from copy import deepcopy
from typing import List

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from vllm import SamplingParams

from verl import DataProto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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


class vLLMRollout(BaseRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
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
        max_num_batched_tokens = int(self.config.get("max_num_batched_tokens", 8192))

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp")
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp
                )

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
        )

        max_model_len = (
            self.config.max_model_len if self.config.max_model_len else config.prompt_length + config.response_length
        )
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config else OmegaConf.to_container(deepcopy(config.engine_kwargs))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
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
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

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
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = output[0].to(idx.device)
            # log_probs = output[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                # log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.n > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
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

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

from typing import Union
def get_response_mask_last_eos(
    response_id: torch.Tensor, 
    eos_token: Union[int, List[int]] = 2, 
    dtype=torch.int64
):
    """
    mask掉最后一个eos_token之后的token，最后一个eos_token及其之前全部为1
    """
    # 1. 生成eos_mask
    eos_mask = torch.isin(response_id, torch.tensor(eos_token, device=response_id.device))
    # 2. 找到每一行最后一个eos_token的位置，没有则为最后一个位置
    idx = torch.arange(response_id.size(1), device=response_id.device)
    # 把eos_mask为False的位置变成-1，True的位置是原index
    eos_idx = eos_mask * idx + (~eos_mask) * -1
    # 最后一个eos_token的位置
    last_eos_pos = eos_idx.max(dim=1).values
    # 3. 生成mask，mask = idx <= last_eos_pos
    mask = idx.unsqueeze(0) <= last_eos_pos.unsqueeze(1)
    return mask.to(dtype)

from openai import OpenAI
class vLLMRolloutWithMultiTurn(vLLMRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.max_turns = 8
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        
    def call_openai(self, messages):
        """调用OpenAI API获取响应"""
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8099/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        chat_response = client.chat.completions.create(
            model="/mnt/hwfile/medai/qiupengcheng/AgenticRL/SynthesizeGen/SFT/results/EHRGenerator_v1",
            messages=messages
        )
        return chat_response.choices[0].message.content
        
    def request_examination_result(self, ehr_text, examination_name):
        instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Next I will give you a medical record, and a exmaniation name, you need to predict the result of the examination."
        
        input_prompt = "### Medical record: {text} \n\n### Examination name: {examination_name}".format(text=ehr_text, examination_name=examination_name)
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_prompt},
        ]
        
        return self.call_openai(messages)
    
    def check_if_stop(self, text):
        if "The available information is sufficient to make a diagnosis." in text:
            return True
        else:
            return False

    def extract_messages(self, input_text):
        """从Qwen2.5格式的输入文本中提取消息列表"""
        messages = []
        # 使用<|im_start|>和<|im_end|>拆分文本
        parts = input_text.split(self.im_start)
        
        for part in parts:
            if not part.strip():
                continue
                
            # 查找角色和内容分隔点
            if self.im_end in part:
                role_content = part.split('\n', 1)
                if len(role_content) == 2:
                    role = role_content[0].strip()
                    content = role_content[1].split(self.im_end, 1)[0].strip()
                    
                    messages.append({"role": role, "content": content})
        
        return messages

    def format_as_input(self, messages):
        """将消息列表格式化为Qwen2.5格式的模型输入文本"""
        formatted = []
        for msg in messages:
            formatted.append(f"{self.im_start}{msg['role']}\n{msg['content']}{self.im_end}")
        return "\n".join(formatted)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']


        # 提取meta_info中的text字段
        ehr_texts = prompts.meta_info.get('text', [None] * ori_input_ids.size(0))
        # 用于构建attention_mask
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.pad_token_id

        batch_size = ori_input_ids.size(0)

        # 解析输入ID
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        # 设置采样参数
        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # 如果是贪婪搜索，只生成1个响应
            }

        with self.update_sampling_params(**kwargs):
            # 为每个输入准备n份
            curr_inputs = []
            ehr_texts = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
                    ehr_texts.append(ehr_texts[i])
            init_inputs = [ids.copy() for ids in curr_inputs]
            
            # 跟踪每个输入的状态
            active_indices = list(range(len(curr_inputs)))
            
            # 记录每个样本的对话轮次
            turn_counts = [0] * len(curr_inputs)
            
            # 跟踪每个样本的当前消息历史
            message_histories = []
            for idx in range(len(curr_inputs)):
                input_text = self.tokenizer.decode(curr_inputs[idx])
                messages = self.extract_messages(input_text)
                message_histories.append(messages)
            
            # 收集每个rollout的结果掩码
            result_mask_list = [[] for _ in range(len(curr_inputs))]
            
            # 生成直到所有样本完成或达到最大轮次
            while active_indices:
                # 筛选出需要生成助手回复的样本
                assistant_indices = []
                for idx in active_indices:
                    if turn_counts[idx] < self.max_turns and message_histories[idx][-1]['role'] == 'user':
                        assistant_indices.append(idx)
                
                if not assistant_indices:
                    break  # 没有需要生成助手回复的样本
                
                # 为每个活跃样本准备当前输入
                active_inputs = []
                for idx in assistant_indices:
                    current_messages = message_histories[idx]
                    current_input_text = self.format_as_input(current_messages)
                    current_input_ids = self.tokenizer.encode(current_input_text)
                    active_inputs.append(current_input_ids)
                
                # 批量生成助手回复
                with self.update_sampling_params(n=1, max_tokens=self.config.max_tokens_per_turn):
                    outputs = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=active_inputs,
                        use_tqdm=False
                    )
                
                # 处理生成的助手回复
                new_user_indices = []  # 需要生成用户回复的样本索引
                examination_requests = []  # 需要请求的检查
                examination_ehrs = []  # 对应的EHR文本
                
                for i, idx in enumerate(assistant_indices):
                    output_ids = outputs[0][i].tolist()
                    
                    # 找到EOS或PAD标记的位置并截断
                    if self.tokenizer.eos_token_id in output_ids:
                        first_eos_idx = output_ids.index(self.tokenizer.eos_token_id)
                        output_ids = output_ids[:first_eos_idx+1]
                    
                    if self.tokenizer.pad_token_id in output_ids:
                        first_pad_idx = output_ids.index(self.tokenizer.pad_token_id)
                        output_ids = output_ids[:first_pad_idx]
                    
                    # 解码生成的文本
                    output_text = self.tokenizer.decode(output_ids)
                    
                    # 提取新生成的助手消息
                    new_messages = self.extract_messages(output_text)
                    if not new_messages or new_messages[-1]['role'] != 'assistant':
                        # 移除无效样本
                        continue
                    
                    assistant_message = new_messages[-1]
                    assistant_content = assistant_message['content']
                    
                    # 添加助手消息到历史
                    message_histories[idx].append(assistant_message)
                    
                    # 计算助手回复的token ID并设置掩码
                    assistant_response_ids = self.tokenizer.encode(
                        f"{self.im_start}assistant\n{assistant_content}{self.im_end}"
                    )
                    result_mask_list[idx].extend([1] * len(assistant_response_ids))
                    
                    # 更新输入序列
                    curr_inputs[idx].extend(assistant_response_ids)
                    
                    # 检查是否达到终止条件
                    if self.check_if_stop(assistant_content):
                        # 标记该样本已完成
                        turn_counts[idx] = self.max_turns  # 设为最大轮次表示完成
                    else:
                        # 尝试提取检查名称
                        try:
                            examination_name = assistant_content.split("I would like to request ", 1)[1].split(".", 1)[0]
                            
                            # 记录需要请求的检查
                            new_user_indices.append(idx)
                            examination_requests.append(examination_name)
                            examination_ehrs.append(ehr_texts[idx])
                        except:
                            # 无法提取检查名称，标记完成
                            turn_counts[idx] = self.max_turns
                
                # 批量请求检查结果
                if new_user_indices:
                    # 并行处理所有检查请求
                    examination_results = []
                    for ehr, exam in zip(examination_ehrs, examination_requests):
                        # 这里可以进一步优化为真正的并行请求
                        result = self.request_examination_result(ehr, exam)
                        examination_results.append(result)
                    
                    # 处理检查结果，添加为用户消息
                    for i, idx in enumerate(new_user_indices):
                        examination_name = examination_requests[i]
                        examination_result = examination_results[i]
                        
                        # 创建用户回复
                        user_content = f"Examination result for {examination_name}: {examination_result}"
                        user_message = {"role": "user", "content": user_content}
                        
                        # 计算用户回复的token并添加掩码（标记为0，排除在损失计算外）
                        user_response_ids = self.tokenizer.encode(
                            f"{self.im_start}user\n{user_content}{self.im_end}"
                        )
                        result_mask_list[idx].extend([0] * len(user_response_ids))
                        
                        # 更新输入序列和消息历史
                        curr_inputs[idx].extend(user_response_ids)
                        message_histories[idx].append(user_message)
                        
                        # 增加轮次计数
                        turn_counts[idx] += 1
                
                # 更新活跃索引：移除已完成的样本
                active_indices = [idx for idx in active_indices if turn_counts[idx] < self.max_turns]
                
                # 检查序列长度，如果有样本超过最大长度，提前截断并完成
                for idx in active_indices[:]:  # 使用切片创建副本以便在循环中修改
                    input_len = len(init_inputs[idx])
                    output_len = len(curr_inputs[idx]) - input_len
                    
                    if output_len >= self.config.response_length:
                        # 超过最大长度，需要截断
                        truncation_point = input_len + self.config.response_length
                        curr_inputs[idx] = curr_inputs[idx][:truncation_point]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                        # 将样本标记为完成
                        active_indices.remove(idx)
                        turn_counts[idx] = self.max_turns

            # 最终处理：确保所有样本都满足长度要求
            response_list = []
            result_mask_list_padded = []
            
            for idx in range(len(curr_inputs)):
                input_len = len(init_inputs[idx])
                output_ids = curr_inputs[idx][input_len:]
                result_mask = result_mask_list[idx]
                
                # 如果超出最大长度，截断
                if len(output_ids) > self.config.response_length:
                    output_ids = output_ids[:self.config.response_length]
                    result_mask = result_mask[:self.config.response_length]
                
                # 如果不足最大长度，填充
                if len(output_ids) < self.config.response_length:
                    pad_length = self.config.response_length - len(output_ids)
                    output_ids = output_ids + [pad_token_id] * pad_length
                    result_mask = result_mask + [0] * pad_length  # 填充部分不计入损失
                
                # 转换为张量
                response = torch.tensor(output_ids, device=ori_input_ids.device)
                mask = torch.tensor(result_mask, device=ori_input_ids.device)
                
                response_list.append(response)
                result_mask_list_padded.append(mask)
            
            # 堆叠结果
            response = torch.stack(response_list, dim=0)
            result_mask = torch.stack(result_mask_list_padded, dim=0)

        # 如果生成多个样本，处理输入重复
        if self.sampling_params.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.sampling_params.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
            batch_size = batch_size * self.sampling_params.n
        
        # 构建完整序列
        seq = torch.cat([ori_input_ids, response], dim=-1)

        # 构建位置编码
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # 构建注意力掩码
        response_attention_mask = get_response_mask_last_eos(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # 构建损失掩码：仅对模型生成的部分计算损失，忽略local model返回的部分
        loss_mask = result_mask * response_attention_mask

        # 构建返回批次
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # 完整的序列
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,  # 用于训练的损失掩码
            'position_ids': position_ids
        }, batch_size=batch_size)
        
        # 释放vllm缓存引擎
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        
        return DataProto(batch=batch)

class vLLMRolloutWithSearch(vLLMRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.search_mode = config.search_mode
        self.csv_file_path = config.search_source
        self.search_top_n = config.search_top_n
        self.searcher = bm25Searcher(csv_path=self.csv_file_path)
        self.search_token_ids = tokenizer.encode('</search>', add_special_tokens=False)
        # print("INITIALIZATION!!!!!!\n\n\n\n")



    def batch_search(self, query: Union[str, List[str]], top_n=5) -> List[str]:
        if len(query) == 0:
            return 'invalid query'

        if isinstance(query, str):
            query = [query]

        result_list = []
        for q in query:
            result = self.searcher.search_str(q, top_n)
            result_list.append(result)
        # print("YYYYYYYYEEEEEEEESSSSSSS\n\n\n\n")
        return result_list


    def search(self, query: str, top_n=5):
        if query == '':
            return 'invalid query'

        result = self.searcher.search_str(query, top_n)
        return result

    def extract_search_content(self, text: str) -> str:
        try:
            start_tag = '<search>'
            end_tag = '</search>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        # print("BEGIN GENERATION\n\n\n\n")
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)

        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = ori_input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
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
                'n': 1  # if greedy, only 1 response
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]

            # track the status of each input
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            # collect the result mask of each rollout
            result_mask_list = [[] for _ in range(len(curr_inputs))]
            # print("BEGIN ACTIVE INDICESn\n\n\n")
            # generate until all inputs are finished
            while active_indices:
                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                print("Generate Timing BEGIN ###################\n")
                start_time = time.time()
                # generate in batch, according to active max tokens
                with self.update_sampling_params(n=1, stop=['</search>'], max_tokens=max(active_max_tokens), detokenize=True):
                # print(self.search_token_ids)
                # print(type(self.search_token_ids[0]))
                # with self.update_sampling_params(n=1, stop_token_ids=self.search_token_ids, max_tokens=max(active_max_tokens)):
                # with self.update_sampling_params(**kwargs):
                # with self.update_sampling_params(max_tokens=max(active_max_tokens)):
                    outputs = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=active_inputs,
                        use_tqdm=False
                    )
                end_time = time.time()
                print("Generate Timing END ###################\n")
                print(f"Generate Time: {end_time - start_time:.2f} seconds")
                print("Search Timimg BEGIN ###################\n")
                start_time = time.time()
                # print("BEGIN SEARCHING\n\n\n\n\n")
                # collect the queries to search
                search_queries = []
                search_indices = []

                # process each output
                new_active_indices = []
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[0][i].tolist()
                    if self.tokenizer.eos_token_id in output_ids:
                        first_eos_idx = output_ids.index(self.tokenizer.eos_token_id)
                    else:
                        first_eos_idx = len(output_ids)

                    if self.tokenizer.pad_token_id in output_ids:
                        first_pad_idx = output_ids.index(self.tokenizer.pad_token_id)
                    else:
                        first_pad_idx = len(output_ids)

                    finish_reason = outputs[2][i]
                    stop_reason = outputs[3][i]

                    if finish_reason == 'stop' and isinstance(stop_reason, str) and '</search>' in stop_reason:               
                        # need to search
                        ## truncate from the first pad token
                        output_ids = output_ids[:first_pad_idx]
                        output_str = self.tokenizer.decode(output_ids)
                        ## process the search
                        search_content = self.extract_search_content(output_str)
                        search_queries.append(search_content)
                        search_indices.append(idx)
                        new_active_indices.append(idx)
                        ## update the current input
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == None:
                        # print("STOP NONE ###################\n\n\n")
                        # output eos, indicating finished; truncate from the first eos token
                        output_ids = output_ids[:first_eos_idx+1]
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == self.tokenizer.pad_token_id:
                        # print("STOP END ###################\n\n\n")
                        # for instruction model, there is a chance that the end is endoftext, not im_end, this case needs special handling
                        output_ids = output_ids[:first_pad_idx+1]
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'length':
                        # print("STOP LENGTH ###################\n\n\n")
                        # output is too long
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)

                # batch process the search requests
                if search_queries:
                    search_results = self.batch_search(search_queries, self.search_top_n)
                    for idx, result in zip(search_indices, search_results):
                        # update the output, add the search result
                        output_ids = self.tokenizer.encode(f" <refer>\n{result}\n</refer>")
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [0] * len(output_ids)

                # check if need to truncate for active indices
                length_checked_active_indices = []
                for idx in active_indices:
                    assert len(curr_inputs[idx]) - len(init_inputs[idx]) == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        curr_inputs[idx] = init_inputs[idx] \
                            + curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                active_indices = length_checked_active_indices
                end_time = time.time()
                print("Search Timimg END ###################\n")
                print(f"Search Time: {end_time - start_time:.2f} seconds")

            output_ids_list = []
            # collect the results
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    idx = i * self.sampling_params.n + j
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])

        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            response_list.append(response)
            result_mask_list_padded.append(result_mask)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        if self.config.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)