import os
import json
import time
import tqdm
import torch
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def load_instruction(txt_filepath):
    """Load instruction text from a given file path."""
    with open(txt_filepath, 'r', encoding='utf-8') as f:
        return f.read()


class APIDiagnoser:
    """
    Diagnoser for API-based inference.
    Supports OpenAI-compatible APIs with custom base_url, or raw HTTP requests.
    """

    def __init__(self, model_name, max_tokens=2048, temperature=0.0,
                 api_type="openai", api_key=None, api_base=None, api_url=None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_type = api_type
        self.api_key = api_key
        self.api_base = api_base
        self.api_url = api_url

        if self.api_type == "openai":
            if not self.api_key:
                raise ValueError("API key is required for api_type='openai'")
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        elif self.api_type == "requests":
            if not self.api_url or not self.api_key:
                raise ValueError("api_url and api_key are required for api_type='requests'")
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        else:
            raise ValueError("Invalid api_type. Choose 'openai' or 'requests'.")

    def _call_openai(self, messages):
        """Call an OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()

    def _call_requests(self, messages):
        """Call a custom API endpoint via requests."""
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        for attempt in range(5):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=json.dumps(data),
                    timeout=90,
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                if attempt == 4:
                    print(f"API request failed after retries: {e}")
                    return None
                time.sleep(2)

    def diagnose(self, messages):
        """Get single diagnosis."""
        messages[-1]['content'] += "\n\nCurrent infomation is not sufficient. In the next turn you still need to query more examination."
        if self.api_type == "openai":
            return self._call_openai(messages)
        return self._call_requests(messages)

    def batch_diagnose(self, conversations):
        """Batch diagnosis using thread pool."""
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self.diagnose, conv) for conv in conversations]
            results = []
            for future in futures:
                try:
                    results.append(future.result() or "Unable to generate response.")
                except Exception as e:
                    print(f"Batch error: {e}")
                    results.append("Unable to generate response.")
            return results


class TransformersDiagnoser:
    """Diagnoser using local HuggingFace Transformers models."""

    def __init__(self, model_name_or_path, max_tokens=2048, temperature=0.0, device="cuda"):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_messages(self, messages):
        """Format chat messages into a single prompt."""
        messages[-1]['content'] += "\n\nCurrent infomation is not sufficient. In the next turn you still need to query more examination."
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return text

    def diagnose(self, messages):
        """Generate a single response."""
        prompt = self._format_messages(messages)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        response_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()

    def batch_diagnose(self, conversations):
        """Generate responses in batch."""
        prompts = [self._format_messages(m) for m in conversations]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        all_responses = []
        for i in range(len(prompts)):
            response_ids = output_ids[i, inputs.input_ids.shape[1]:]
            all_responses.append(self.tokenizer.decode(response_ids, skip_special_tokens=True).strip())
        return all_responses



def load_dataset(data_path, instruction_path):
    """Load dataset from JSON file and prepare exams_dict"""
    INSTRUCTION = load_instruction(instruction_path)
    with open(data_path, 'r') as fp:
        testset_data = json.load(fp)
    processed_data = []
    for testset_idx, item in enumerate(testset_data):
        if 'recommended_exam_names' not in item.keys():
            continue

        case_summary = item['case_summary']
        stepwise_diagnostic_reasoning_timeline = item['stepwise_diagnostic_reasoning_timeline_list']
        messages = [{'role':'system', 'content':INSTRUCTION}, {'role':'user', 'content':case_summary}]
        
        try:
            for idx, step in enumerate(stepwise_diagnostic_reasoning_timeline):
                if idx == len(stepwise_diagnostic_reasoning_timeline) - 1:
                    final_diagnosis = step.split('The available information is sufficient to make a diagnosis.')[1].split('Diagnosis:')[1].split('Reason:')[0].strip()
                    reason = step.split('Reason: ')[1].strip()
                    messages.append({
                        'role': 'assistant', 
                        'content': f"The available information is sufficient to make a diagnosis. \n\n Diagnosis: {final_diagnosis}\nReason: {reason}"
                    })
                else:
                    current_diagnosis = step.split('Current diagnosis: ')[1].split('Based on the')[0].strip()
                    examination = "Based on the " + step.split('Based on the')[1].split('Reason:')[0].strip()
                    reason = step.split('Reason: ')[1].split('Test result:')[0].strip()
                    test_result = step.split('Test result: ')[1].strip()
                    
                    content = f"Current diagnosis: {current_diagnosis}\n{examination}\nReason: {reason}"
                    messages.append({'role': 'assistant', 'content': content})
                    
                    content = f"Here is the test result: {test_result}"
                    messages.append({'role': 'user', 'content': content})

            testset_data[testset_idx]['messages'] = messages
            processed_data.append(testset_data[testset_idx])
                
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    print(len(processed_data))
    return processed_data


def parse_args():
    parser = argparse.ArgumentParser(description="Process multi-turn diagnosis conversations")
    parser.add_argument('--diagnoser_type', choices=['api', 'transformers'], required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--instruction_path', required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--api_type', choices=['openai', 'requests'], default='openai')
    parser.add_argument('--api_key', type=str, help="API key for OpenAI-compatible or custom API")
    parser.add_argument('--api_base', type=str, help="Base URL for OpenAI-compatible API")
    parser.add_argument('--api_url', type=str, help="Custom API URL if api_type='requests'")
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.data_path, args.instruction_path)

    if args.diagnoser_type == 'api':
        diagnoser = APIDiagnoser(
            model_name=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            api_type=args.api_type,
            api_key=args.api_key,
            api_base=args.api_base,
            api_url=args.api_url
        )
    else:
        diagnoser = TransformersDiagnoser(
            model_name_or_path=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

    # Collect all assistant turns except the last
    conversations, metadata = [], []
    for item_idx, item in enumerate(dataset):
        messages = item['messages']
        for msg_idx, msg in enumerate(messages):
            if msg['role'] == 'assistant' and msg_idx != len(messages) - 1:
                conversations.append(messages[:msg_idx])
                metadata.append({
                    'item_idx': item_idx,
                    'msg_idx': msg_idx,
                    'note_id': item.get('note_id', f'item_{item_idx}'),
                    'ground_truth': msg['content'],
                    'is_final_turn': msg_idx == len(messages) - 1
                })

    results = []
    for i in tqdm.tqdm(range(0, len(conversations), args.batch_size)):
        batch_convs = conversations[i:i + args.batch_size]
        batch_meta = metadata[i:i + args.batch_size]
        outputs = diagnoser.batch_diagnose(batch_convs)
        for out, meta in zip(outputs, batch_meta):
            if args.verbose:
                print(f"Item {meta['item_idx']} Turn {meta['msg_idx']}")
                print(f"Generated: {out[:100]}...")
                print(f"Ground Truth: {meta['ground_truth'][:100]}")
            results.append({
                **meta,
                'generated_response': out,
                'input_messages': batch_convs
            })

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {args.save_path}")


if __name__ == '__main__':
    main()
    
    
'''
python process_exam.py \
    --diagnoser_type api \
    --api_type openai \
    --api_key sk-xxxx \
    --api_base https://api.openai.com/v1 \
    --model_path gpt-4o-mini \
    --data_path ./data/testset.json \
    --instruction_path ./instructions/diagnose.txt \
    --save_path ./outputs/results_api.json
    
python process_exam.py \
    --diagnoser_type api \
    --api_type requests \
    --api_key mytoken123 \
    --api_url https://example.com/v1/chat/completions \
    --model_path MyModel \
    --data_path ./data/testset.json \
    --instruction_path ./instructions/diagnose.txt \
    --save_path ./outputs/results_custom.json


python process_exam.py \
    --diagnoser_type transformers \
    --model_path /path/to/local/model \
    --data_path ./data/testset.json \
    --instruction_path ./instructions/diagnose.txt \
    --save_path ./outputs/results_local.json \
    --batch_size 8 \
    --verbose
'''