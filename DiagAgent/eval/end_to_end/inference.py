import os
import json
import time
import tqdm
import requests
import argparse
import pandas as pd
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_instruction(path):
    """Load instruction text from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_dataset_from_parquet(data_path):
    """Load dataset from parquet file."""
    df = pd.read_parquet(data_path)
    data = []
    for _, row in df.iterrows():
        extra_info = row['extra_info']
        data.append({
            'note_id': f"note_{extra_info['index']}",
            'case_summary': extra_info['question'],
            'text': extra_info['text'],
            'key_exam_names': list(extra_info.get('key_exam_names', [])),
            'final_diagnosis': extra_info['answer']
        })
    return data


class ExaminationGenerator:
    """Generate examination results based on EHR text and exam name."""
    def __init__(self, api_key, api_base, model_name="EHRGenerator"):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.SEP = "<SEP>"
        self.stop_tokens = [self.SEP, "<endoftext>"]

    def call_openai(self, prompt):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=3000)
        for attempt in range(3):
            try:
                resp = client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.0,
                    stop=self.stop_tokens
                )
                return resp.choices[0].text.strip()
            except Exception as e:
                print(f"Exam API call failed ({attempt+1}/3): {e}")
                if attempt == 2:
                    return None
        return None

    def request_examination_result(self, ehr_text, exam_name, past_events=None):
        if past_events is None:
            past_events = []
        context = ehr_text + "\nThe following summarizes the results from the patient's medical examination:\n"
        if not past_events:
            prompt = context + f"Exam name:\n{exam_name}\nExam results:\n"
        else:
            prompt = context + self.SEP.join(past_events) + self.SEP + f"Exam name:\n{exam_name}\nExam results:\n"
        resp = self.call_openai(prompt)
        return resp if resp else "Unable to generate examination result."


class APIDiagnoser:
    """Diagnoser using API (OpenAI-compatible or raw requests)."""
    def __init__(self, model, max_tokens, temperature, api_type, api_key, api_base=None, api_url=None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_type = api_type
        self.api_key = api_key
        self.api_base = api_base
        self.api_url = api_url
        if api_type == "openai":
            self.client = OpenAI(api_key=api_key, base_url=api_base)
        elif api_type == "requests":
            self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def _call_openai(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return resp.choices[0].message.content.strip()

    def _call_requests(self, messages):
        data = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        for attempt in range(5):
            try:
                r = requests.post(self.api_url, headers=self.headers, json=data, timeout=90)
                r.raise_for_status()
                return r.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                if attempt == 4:
                    print(f"API request failed: {e}")
                    return None
                time.sleep(2)

    def diagnose(self, messages):
        return self._call_openai(messages) if self.api_type == "openai" else self._call_requests(messages)

    def batch_diagnose(self, conversations):
        with ThreadPoolExecutor(max_workers=32) as ex:
            futures = [ex.submit(self.diagnose, c) for c in conversations]
            return [f.result() or "Unable to generate response." for f in futures]


class TransformersDiagnoser:
    """Diagnoser using local HuggingFace Transformers."""
    def __init__(self, model_path, max_tokens, temperature):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _format_messages(self, messages):
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return text + "\nCurrent diagnosis:"

    def diagnose(self, messages):
        prompt = self._format_messages(messages)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                          temperature=self.temperature if self.temperature > 0 else None,
                                          do_sample=self.temperature > 0,
                                          pad_token_id=self.tokenizer.pad_token_id)
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

    def batch_diagnose(self, conversations):
        prompts = [self._format_messages(m) for m in conversations]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                          temperature=self.temperature if self.temperature > 0 else None,
                                          do_sample=self.temperature > 0,
                                          pad_token_id=self.tokenizer.pad_token_id)
        res = []
        for i in range(len(prompts)):
            gen_ids = out_ids[i, inputs.input_ids.shape[1]:]
            res.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return res


def check_if_stop(text):
    return "The available information is sufficient to make a diagnosis." in (text or "")


def extract_examination_name(output_text):
    lines = output_text.split('\n')
    if len(lines) < 2:
        raise ValueError("Invalid format")
    line = lines[1]
    if 'needed:' in line:
        return line.split('needed:')[1].replace('.', '').strip()
    if 'performed:' in line:
        return line.split('performed:')[1].replace('.', '').strip()
    raise ValueError("No exam name found")


def process_case(diagnoser, exam_gen, item, instruction, max_turns, verbose=False):
    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": item['case_summary']}]
    exam_history = []
    for turn in range(max_turns):
        resp = diagnoser.diagnose(messages)
        if verbose:
            print(resp[:200] if resp else "No response")
        messages.append({"role": "assistant", "content": resp})
        if check_if_stop(resp) or turn == max_turns - 1:
            break
        try:
            exam_name = extract_examination_name(resp)
        except Exception as e:
            print(f"Exam name extraction failed: {e}")
            break
        exam_result = exam_gen.request_examination_result(item['text'], exam_name, exam_history)
        exam_history.append(f"Exam name:\n{exam_name}\nExam results:\n{exam_result}")
        messages.append({"role": "user", "content": f"Here is the test result: {exam_result}"})
    return {"note_id": item['note_id'], "messages": messages, "examination_history": exam_history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diagnoser_type', choices=['api', 'transformers'], required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--save_path', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--instruction_path', required=True)
    ap.add_argument('--max_turns', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max_tokens', type=int, default=2048)
    ap.add_argument('--api_type', choices=['openai', 'requests'], default='openai')
    ap.add_argument('--api_key', type=str)
    ap.add_argument('--api_base', type=str)
    ap.add_argument('--api_url', type=str)
    ap.add_argument('--exam_api_key', type=str, required=True)
    ap.add_argument('--exam_api_base', type=str, required=True)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print(f"Running {args.diagnoser_type} diagnoser")
    instruction = load_instruction(args.instruction_path)
    dataset = load_dataset_from_parquet(args.data_path)
    print(f"Loaded {len(dataset)} cases")

    if args.diagnoser_type == 'api':
        diagnoser = APIDiagnoser(args.model_path, args.max_tokens, args.temperature,
                                 args.api_type, args.api_key, args.api_base, args.api_url)
    else:
        diagnoser = TransformersDiagnoser(args.model_path, args.max_tokens, args.temperature)

    exam_gen = ExaminationGenerator(args.exam_api_key, args.exam_api_base)

    results = []
    for item in tqdm.tqdm(dataset):
        results.append(process_case(diagnoser, exam_gen, item, instruction, args.max_turns, args.verbose))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {args.save_path}")


if __name__ == '__main__':
    main()
    
'''
python dynamic_inference.py \
    --diagnoser_type api \
    --api_type openai \
    --api_key sk-xxx \
    --api_base https://api.deepseek.com/v1 \
    --model_path deepseek-chat \
    --data_path ./data/test.parquet \
    --instruction_path ./instructions/openai.txt \
    --exam_api_key sk-exam-xxx \
    --exam_api_base https://examapi.com/v1 \
    --save_path ./outputs/deepseek.json \
    --verbose
    
python dynamic_inference.py \
    --diagnoser_type transformers \
    --model_path /path/to/local/model \
    --data_path ./data/test.parquet \
    --instruction_path ./instructions/openai.txt \
    --exam_api_key sk-exam-xxx \
    --exam_api_base https://examapi.com/v1 \
    --save_path ./outputs/local_model.json
'''