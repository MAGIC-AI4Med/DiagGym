#!/usr/bin/env python3
"""
Radiology Report Generator

A tool for generating realistic radiology examination reports based on patient case summaries
using various LLM APIs. Analyzes the most frequent radiology exams and generates comprehensive
findings for each case.
"""

import argparse
import json
import os
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
import requests
from tqdm import tqdm
from openai import OpenAI


class RadiologyGenerator:
    """Base class for radiology report generation using LLM APIs."""
    
    SYSTEM_PROMPT_TEMPLATE = """You are an expert radiologist AI assistant specialized in generating realistic radiology examination results. Your task is to analyze patient information and generate comprehensive radiology findings.

CRITICAL FORMATTING REQUIREMENTS:
- Generate a detailed and realistic radiology findings section for the specified examination
- Include relevant anatomical findings
- Use appropriate medical terminology and standard radiological language
- Provide specific details that would be clinically relevant
- Ensure the findings are consistent with the patient's clinical presentation

For the examination "{exam_name}", generate a comprehensive radiology report that includes:
1. Detailed findings of anatomical structures
2. Any abnormalities or normal variations observed

The report should be professional, detailed, and clinically appropriate."""

    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 use_custom_api: bool = False,
                 custom_url: Optional[str] = None) -> None:
        """
        Initialize the radiology generator.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            api_base: Base URL for OpenAI-compatible API
            use_custom_api: Whether to use custom API endpoint
            custom_url: Custom API endpoint URL
        """
        self.model_name = model_name
        self.use_custom_api = use_custom_api
        
        if use_custom_api:
            self.url = custom_url
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        else:
            self.client = OpenAI(api_key=api_key, base_url=api_base)

    def inference(self, context: str, exam_name: str) -> Optional[str]:
        """
        Generate radiology findings for a specific examination.
        
        Args:
            context: Patient case summary and diagnosis
            exam_name: Name of the radiology examination
            
        Returns:
            Generated radiology findings or None if failed
        """
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(exam_name=exam_name)
        
        query = f"""Patient Case Summary:
{context}

Radiology Examination to Generate: {exam_name}

Please generate comprehensive and realistic radiology findings for this examination. The findings should be detailed, professional, and consistent with the patient's clinical presentation described above."""

        if self.use_custom_api:
            return self._inference_custom(system_prompt, query)
        else:
            return self._inference_openai(system_prompt, query)

    def _inference_openai(self, system_prompt: str, query: str) -> Optional[str]:
        """Make inference using OpenAI-compatible API."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
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

    def _inference_custom(self, system_prompt: str, query: str) -> Optional[str]:
        """Make inference using custom API endpoint."""
        max_retries = 3
        base_delay = 2
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 1.0,
            "max_tokens": 16324,
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    data=json.dumps(data),
                    timeout=90
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


class RadiologyAnalyzer:
    """Analyzer for radiology examination data and generation."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the analyzer.
        
        Args:
            dataset_path: Path to the dataset JSON file
        """
        self.dataset_path = dataset_path
        self.data = self._load_dataset()
        
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset from JSON file."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_top_radiology_exams(self, top_n: int = 5) -> List[str]:
        """
        Extract the most frequent radiology examination names.
        
        Args:
            top_n: Number of top exams to extract
            
        Returns:
            List of top examination names
        """
        print(f"\nExtracting Top {top_n} Frequent Radiology Exam Names...")
        print(f"Loaded {len(self.data)} cases")
        
        radiology_counter = Counter()
        
        for note_id, note_data in self.data.items():
            for event in note_data.get('events', []):
                if event.get('source') == 'radiology':
                    exam_name = event.get('data', {}).get('label', 'Unknown').strip()
                    radiology_counter[exam_name] += 1
        
        top_exams = radiology_counter.most_common(top_n)
        top_names = []
        
        print("\nTop Radiology Examinations:")
        for rank, (exam_name, count) in enumerate(top_exams, 1):
            print(f"{rank}. {exam_name}: {count} occurrences")
            top_names.append(exam_name)
        
        return top_names
    
    def collect_exam_specific_cases(self, exam_names: List[str]) -> Dict[str, List[Dict]]:
        """
        Collect cases for specific examination types.
        
        Args:
            exam_names: List of examination names to collect
            
        Returns:
            Dictionary mapping exam names to case data
        """
        exam_cases = defaultdict(list)
        
        for case_id, case_data in self.data.items():
            # Skip incomplete cases
            if not all(k in case_data for k in ["text", "discharge_diagnosis", "events"]):
                continue
            
            # Build context
            discharge_diagnosis = case_data["discharge_diagnosis"]
            discharge_note = case_data["text"]
            
            if 'Physical Exam:' in discharge_note:
                context = discharge_note.split('Physical Exam:')[0].strip()
            else:
                context = discharge_note
            
            context += "\n\nFinal Diagnosis:\n" + discharge_diagnosis
            context += "\nThe following summarizes the results from the patient's medical examination:\n"
            
            # Find radiology exams in this case
            case_radiology_exams = {}
            for event in case_data["events"]:
                if event.get('source') == 'radiology':
                    exam_name = event.get('data', {}).get('label', 'Unknown').strip()
                    if exam_name in exam_names:
                        original_result = event.get('data', {}).get('lab_data', '')
                        case_radiology_exams[exam_name] = original_result
            
            # Add case to each relevant exam
            for exam_name, original_result in case_radiology_exams.items():
                exam_cases[exam_name].append({
                    'case_id': case_id,
                    'context': context,
                    'original_result': original_result
                })
        
        return dict(exam_cases)


def process_single_radiology_inference(args: Tuple) -> Dict[str, Any]:
    """
    Process a single radiology inference task.
    
    Args:
        args: Tuple of (sample, exam_name, generator)
        
    Returns:
        Dictionary with inference results
    """
    sample, exam_name, generator = args
    result = generator.inference(sample['context'], exam_name)
    
    return {
        'exam_name': exam_name,
        'case_id': sample['case_id'],
        'context': sample['context'],
        'prediction': result,
        'original_result': sample.get('original_result', '')
    }


def run_radiology_generation(
    analyzer: RadiologyAnalyzer,
    generator: RadiologyGenerator,
    exam_names: List[str],
    max_workers: int = 16
) -> Dict[str, List[Dict]]:
    """
    Run radiology report generation for specified examinations.
    
    Args:
        analyzer: RadiologyAnalyzer instance
        generator: RadiologyGenerator instance
        exam_names: List of examination names
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary of generation results by exam name
    """
    # Collect cases for each exam type
    exam_cases = analyzer.collect_exam_specific_cases(exam_names)
    
    # Prepare generation tasks
    generation_tasks = [
        (case_sample, exam_name, generator)
        for exam_name, cases in exam_cases.items()
        for case_sample in cases
    ]
    
    if not generation_tasks:
        print("No generation tasks found")
        return {}
    
    print(f"\nProcessing {len(generation_tasks)} generation tasks...")
    
    # Execute generation with thread pool
    generation_results = defaultdict(list)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_radiology_inference, task): task
            for task in generation_tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Generating radiology reports"):
            try:
                result = future.result()
                if result and result['prediction']:
                    generation_results[result['exam_name']].append({
                        'case_id': result['case_id'],
                        'context': result['context'],
                        'prediction': result['prediction'],
                        'original_result': result['original_result']
                    })
            except Exception as e:
                print(f"Task failed: {e}")
    
    return dict(generation_results)


def save_generation_results(
    results: Dict[str, List[Dict]],
    exam_names: List[str],
    model_name: str,
    output_file: str
) -> str:
    """
    Save generation results to JSON file.
    
    Args:
        results: Generation results dictionary
        exam_names: List of examination names
        model_name: Name of the model used
        output_file: Path to output file
        
    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    output_data = {
        'model_name': model_name,
        'top_exam_names': exam_names,
        'total_generations': sum(len(r) for r in results.values()),
        'exam_count': len(results),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'generation_settings': {
            'temperature': 1.0,
            'max_tokens': 16324,
            'method': 'case_specific'
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total generations: {output_data['total_generations']}")
    print(f"Examinations covered: {output_data['exam_count']}")
    
    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Radiology Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using DeepSeek API
  %(prog)s --dataset data.json --model deepseek --api-key YOUR_KEY

  # Using local Qwen model
  %(prog)s --dataset data.json --model qwen72b --api-key dummy --api-base http://localhost:8081/v1

  # Extract top 10 exams with more workers
  %(prog)s --dataset data.json --model medgemma --top-n 10 --max-workers 32
        """
    )
    
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--model", required=True, 
                       choices=["deepseek", "medgemma", "qwen7b", "qwen72b"],
                       help="Model to use for generation")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--api-base", help="API base URL (for OpenAI-compatible models)")
    parser.add_argument("--top-n", type=int, default=5, 
                       help="Number of top exams to process (default: 5)")
    parser.add_argument("--max-workers", type=int, default=16,
                       help="Maximum parallel workers (default: 16)")
    parser.add_argument("--output-dir", default="radiology_generation_outputs",
                       help="Output directory (default: radiology_generation_outputs)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RadiologyAnalyzer(args.dataset)
    
    # Extract top examinations
    top_exam_names = analyzer.extract_top_radiology_exams(args.top_n)
    
    if not args.api_base:
        parser.error(f"--api-base is required for {args.model}")
    
    generator = RadiologyGenerator(
        model_name=args.model_name,
        api_key=args.api_key,
        api_base=args.api_base,
        use_custom_api=False
    )
    output_file = os.path.join(args.output_dir, args.model_name)
    
    # Run generation
    generation_results = run_radiology_generation(
        analyzer, generator, top_exam_names, args.max_workers
    )
    
    # Save results
    save_generation_results(
        generation_results, top_exam_names, 
        generator.model_name, output_file
    )


if __name__ == "__main__":
    main()