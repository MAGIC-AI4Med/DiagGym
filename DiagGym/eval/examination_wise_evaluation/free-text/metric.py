import json
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time


class RadiologyDiversityFIDAnalyzer:
    """
    Analyzer for evaluating diversity and quality (FID) of generated radiology reports
    compared to ground truth, using sentence embeddings.
    """

    def __init__(self, model_path="BioLORD-2023-C"):
        """
        Initialize the analyzer and load the BioLORD embedding model.

        Args:
            model_path (str): Path to the BioLORD model.
        """
        print(f"[INFO] Initializing RadiologyDiversityFIDAnalyzer...")
        print(f"[INFO] Loading BioLORD model from: {model_path}")
        self.model = SentenceTransformer(model_path)
        print("[INFO] BioLORD model loaded successfully.")

    def load_generation_results(self, json_file_path):
        """
        Load generated results from a model output JSON file.

        Args:
            json_file_path (str): Path to generated results JSON.

        Returns:
            dict: Mapping {exam_name: [generated_texts]}.
        """
        print(f"[INFO] Loading generation results from: {json_file_path}")
        if not os.path.exists(json_file_path):
            print(f"[ERROR] File not found: {json_file_path}")
            return {}

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results_data = data.get('results', data)
        generation_results = {}

        for exam_name, exam_results in results_data.items():
            if isinstance(exam_results, list):
                generated_texts = []
                for result in exam_results:
                    if isinstance(result, dict):
                        prediction = result.get('prediction', result.get('generated_text', ''))
                    else:
                        prediction = str(result)
                    if prediction:
                        generated_texts.append(prediction)

                if generated_texts:
                    generation_results[exam_name] = generated_texts
                    print(f"  {exam_name}: {len(generated_texts)} samples")

        total_results = sum(len(v) for v in generation_results.values())
        print(f"[INFO] Loaded {total_results} generated samples covering {len(generation_results)} exam types.")
        return generation_results

    def load_ground_truth(self, gt_file_path):
        """
        Load ground truth radiology reports from dataset.

        Args:
            gt_file_path (str): Path to ground truth JSON.

        Returns:
            dict: Mapping {exam_name: [ground_truth_texts]}.
        """
        print(f"[INFO] Loading ground truth data from: {gt_file_path}")
        if not os.path.exists(gt_file_path):
            print(f"[ERROR] Ground truth file not found: {gt_file_path}")
            return {}

        with open(gt_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        gt_results = defaultdict(list)
        for case_id, case_data in data.items():
            if "events" not in case_data:
                continue
            for event in case_data["events"]:
                if event['source'] == 'radiology':
                    exam_name = event['data'].get('label', 'Unknown').strip()
                    gt_text = event['data'].get('lab_data', '')
                    if gt_text:
                        gt_results[exam_name].append(gt_text)

        # Filter out exam types with fewer than 5 samples
        filtered = {k: v for k, v in gt_results.items() if len(v) >= 5}
        for exam_name, texts in filtered.items():
            print(f"  {exam_name}: {len(texts)} ground truth samples")

        total_gt = sum(len(v) for v in filtered.values())
        print(f"[INFO] Loaded {total_gt} ground truth samples covering {len(filtered)} exam types.")
        return filtered

    def get_embeddings(self, texts, batch_size=32):
        """
        Compute sentence embeddings for a list of texts using BioLORD.

        Args:
            texts (list): List of text strings.
            batch_size (int): Batch size for encoding.

        Returns:
            np.ndarray: Embedding matrix.
        """
        if not texts:
            return np.array([])
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)

    def calculate_diversity_pairwise_similarity(self, embeddings):
        """
        Calculate diversity score as 1 - mean cosine similarity between all text pairs.

        Args:
            embeddings (np.ndarray): Shape [n_samples, embedding_dim].

        Returns:
            float: Diversity score in [0,1], higher means more diverse.
        """
        if len(embeddings) < 2:
            return 0.0
        similarity_matrix = cosine_similarity(embeddings)
        similarities = [similarity_matrix[i, j] for i in range(len(embeddings)) for j in range(i+1, len(embeddings))]
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity

    def calculate_fid(self, embeddings1, embeddings2):
        """
        Calculate Fréchet Inception Distance (FID) between two embedding sets.

        Args:
            embeddings1 (np.ndarray): Generated embeddings.
            embeddings2 (np.ndarray): Ground truth embeddings.

        Returns:
            float: FID score (lower is better).
        """
        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return float('inf')
        mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
        sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
        if sigma1.ndim == 0:
            sigma1 = sigma1.reshape(1, 1)
        if sigma2.ndim == 0:
            sigma2 = sigma2.reshape(1, 1)
        diff_squared = np.sum((mu1 - mu2) ** 2)
        trace_sigma = np.trace(sigma1) + np.trace(sigma2)
        try:
            sqrt_sigma_product = sqrtm(sigma1.dot(sigma2))
            if np.iscomplexobj(sqrt_sigma_product):
                sqrt_sigma_product = sqrt_sigma_product.real
            trace_sqrt = np.trace(sqrt_sigma_product)
        except Exception:
            trace_sqrt = np.sqrt(np.trace(sigma1) * np.trace(sigma2))
        fid = diff_squared + trace_sigma - 2 * trace_sqrt
        return max(0, fid)

    def get_common_exam_types(self, models_config, gt_data):
        """
        Get common exam types across all models and ground truth.

        Args:
            models_config (dict): {model_name: json_file_path}.
            gt_data (dict): Ground truth mapping.

        Returns:
            set: Common exam types.
        """
        print("[INFO] Identifying common exam types...")
        all_model_exam_types = []
        for model_name, json_file_path in models_config.items():
            if not os.path.exists(json_file_path):
                continue
            generated_data = self.load_generation_results(json_file_path)
            if generated_data:
                model_exam_types = set(generated_data.keys())
                all_model_exam_types.append(model_exam_types)
                print(f"  {model_name}: {list(model_exam_types)}")
        if all_model_exam_types:
            common_exam_types = set.intersection(*all_model_exam_types) & set(gt_data.keys())
        else:
            common_exam_types = set()
        print(f"[INFO] Common exam types ({len(common_exam_types)}): {list(common_exam_types)}")
        return common_exam_types

    def analyze_ground_truth_diversity(self, gt_data, target_exam_types):
        """
        Analyze diversity of ground truth samples for target exam types.

        Args:
            gt_data (dict): Ground truth mapping.
            target_exam_types (set): Exam types to analyze.

        Returns:
            dict: Diversity results.
        """
        print(f"[INFO] Analyzing ground truth diversity for {len(target_exam_types)} exam types...")
        results = {'exam_results': {}, 'overall_diversity': 0.0, 'target_exam_types': list(target_exam_types)}
        diversity_scores = []
        for exam_name in target_exam_types:
            if exam_name not in gt_data:
                continue
            gt_texts = gt_data[exam_name]
            gt_embeddings = self.get_embeddings(gt_texts)
            diversity = self.calculate_diversity_pairwise_similarity(gt_embeddings)
            results['exam_results'][exam_name] = {'diversity': diversity, 'sample_count': len(gt_texts)}
            diversity_scores.append(diversity)
        results['overall_diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
        print(f"[INFO] Ground truth overall diversity: {results['overall_diversity']:.6f}")
        return results

    def analyze_single_model(self, json_file_path, gt_data, model_name, target_exam_types):
        """
        Analyze diversity and FID of a single model's results.

        Args:
            json_file_path (str): Path to model's JSON results.
            gt_data (dict): Ground truth mapping.
            model_name (str): Model name.
            target_exam_types (set): Exam types to analyze.

        Returns:
            dict: Analysis results.
        """
        print(f"[INFO] Analyzing model: {model_name}")
        generated_data = self.load_generation_results(json_file_path)
        if not generated_data:
            return None
        common_exams = target_exam_types & set(generated_data.keys()) & set(gt_data.keys())
        if not common_exams:
            return None
        results = {'model_name': model_name, 'exam_results': {}, 'overall_diversity': 0.0, 'overall_fid': 0.0}
        diversity_scores, fid_scores = [], []
        for exam_name in common_exams:
            generated_embeddings = self.get_embeddings(generated_data[exam_name])
            gt_embeddings = self.get_embeddings(gt_data[exam_name])
            diversity = self.calculate_diversity_pairwise_similarity(generated_embeddings)
            fid = self.calculate_fid(generated_embeddings, gt_embeddings)
            results['exam_results'][exam_name] = {'diversity': diversity, 'fid': fid}
            diversity_scores.append(diversity)
            fid_scores.append(fid)
        results['overall_diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
        results['overall_fid'] = np.mean(fid_scores) if fid_scores else float('inf')
        return results

    def analyze_all_models(self, models_config, gt_file_path):
        """
        Analyze all models for diversity and FID.

        Args:
            models_config (dict): {model_name: json_file_path}.
            gt_file_path (str): Path to ground truth file.

        Returns:
            tuple: (all_model_results, gt_diversity_results)
        """
        gt_data = self.load_ground_truth(gt_file_path)
        if not gt_data:
            return {}, {}
        common_exam_types = self.get_common_exam_types(models_config, gt_data)
        if not common_exam_types:
            return {}, {}
        gt_diversity_results = self.analyze_ground_truth_diversity(gt_data, common_exam_types)
        all_results = {}
        for model_name, json_file_path in models_config.items():
            if not os.path.exists(json_file_path):
                continue
            result = self.analyze_single_model(json_file_path, gt_data, model_name, common_exam_types)
            if result:
                all_results[model_name] = result
        return all_results, gt_diversity_results

    def save_results(self, all_results, gt_diversity_results, output_file="radiology_diversity_fid_analysis.json"):
        """
        Save analysis results to a JSON file.
        """
        summary_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_type': 'Radiology Generation Diversity & FID Analysis',
            'embedding_model': 'BioLORD-2023-C',
            'target_exam_types': gt_diversity_results.get('target_exam_types', []),
            'models_analyzed': list(all_results.keys()),
            'ground_truth_diversity': gt_diversity_results,
            'model_results': all_results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Results saved to: {output_file}")


def main():
    """
    Main entry point for diversity and FID analysis.
    """
    models_config = {
        'EHRGenerator': 'path/to/EHRGenerator.json',
        'DeepSeek-V3': 'path/to/deepseekv3.json',
        'MedGemma-27B': 'path/to/medgemma.json',
        'Qwen2.5-7B': 'path/to/qwen7b.json',
        'Qwen2.5-72B': 'path/to/qwen72b.json'
    }
    gt_file_path = 'path/to/test_data.json'

    analyzer = RadiologyDiversityFIDAnalyzer()
    all_results, gt_diversity_results = analyzer.analyze_all_models(models_config, gt_file_path)

    if not all_results:
        print("[ERROR] No models were successfully analyzed.")
        return

    analyzer.save_results(all_results, gt_diversity_results)
    print("[INFO] Analysis complete.")


if __name__ == '__main__':
    main()