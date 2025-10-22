import re
from openai import OpenAI

INSTRUCTION = '''# Task Description
You are a professional medical diagnosis evaluation system. Now, you will receive two diagnosis results: one is the diagnosis predicted by the model ([pred_diag]), and the other is the verified correct diagnosis ([gt_diag]). Your task is to judge whether the model-predicted diagnosis([pred_diag]) is correct.

When evaluating, please consider the following factors:
1.The same disease may have multiple aliases, for example, “Heart disease” may also be called “Cardiac disease”.
2.There may be diversity in language expression, for example, “heart attack” and “myocardial infarction” may refer to the same disease.
3.Only judge whether the diagnosis result is correct, information such as the cause of the disease, symptoms, and treatment recommendations are not included in the evaluation scope.
4.If the correct diagnosis[gt_diag] is included in the predicted diagnosis but some additional complications are mentioned, it is also considered correct

# Output Requirements
Only output your judgment result on the model-predicted [pred_diag] as “Correct|Wrong”, do not output any other content.

# Format to Follow:
[Correct|Wrong]

Below is the diagnosis result predicted by the model and the correct diagnosis:
[pred_diag]
{pred_diag}

[gt_diag]
{gt_diag}'''

import random
def call_localvllm(prompt):
    """Call the OpenAI interface to get a response"""
    openai_api_key = "1234"
    openai_api_base_list = [
        "http://127.0.0.1:8082/v1"
    ]
    openai_api_base = random.choice(openai_api_base_list)

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=3000
    )
    messages = [{'role':'user', 'content':prompt}]
    chat_response = client.chat.completions.create(
        model="Judger",
        messages=messages,
        max_tokens=1024,
        seed=42,
        temperature=0
    )
    return chat_response.choices[0].message.content


def get_diagnostic_decision_making_score(solution_str, ground_truth):
    for try_idx in range(10):
        try:
            input_prompt = INSTRUCTION.format(
                pred_diag=solution_str,
                gt_diag=ground_truth
            )
            response = call_localvllm(input_prompt)

            if 'correct' in response.lower():
                return 1
            else:
                return 0
        except:
            if try_idx == 9:
                return 0


def get_exam_recommendation_reward(key_exam_names, recommended_exam_names, method="direct"):
    """
    Calculates the reward score for exam recommendations.

    Args:
        key_exam_names: List of exam names with standard answers
        recommended_exam_names: List of recommended exam names
        method: Calculation method. "direct" for direct matching, "llm" for LLM matching

    Returns:
        F1 score
    """
    if method == "direct":
        return 0.5 * _calculate_direct_f1(key_exam_names, recommended_exam_names)
    elif method == "llm":
        return 0.5 * _calculate_llm_f1(key_exam_names, recommended_exam_names)
    else:
        raise ValueError("method must be 'direct' or 'llm'")

def _calculate_direct_f1(key_exam_names, recommended_exam_names):
    """Solution 1: Directly calculate the F1 indicator"""
    if not key_exam_names and not recommended_exam_names:
        return 1.0
    
    if not key_exam_names or not recommended_exam_names:
        return 0.0
    
    # Convert to a set for matching
    key_set = set([exam.lower().strip() for exam in key_exam_names])
    recommended_set = set([exam.lower().strip() for exam in recommended_exam_names])
    
    # Calculate intersection
    intersection = key_set.intersection(recommended_set)
    
    # Calculate precision and recall
    precision = len(intersection) / len(recommended_set) if recommended_set else 0
    recall = len(intersection) / len(key_set) if key_set else 0
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def _calculate_llm_f1(key_exam_names, recommended_exam_names):
    """Solution 2: Calculate F1 score using LLM"""
    if not key_exam_names and not recommended_exam_names:
        return 1.0
    
    if not key_exam_names or not recommended_exam_names:
        return 0.0
    
    # Calculate Precision: How many of the recommended checks are in the standard answer?
    precision_matches = _calculate_llm_precision_matches(key_exam_names, recommended_exam_names)
    precision = precision_matches / len(recommended_exam_names) if recommended_exam_names else 0
    
    # Calculate Recall: How many of the standard answers were recommended?
    recall_matches = _calculate_llm_recall_matches(key_exam_names, recommended_exam_names)
    recall = recall_matches / len(key_exam_names) if key_exam_names else 0
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

def _calculate_llm_precision_matches(key_exam_names, recommended_exam_names):
    """Calculate the number of matches using LLM"""
    prompt = f"""
Please determine how many of the recommended exams appear in the key exam list. Note that even if the expressions are different, if they refer to the same examination, they should be considered as matches.

Key exam list: {key_exam_names}
Recommended exam list: {recommended_exam_names}

Please analyze each item in the recommended exam list and determine if it has a corresponding item in the key exam list (even with different expressions).

Please only output the number of matches as an integer. For example: 3
"""
    
    for try_idx in range(10):
        try:
            response = call_localvllm(prompt)
            # Extract the number from the response
            numbers = re.findall(r'\d+', response.strip())
            if numbers:
                return int(numbers[0])
        except:
            if try_idx == 9:
                return 0
    
    return 0

def _calculate_llm_recall_matches(key_exam_names, recommended_exam_names):
    """Use LLM to calculate the number of matches for Recall"""
    prompt = f"""
Please determine how many of the key exams appear in the recommended exam list. Note that even if the expressions are different, if they refer to the same examination, they should be considered as matches.

Key exam list: {key_exam_names}
Recommended exam list: {recommended_exam_names}

Please analyze each item in the key exam list and determine if it has a corresponding item in the recommended exam list (even with different expressions).

Please only output the number of matches as an integer. For example: 2
"""
    
    for try_idx in range(10):
        try:
            response = call_localvllm(prompt)
            numbers = re.findall(r'\d+', response.strip())
            if numbers:
                return int(numbers[0])
        except:
            if try_idx == 9:
                return 0
    
    return 0

def get_exam_recommendation_reward_v1(key_exam_names, recommended_exam_names):
    """Solution 1: Directly calculate the F1 """
    return get_exam_recommendation_reward(key_exam_names, recommended_exam_names, method="direct")

def get_exam_recommendation_reward_v2(key_exam_names, recommended_exam_names):
    """Solution 2: Calculate F1 index using LLM"""
    return get_exam_recommendation_reward(key_exam_names, recommended_exam_names, method="llm")


def compute_score(output_str, ground_truth, extra_info):
    dialogue_complete_reward, outcome_accuracy_reward, exam_recommendation_reward = 0, 0, 0
    if 'The available information is sufficient to make a diagnosis' not in output_str:
        dialogue_complete_reward = 0
    else:
        dialogue_complete_reward = 0.1
        try:
            solution_str = output_str.split('The available information is sufficient to make a diagnosis.')[1].split('Reason:')[0].strip()
            outcome_accuracy_reward = get_diagnostic_decision_making_score(solution_str, ground_truth)
        except:
            outcome_accuracy_reward = 0
            dialogue_complete_reward = 0

    exam_recommendation_reward = get_exam_recommendation_reward_v2(extra_info['key_exam_names'], extra_info['examination_history'])

    return {
        "dialogue_complete_reward" : dialogue_complete_reward,
        'outcome_accuracy_reward' : outcome_accuracy_reward,
        "exam_recommendation_reward" : exam_recommendation_reward,
        "score": dialogue_complete_reward + outcome_accuracy_reward + exam_recommendation_reward
    }

