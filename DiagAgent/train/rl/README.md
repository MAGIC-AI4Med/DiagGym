# DiaAgent RL Training Guideline

<img src="../../../assets/DiagAgent_training.png"/> 

This guide explains how to run the entire RL training pipeline, primarily based on the vLLM inference engine and VERL reinforcement learning framework. Before following this guide, you can SFT the model with [cold_start](../cold_start/).

## Environment Setup

You need to install specific versions of torch, vLLM, and VERL. The versions used in this experiment are as follows:

### 1. Install PyTorch
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install vLLM
```bash
pip install vllm==0.8.3
pip install flash-attn --no-build-isolation
```

### 3. Install VERL locally
```bash
cd /mnt3/wenyou/project_code/Agentic_RL/training_code/AgenticRL_250430_code/RLTraining/verl
pip install -e .
```

## Training Data Preparation

The preprocessing script is located at `verl/examples/data_preprocess/diagnose.py`. You need to convert your data into `.parquet` format for efficient loading by the main training program.

### Usage Example

Assuming your raw data files are stored in `./data/raw/` and you want to output the processed `.parquet` files to `./data/processed/`, use the following command:

```bash
python verl/examples/data_preprocess/diagnose.py \
    --local_dir ./data/processed \
    --diaggym_ref_data_path ./data/raw/diaggym_train_data.json \
    --diagagent_train_data_path ./data/raw/diagagent_train_data.json \
    --diagagent_test_data_path ./data/raw/diagagent_test_data.json
```

### Parameter Description

- `--local_dir`: Specifies the storage directory for generated `.parquet` files and other intermediate files. The script will create corresponding output folders in this directory.
- `--diaggym_ref_data_path`: Reference data path for the DiagGym model. Since DiagGym needs to predict examination results based on patient profiles (basic information, chief complaints, etc.), a dataset containing this information is required as reference. In this experiment, we directly use DiagGym's complete training data (`diaggym_train_data.json`) as reference data.
- `--diagagent_train_data_path`: Training set data path for the DiagAgent model (e.g., `diagagent_train_data.json`).
- `--diagagent_test_data_path`: Test set data path for the DiagAgent model (e.g., `diagagent_test_data.json`).

## Online Inference Setup

### DiagGym Inference Server

First, you need to start the DiagGym inference server. This is necessary because DiagGym interacts with DiagAgent during training. When DiagAgent requests examination results, DiagGym simulates reasonable examination outcomes based on the patient's basic information.

```bash
vllm serve Henrychur/DiagGym --api-key 1234 --port 8079 --served-model-name EHRGenerator --gpu-memory-utilization 0.95 -tp 8
```

**Important:** After startup, you need to modify the URL and port in `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`:

```python
def call_openai_diaggym(self, input_prompt):
    """Call OpenAI API to get response"""
    openai_api_key = "1234"

    openai_api_base_ls = [
        "http://127.0.0.1:8079/v1",  # Change this URL/port if needed
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
```

### Reward Score Computation

Reward score calculation is implemented using an auxiliary general-purpose LLM inference service. In this experiment, we use Qwen2.5-72B to accomplish this task, which includes:
- Computing diagnosis accuracy
- Computing F1 score for examination recommendations

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --served-model-name Judger --api-key 1234 --port 8082 --gpu-memory-utilization 0.95 -tp 8
```

**Important:** After startup, you need to modify the URL and port in `verl/utils/reward_score/diagnose.py`:

```python
def call_localvllm(prompt):
    """Call the OpenAI interface to get a response"""
    openai_api_key = "1234"
    openai_api_base_list = [
        "http://127.0.0.1:8082/v1"  # Change this URL/port if needed
    ]
    openai_api_base = random.choice(openai_api_base_list)

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=3000
    )
    messages = [{'role': 'user', 'content': prompt}]
    chat_response = client.chat.completions.create(
        model="Judger",
        messages=messages,
        max_tokens=1024,
        seed=42,
        temperature=0
    )
    return chat_response.choices[0].message.content
```

## VERL RL Training

To start the RL training process:

```bash
cd verl/examples/grpo_trainer
bash my_run_qwen2_5.sh
```

## Convert to HuggingFace Weights

Finally, since the produced checkpoint format differs from the standard HuggingFace format, you need to convert it to HuggingFace weights to enable loading with the transformers library.

### Convert Checkpoint to HuggingFace Format

```bash
python verl/scripts/model_merger.py \
    --backend fsdp \
    --tie-word-embedding \
    --hf_model_path ckpt/coldstart_Qwen2.5-7B_test_v2 \
    --local_dir AgenticRL_2/Diagnoser/RLTraining/ckpt/v9/checkpoints/global_step_50/actor \
    --target_dir AgenticRL_2/Diagnoser/RLTraining/ckpt/v9/model_weights/step_50
```

### Copy Configuration Files

Copy necessary configuration files (JSON files) from the base model to the converted checkpoint:

```bash
for file in ckpt/EHRGenerator_v2_Qwen2_5_7B_backup/*.json; do
    [ -f "$file" ] || continue  # Ensure it's a file
    filename=$(basename "$file")
    [ -f "AgenticRL_2/Diagnoser/RLTraining/ckpt/v9/model_weights/step_50/$filename" ] && continue  # Skip if target exists
    cp "$file" "AgenticRL_2/Diagnoser/RLTraining/ckpt/v9/model_weights/step_50"
done
```
