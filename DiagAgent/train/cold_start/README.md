# DiaAgent SFT Training Guideline

Supervised Fine-Tuning (SFT) is used to cold-start the DiaAgent model before RL training. This guide explains the data format, environment setup, and how to launch the SFT pipeline using `accelerate` + DeepSpeed with the provided `train.py` and `MedicalMultiTurnDataset`.

## Prerequisites

- Python 3.10+
- PyTorch, Transformers, Accelerate, DeepSpeed, and optional FlashAttention 2

Recommended installs:
```bash
# PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Core libraries
pip install transformers>=4.44.0 accelerate>=1.0.0 deepspeed>=0.14.0 datasets

# Optional but recommended: FlashAttention 2 (matches train.py setting)
pip install flash-attn --no-build-isolation
```

## Repository Layout

- `Dataset.py`: Contains `MedicalMultiTurnDataset` and data collator for SFT
- `train.py`: Main SFT script using `transformers.Trainer`
- `run_sft.sh`: Convenience script to launch `accelerate`
- `deepspeed_config_zero2.yaml`: Accelerate/DeepSpeed config
- `diagnose.txt`: System prompt used as the conversation initializer

## Dataset and Labeling Logic

`MedicalMultiTurnDataset`:
- Splits each case into multiple conversations where each sample ends with an assistant’s response.
- Uses the tokenizer’s chat template (`tokenizer.apply_chat_template`) to:
  - Build the full example (source + last assistant reply)
  - Build the source-only part (conversation without the last assistant reply, with `add_generation_prompt=True`)
- Constructs labels by setting non-target tokens to `IGNORE_INDEX` (−100) so the loss only applies to the last assistant response.

Requirements:
- Your tokenizer must implement `apply_chat_template`.
- The tokenizer should have a valid `eos_token_id`.
- `model_max_length` controls truncation in both tokenization and the dataset.

## Launch SFT

Use `train.sh` to start training:

Replace placeholders:
- `MODEL_PATH`: Base model (e.g., `Qwen/Qwen2.5-7B-Instruct`)
- `TRAIN_DATA_PATH`: Path to your training JSON (e.g., `./data/train.json`)
- `OUTPUT_DIR`: Directory to save the fine-tuned model (e.g., `./ckpt/sft_diaagent`)
