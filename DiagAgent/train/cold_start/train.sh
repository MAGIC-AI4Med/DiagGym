#!/bin/bash
# Script to launch supervised fine-tuning with DeepSpeed and Accelerate

accelerate launch --config_file deepspeed_config_zero2.yaml train.py \
    --model_name_or_path MODEL_PATH \
    --model_max_length 8192 \
    --train_root_path TRAIN_DATA_PATH \
    --system_prompt_path diagnose.txt \
    --bf16 True \
    --optim adamw_torch \
    --output_dir OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True