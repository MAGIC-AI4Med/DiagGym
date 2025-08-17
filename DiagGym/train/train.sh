#!/bin/bash
# export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:1024

accelerate launch --config_file deepspeed_config_zero3.yaml training.py \
    --model_name_or_path BASE_MODEL_NAME \
    --train_root_path TRAINSET_DATA_PATH \
    --eval_root_path TESTSET_DATA_PATH \
    --bf16 True \
    --optim adamw_torch \
    --output_dir OUTPUT_PATH \
    --model_max_length 8192 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --tf32 True \
    --torch_compile True \
    --dataloader_pin_memory True