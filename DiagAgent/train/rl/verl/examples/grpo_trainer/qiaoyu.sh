PROJECT_ROOT="/remote-home/qiaoyuzheng/RareRL/DiagR1"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2
# 将src目录添加到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

PROJECT_NAME=RareDiag
EXPERIMENT_NAME=Test1
TRAIN_FILES=/remote-home/qiaoyuzheng/RareRL/DiagR1/data/processed_patient_data/train.parquet
TEST_FILES=/remote-home/qiaoyuzheng/RareRL/DiagR1/data/processed_patient_data/val.parquet
TRAIN_BATCH_SIZE=16
PROMPT_KEY=input
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
PROMPT_TEMPLATE_NAME=rare_search_template_sys
APPLY_CHAT=True
ACTOR_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
PPO_MINI_BATCH_SIZE=8
REWARD_MANAGER=rare_search
TOTAL_TRAINING_STEPS=1000
SAVE_PATH=


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.apply_chat=${APPLY_CHAT} \
    data.prompt_template_name=${PROMPT_TEMPLATE_NAME} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    reward_model.reward_manager=${REWARD_MANAGER} \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.rollout_save_path=/remote-home/qiaoyuzheng/RareRL/DiagR1/scripts/train/outputs \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log