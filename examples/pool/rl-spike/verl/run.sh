#!/bin/bash
# A tiny GRPO run with verl inside a pool job: Qwen2.5-0.5B-Instruct (a pool ModelRef at /inputs/base),
# LoRA r=16, vLLM rollouts, synthetic arithmetic. Metrics land in /outputs.
set -euo pipefail
STEPS="${STEPS:-20}"
OUT="${POOL_OUTPUTS:-/outputs}"
MODEL="${POOL_INPUTS:-/inputs}/base"
python /work/make_data.py /tmp/data
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=/tmp/data/train.parquet data.val_files=/tmp/data/test.parquet \
  data.train_batch_size=32 data.max_prompt_length=128 data.max_response_length=256 \
  data.filter_overlong_prompts=True data.truncation=error \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank=16 actor_rollout_ref.model.lora_alpha=32 \
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=3e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  custom_reward_function.path=/work/reward.py custom_reward_function.name=compute_score \
  trainer.logger='["console"]' \
  trainer.project_name=s1 trainer.experiment_name=grpo-arith \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.total_training_steps="$STEPS" trainer.total_epochs=1 \
  trainer.val_before_train=True trainer.test_freq=10 trainer.save_freq=-1 \
  trainer.default_local_dir="$OUT/ckpt" hydra.run.dir=/tmp/hydra "$@" 2>&1 | tee /tmp/train.log
python /work/curve.py /tmp/train.log "$OUT"
