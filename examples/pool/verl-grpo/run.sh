#!/bin/bash
# Recipe B on verl, the trainer's side: verl's fully-async GRPO trainer (pool_verl.py) with LoRA on
# Qwen2.5-0.5B-Instruct (a pool ModelRef at /inputs/base), fed by vLLM rollers through a pool queue.
# The batch matches Demo B (../async-grpo): 8 tasks x 2 prompts x 8 samples = 128 samples per step,
# one optimizer step per training step, lr 5e-5, no KL term.
set -euo pipefail
export PYTHONPATH="/work:$POOL_LIB${PYTHONPATH:+:$PYTHONPATH}"   # Ray's workers import pool_job and common too
PROMPTS=$((${TASKS_PER_STEP:-8} * ${PROMPTS_PER_TASK:-2}))
exec python /work/pool_verl.py \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=True \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.path="${POOL_INPUTS:-/inputs}/base" \
  actor_rollout_ref.model.lora_rank=16 actor_rollout_ref.model.lora_alpha=32 \
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR:-5e-5}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PROMPTS" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.use_rollout_log_probs=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 rollout.n=8 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
  rollout.nnodes=1 rollout.n_gpus_per_node=1 \
  async_training.trigger_parameter_sync_step=1 async_training.require_batches=1 \
  data.max_prompt_length=128 data.max_response_length=256 \
  trainer.logger='["console"]' trainer.project_name=recipe-b trainer.experiment_name=verl-pool \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=cuda \
  trainer.total_training_steps="${STEPS:-30}" trainer.val_before_train=False \
  trainer.test_freq=-1 trainer.save_freq=-1 trainer.resume_mode=disable \
  trainer.default_local_dir=/tmp/ckpt \
  ray_kwargs.ray_init.num_cpus="${RAY_CPUS:-6}" \
  hydra.run.dir=/tmp/hydra "$@"
