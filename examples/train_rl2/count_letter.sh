#!/bin/bash

# GEM Environment PPO Training Example for RL2
# Uses configuration-driven rollout class selection
# Aligned with VeRL training configuration (only matching parameters)

# Configuration variables
n_gpus=8
batch_size=128
env=rg:letter_counting

# Set logging to DEBUG mode for detailed vectorized environment tracking
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)/RL2"

# Run PPO training with GEM environment
torchrun \
    --nproc_per_node=$n_gpus \
    -m RL2.trainer.ppo \
    \
    trainer.project=gem \
    trainer.experiment_name=rl2-qwen3-1.7b-${env} \
    trainer.n_epochs=500 \
    trainer.test_freq=9999999 \
    trainer.save_freq=9999999 \
    \
    train_data.responses_per_prompt=1 \
    train_data.prompts_per_rollout=1 \
    \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.lr=1e-6 \
    actor.max_length_per_device=8192 \
    actor.sp_size=2 \
    actor.update_per_rollout=2 \
    actor.warmup_ratio=0.0 \
    actor.tis_coef=2.0 \
    \
    adv.estimator=reinforce \
    adv.norm_var=true \
    adv.global_norm=true \
    \
    rollout.rollout_class=gem_rollout.GEMRollout \
    rollout.model_name=Qwen/Qwen3-1.7B-Base \
    rollout.tp_size=1 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    +rollout.gem_env.env_id=${env} \
    +rollout.gem_env.wrappers="" \
    +rollout.gem_env.num_env=16 \
    +rollout.gem_env.async_env=true \
    +rollout.gem_env.prompt_template=qwen3_general \
    +rollout.gem_env.rollout_batch_size=${batch_size} \
    +rollout.gem_env.max_model_len=12800