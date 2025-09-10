
n_gpus=8
batch_size=128
env=game:GuessTheNumber-v0

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)/RL2"

torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=null \
    train_data.prompts_per_rollout=64 \
    train_data.responses_per_prompt=1 \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.sp_size=2 \
    actor.max_length_per_device=8192 \
    actor.tis_coef=2.0 \
    actor.update_per_rollout=2 \
    actor.warmup_ratio=0.0 \
    actor.lr=1e-6 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.train_sampling_params.temperature=1.0 \
    rollout.env_path=async_env_game.py \
    rollout.max_turns=16 \
    rollout.dynamic_filtering=false \
    adv.estimator=reinforce \
    adv.global_norm=true \
    adv.norm_var=true \
    trainer.project=gem \
    trainer.experiment_name=gem-qwen3-1.7b-reinforce-guess-the-number \
    trainer.n_epochs=500 \
    trainer.test_freq=9999999 \
    trainer.save_freq=64 \
    trainer.use_wandb=true