## Training with VeRL
[VeRL](https://github.com/volcengine/verl) can be easily integrated with GEM to train LLM agents. In this section, we first provide the installation guide then give a few examples.


```bash
# recommend python==3.10
conda create -n verl python==3.10 -y
conda activate verl

# git clone verl (make sure verl and gem are at the same level of directory)
git clone git@github.com:volcengine/verl.git && cd verl
git checkout 4aa02fe16

# install
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

### Reasoning Gym
<details>
<summary>Click Me for the Script</summary>

```bash
n_gpus=8
batch_size=128
env=rg:letter_counting

PYTHONUNBUFFERED=1 python -m examples.train_verl.train_verl \
    actor_rollout_ref.env.env_id=${env} \
    actor_rollout_ref.env.wrappers="" \
    actor_rollout_ref.env.num_env=16 \
    actor_rollout_ref.env.async_env=True \
    actor_rollout_ref.prompt_template=qwen3_general \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_batch_size=${batch_size} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=gem \
    trainer.experiment_name=verl-qwen3-1.7b-base-${env} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=9999999 \
    trainer.test_freq=9999999 \
    trainer.total_epochs=15
```

</details>

### Game

<details>
<summary>Click Me for the Script</summary>

```bash
n_gpus=8
batch_size=128
env=game:GuessTheNumber-v0

PYTHONUNBUFFERED=1 python -m examples.train_verl.train_verl \
    actor_rollout_ref.env.env_id=${env} \
    actor_rollout_ref.env.wrappers=concat \
    actor_rollout_ref.env.num_env=16 \
    actor_rollout_ref.env.async_env=True \
    actor_rollout_ref.prompt_template=qwen3_game \
    actor_rollout_ref.gamma=0.9 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_batch_size=${batch_size} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=gem \
    trainer.experiment_name=verl-qwen3-1.7b-base-${env} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=9999999 \
    trainer.test_freq=9999999 \
    trainer.total_epochs=15
```

</details>
