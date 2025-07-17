# Reinforcement Learning with GEM

## Training with Oat
[Oat](https://github.com/sail-sg/oat) is natively supported as the RL framework to integrate with GEM.
Before you start the experiments, you could install the library using:
```bash
# requires python==3.10
pip install vllm==0.8.4 && pip install oat-llm==0.1.4
```
then, please patch `LD_LIBRARY_PATH` to avoid dependency errors:
```bash
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
```

Next we give example command lines to run experiments for training LLM to do math, code, language game and general QA, as well as to use tools like python or search for them.

> **_NOTE_**: All scripts below assume a single-node (8 GPUs) setup. You should modify the arguments following the example below to customize the training on different hardware setups.

```diff
GRADIENT_BATCH_SIZE=128

N_GPU=8 # change me to e.g., 1, 2, 4, 8

python examples/train_oat.py \
    ... \
+   --gpus $N_GPU \
+   --rollout_batch_size_per_device $((GRADIENT_BATCH_SIZE / N_GPU)) \
+   --pi_buffer_maxlen_per_device $((GRADIENT_BATCH_SIZE / N_GPU)) \
    ...
```

### Math (with Tool)

In this section we show examples of training LLMs as math solvers, with and without **python tool usage**. Note that we can train our model on different datasets by specifying a different `--env_id` (a distribution of math questions can be essentially treated as an environment).

#### Solving Math Problems Using Natural Languages

<details>
<summary>Click Me for the Script</summary>

```bash
python examples/train_oat.py \
    --env_id math:Math12K \
    --wrappers "concat_chat" \
    --prompt_template "no" \
    --gamma 1.0 \
    --norm_adv \
    --gpus 8 \
    --zero_stage 2 \
    --gradient-checkpointing \
    --rollout_batch_size 128 \
    --num_env 16 \
    --async_env \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen2.5-1.5B \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.45 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 12800 \
    --generate_max_length 4096 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps -1 \
    --save_steps -1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 65000 \
    --max_save_num 30 \
    --use-wb \
    --wb-run-name Qwen2.5-1.5b-base-math:Math12K \
    --wb_project goat \
    --debug
```
</details>

#### Solving Math Problems Using *Python Tools*

<details>
<summary>Click Me for the Script</summary>

```bash
python examples/train_oat.py \
    --env_id math:Math12K \
+   --wrappers "python_tool_no_int_reward,concat_chat" \
    --prompt_template "no" \
    --gamma 1.0 \
    --norm_adv \
    --gpus 8 \
    --zero_stage 2 \
    --gradient-checkpointing \
    --rollout_batch_size 128 \
    --num_env 16 \
    --async_env \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen2.5-1.5B \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.45 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 12800 \
    --generate_max_length 4096 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps -1 \
    --save_steps -1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 65000 \
    --max_save_num 30 \
    --use-wb \
+   --wb-run-name Qwen2.5-1.5b-base-math:Math12K-python-tool \
    --wb_project goat \
    --debug
```
</details>

### General QA (with Tool)

### Game

In this section we show examples of training agents to solve multi-turn language games. Note that we set the discount factor `gamma=0.9` to encourage solutions with shorter horizon lengths, which are generally preferred for strategic games (i.e., the agent accomplishes goals faster).

<details>
<summary>Click Me for the Script</summary>

```bash
python train.py \
    --env_id ta:GuessTheNumber-v0 \
    --wrappers concat \
    --gamma 0.9 \
    --norm_adv \
    --gpus 8 \
    --gradient-checkpointing \
    --num_samples 1 \
    --rollout_batch_size 128 \
    --num_envs 2 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-1.7B-Base \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.45 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 12800 \
    --generate_max_length 4096 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps -1 \
    --save_steps -1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 65000 \
    --max_save_num 30 \
    --use-wb \
    --wb-org axon-rl \
    --wb-run-name qwen3-0.6b-base-ta:GuessTheNumber-v0 \
    --wb_project goat \
    --debug
```

</details>

### Reasoning Gym

### Code

## Training with VeRL
[VeRL](https://github.com/volcengine/verl)

Next we show a few example scripts for training LLM agents using VeRL.

### Reasoning Gym
<details>
<summary>Click Me for the Script</summary>

```bash
n_gpus=8
batch_size=128

PYTHONUNBUFFERED=1 python -m examples.train_verl.train_verl \
    actor_rollout_ref.env.env_id=$1 \
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
    trainer.experiment_name=zichen-qwen3-1.7b-base-$1-norm_adv \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=9999999 \
    trainer.test_freq=9999999 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log

```

</details>
