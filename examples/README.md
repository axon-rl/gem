# Reinforcement Learning with GEM

## Training with Oat
[Oat](https://github.com/sail-sg/oat) is natively supported as the RL framework to integrate with GEM.
Before you start the experiments, you could install the library using:
```bash
# requires python==3.10
conda create -n oat python==3.10 -y
conda activate oat
# install
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
    --norm_return \
    --gpus 8 \
    --zero_stage 2 \
    --gradient-checkpointing \
    --rollout_batch_size 128 \
    --num_env 4 \
    --async_env \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-4B-Base \
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
    --wb-run-name oat-Qwen3-4b-base-math:Math12K \
    --wb_project gem \
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
    --norm_return \
    --gpus 8 \
    --zero_stage 2 \
    --gradient-checkpointing \
    --rollout_batch_size 128 \
    --num_env 4 \
    --async_env \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-4B-Base \
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
+   --wb-run-name oat-Qwen3-4b-base-math:Math12K-python-tool \
    --wb_project gem \
    --debug
```
</details>

### General QA (with Tool)

In this section, we show examples of training LLMs as general and multi-hop question-answering agents, with and without **search tool usage**. Note that we can train our model on different QA datasets by specifying a different `--env_id` similar to math environment.

#### Training Questing-Answering Agents Using Natural Languages

<details>
<summary>Click Me for the Script</summary>

```bash
python examples/train_oat.py \
    --env_id qa:HotpotQA \
    --wrappers "concat_chat" \
    --prompt_template "no" \
    --gamma 1.0 \
    --norm_return \
    --gpus 8 \
    --zero_stage 2 \
    --gradient-checkpointing \
    --rollout_batch_size 128 \
    --num_env 4 \
    --async_env \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-4B-Base \
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
    --max_model_len 5120 \
    --generate_max_length 500 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps -1 \
    --save_steps -1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 500 \
    --max_train 65000 \
    --max_save_num 30 \
    --use-wb \
    --wb-run-name oat-Qwen3-4b-base-qa:HotpotQA \
    --wb_project gem \
    --debug
```
</details>

#### Training Questing-Answering Agents Using *Search Tools*

<details>
<summary>Click Me for the Script</summary>

In this example we use the local dense retriever provided in the search-R1 as the search engine. Detailed instructions are in the [search-R1 documents](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md). 

Download the indexing and corpus: 

```bash
save_path=/the/path/to/save
huggingface-cli download PeterJinGo/wiki-18-corpus --repo-type dataset --local-dir $save_path
huggingface-cli download PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir $save_path

gzip -d $save_path/wiki-18.jsonl.gz
cat $save_path/part_* > $save_path/e5_HNSW64.index
```

Run local retriever and start training: 
```bash
# before run the script below, change SAVE_PATH_RETRIEVER in start_retrieval_server.sh
#   to the dir where you download indexing and corpus 
export SEARCH_URL="http://localhost:8000/retrieve"

bash examples/start_retrieval_server.sh

python examples/train_oat.py \
    --env_id qa:HotpotQA \
+   --wrappers "search_tool_no_int_reward,concat_chat" \
    --prompt_template "no" \
    --gamma 1.0 \
    --norm_return \
    --gpus 8 \
    --zero_stage 2 \
    --gradient-checkpointing \
    --rollout_batch_size 128 \
    --num_env 4 \
    --async_env \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-4B-Base \
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
    --max_model_len 5120 \
    --generate_max_length 500 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps -1 \
    --save_steps -1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 500 \
    --max_train 65000 \
    --max_save_num 30 \
    --use-wb \
+   --wb-run-name oat-Qwen3-4b-base-qa:HotpotQA-search-tool \
    --wb_project gem
```
</details>

### Game

In this section we show examples of training agents to solve multi-turn language games. Note that we set the discount factor `gamma=0.9` to encourage solutions with shorter horizon lengths, which are generally preferred for some strategic games (i.e., the agent accomplishes goals faster).

<details>
<summary>Click Me for the Script</summary>

```bash
python train.py \
    --env_id game:GuessTheNumber-v0 \
    --prompt_template qwen3_game \
    --wrappers concat \
    --gamma 0.9 \
    --norm_return \
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
    --wb-run-name oat-qwen3-1.7b-base-game:GuessTheNumber-v0 \
    --wb_project gem
```

</details>

### Reasoning Gym

In this section we show examples of training agents to solve procedurally generated reasoning intensive tasks originally from ReasoningGym.

<details>
<summary>Click Me for the Script</summary>

```bash
python train.py \
    --use_fused_lm_head \
    --env_id rg:arc_1d \
    --prompt_template qwen3_general \
    --gamma 1 \
    --gpus 8 \
    --zero_stage 2 \
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
    --wb-run-name oat-qwen3-1.7b-base-rg:arc_1d \
    --wb_project gem
```

</details>

### Code

In this section we show examples of training agents to solve coding tasks with the programs executed in independent environment. By default we use the same environment and execute the codes using a subprocess; but `bwrap` can be used for more secure sandboxing.

<details>
<summary>Click Me for the Script</summary>

```bash
python train.py \
    --env_id code:PrimeIntellect15k \
    --prompt_template code \
    --apply_chat_template \
    --gamma 1 \
    --gpus 8 \
    --gradient-checkpointing \
    --num_samples 1 \
    --rollout_batch_size 128 \
    --num_envs 4 \
    --async_env \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-4B-Base \
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
    --wb-run-name oat-qwen3-4b-base-code:PrimeIntellect15k \
    --wb_project gem \
```
## Training with VeRL
[VeRL](https://github.com/volcengine/verl) can be easily integrated with GEM to train LLM agents. In this section, we first provide the installation guide then give a few examples.

</details>

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
