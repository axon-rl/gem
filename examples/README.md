# RL Training with GEM

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

### Math
#### Solving Math Problems Using Natural Languages

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
    --ignore_no_eos \
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

#### Solving Math Problems Using *Python Tools*

```diff
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
    --ignore_no_eos \
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