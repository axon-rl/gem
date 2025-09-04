
apt-get install net-tools

cd /mnt/

git clone https://github.com/OpenRLHF/OpenRLHF.git
pip3 install -e .

pip install gem-llm

pip3 install math-verify loguru fastapi uvicorn httpx python-multipart aiohttp aiolimiter pysbd jsonlines coloredlogs pebble aiolimiter
pip3 install func_timeout sentencex requests_futures timeout_decorator flashtext pygments 

export WARMUP=0.0
export LR=1e-6
export KL=0.0
export N_ROLLOUT=1

cd ./OpenRLHF
chmod -R 777 ./examples/scripts/

ln -s ./OpenRLHF /openrlhf
cd /openrlhf/examples/scripts
chmod -R 777 /openrlhf/examples/scripts


export OUTPUT_PATH=YOUR_OUTPUT_PATH
mkdir ${OUTPUT_PATH}

export SAVE_PATH=${OUTPUT_PATH}/gem_agent_qwen3_1.7b_base_guess_number_fix-random2
export expname=${SAVE_PATH}

export PRETRAIN=Qwen/Qwen3-1.7B-Base/
export REF_PRETRAIN=Qwen3-1.7B-Base/
export DATA_PATH=./dataset/gem_guess_game.jsonl

export TENSORBOARD=${SAVE_PATH}/tensorboard/

mkdir ${SAVE_PATH}
mkdir ${TENSORBOARD}

export PATH=$HOME/.local/bin/:$PATH

set -x
if [ "$RANK" -eq 0 ]; then
    ray start --head --port=6379  --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus 8
    ifconfig net0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1 > $expname/node_ip.txt
    export MASTER_NODE=$(cat $expname/node_ip.txt)
    # sleep 2m
    set -x
    ray job submit --address="http://${MASTER_NODE}:8265/" \
        --runtime-env-json='{"working_dir": "/openrlhf"}' \
        -- python3 -m openrlhf.cli.train_ppo_ray \
        --ref_num_nodes 1 \
        --ref_num_gpus_per_node 1 \
        --actor_num_nodes 1 \
        --actor_num_gpus_per_node 1 \
        --vllm_num_engines 1 \
        --vllm_tensor_parallel_size 1 \
        --colocate_all_models \
        --vllm_gpu_memory_utilization 0.45 \
        --gamma 1.0 \
        --l2 0.01 \
        --eps_clip_low_high 0.22 0.28 \
        --advantage_estimator reinforce \
        --pretrain ${PRETRAIN} \
        --agent_func_path /openrlhf/examples/python/agent_func_multi_turn.py \
        --save_path ${SAVE_PATH} \
        --ckpt_path ${SAVE_PATH} \
        --save_hf_ckpt \
        --micro_train_batch_size 2 \
        --train_batch_size 128 \
        --micro_rollout_batch_size 2 \
        --rollout_batch_size 128 \
        --n_samples_per_prompt ${N_ROLLOUT} \
        --max_epochs 1 \
        --num_episodes 100000000 \
        --prompt_max_len 1024 \
        --max_samples 100000000 \
        --generate_max_len 10240 \
        --zero_stage 3 \
        --bf16 \
        --init_kl_coef ${KL} \
        --lr_warmup_ratio ${WARMUP} \
        --actor_learning_rate ${LR} \
        --critic_learning_rate 9e-6 \
        --prompt_data ${DATA_PATH} \
        --input_key query \
        --label_key label \
        --normalize_reward \
        --gradient_checkpointing \
        --use_dynamic_batch \
        --vllm_sync_backend nccl \
        --vllm_enable_sleep \
        --deepspeed_enable_sleep \
        --adam_offload \
        --flash_attn \
        --gradient_checkpointing \
        --packing_samples \
        --enforce_eager \
        --load_checkpoint \
        --save_steps 50 \
        --use_tensorboard ${TENSORBOARD} \
        --remote_rm_url 'agent'
else
    sleep 1m
    export MASTER_NODE=$(cat $expname/node_ip.txt)
    ray start --address="${MASTER_NODE}:6379"
fi
 
sleep 365d