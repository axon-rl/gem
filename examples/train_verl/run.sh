# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# User-specific =========
cd /home/aiops/liuzc
source ./.zshrc
conda activate verl
cd gem

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
    trainer.experiment_name=zichen-qwen3-1.7b-base-$1-norm_return \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=9999999 \
    trainer.test_freq=9999999 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log

# bash /home/aiops/liuzc/gem/examples/train_verl/run.sh rg:letter_counting
