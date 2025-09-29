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


n_gpus=8
batch_size=128
env=game:GuessTheNumber-v0

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.prompts_per_rollout=128 \
    test_data.prompts_per_rollout=1 \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.max_length_per_device=8192 \
    actor.update_per_rollout=2 \
    actor.warmup_ratio=0.0 \
    actor.lr=1e-6 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.train_sampling_params.temperature=1.0 \
    rollout.env_path=async_single_turn.py \
    adv.global_norm=true \
    adv.norm_var=true \
    trainer.project=GEM \
    trainer.experiment_name=guess-the-number_qwen3-1.7b_reinforce \
    trainer.n_epochs=512 \
    trainer.save_freq=64