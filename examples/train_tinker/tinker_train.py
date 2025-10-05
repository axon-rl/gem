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

"""GEM ❤️ Tinker.

A basic RL implementation to train agents on GEM environments using Tinker backends.
"""

import asyncio
import logging
import pprint
import time
from typing import Literal

import chz
import numpy as np
import tinker
import torch
import wandb
from termcolor import colored
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-8B-Base"
    batch_size: int = 128
    learning_rate: float = 4e-5
    lora_rank: int = 32
    max_tokens: int = 2048
    seed: int = 0
    max_steps: int = 200

    env_id: str = "rg:simple_equations"
    num_env: int = 4  # number of parallel environments
    env_wrappers: str = (
        "concat"  # wrappers are typically used to concat chat history, etc.
    )
    template: Literal["qwen3_general", "qwen3_game", "no"] = "qwen3_general"

    gamma: float = 0.9
    use_rebn: bool = True

    wandb_project: str | None = None
    wandb_name: str | None = None
    log_dir: str | None = None


# Define a lightweight renderer following tinker's renderer logics
def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_game_no_think_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_general_template(question: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_no_template(observation: str) -> str:
    return observation


TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "qwen3_general": apply_qwen3_general_template,
    "no": apply_no_template,
}


def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    # Avoid gating of Llama 3 models:
    if model_name.startswith("meta-llama/Llama-3"):
        model_name = "baseten/Meta-Llama-3-tokenizer"
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


async def main(config: Config):
    # Setup logging
    wandb_name = (
        config.wandb_name or config.model_name.split("/")[-1] + f"_{config.env_id}"
    )
    wandb.init(
        project=config.wandb_project,
        config=chz.asdict(config),
        dir=str(config.log_dir) if config.log_dir else None,
        name=wandb_name,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)

    # Setup environment
    wrappers = get_wrapper_fns(config.env_wrappers, tokenizer=tokenizer)
    # init one env first, check if it has dataset; if so we avoid load from HF multiple times
    # by directly providing dataset when creating the env. (we can also use the gem.Env.spawn api).
    envs = [gem.make(config.env_id, seed=int(time.time_ns()), use_mp=False)]
    for i in range(config.num_env - 1):
        dataset = envs[0].dataset if hasattr(envs[0], "dataset") else None
        envs.append(
            gem.make(
                config.env_id,
                seed=int(time.time_ns()) * i,
                dataset=dataset,
                use_mp=False,
            )
        )

    for i in range(len(envs)):
        for wrapper in wrappers:
            envs[i] = wrapper(envs[i])

    # Setup agent (tinker training client)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_name, rank=config.lora_rank
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    async def collect_episode(sampling_client, env: gem.Env):
        transitions = []
        obs, _ = env.reset()
        while True:
            # 1) prepare observation
            obs = TEMPLATE_FACTORY[config.template](obs)  # templated string
            obs_tokens = tokenizer.encode(obs, add_special_tokens=False)

            # 2) sample an action from the policy
            try:
                sample_result = await sampling_client.sample_async(
                    prompt=types.ModelInput.from_ints(tokens=obs_tokens),
                    num_samples=1,
                    sampling_params=sampling_params,
                )
            except Exception:
                transitions = []
                break
            sampled_tokens = sample_result.sequences[0].tokens
            sampled_logprobs = sample_result.sequences[0].logprobs
            action = tokenizer.decode(sampled_tokens)

            # 3) step the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            obs = next_obs

            # 4) save into buffer
            transitions.append(
                {
                    "obs_tokens": obs_tokens,
                    "act_tokens": sampled_tokens,
                    "act_logprobs": sampled_logprobs,
                    "reward": reward,
                    "done": done,
                }
            )

            if done:
                break
        return transitions

    # Start agent-environment loop (Algo: https://arxiv.org/pdf/2510.01051#page=15.10):
    for policy_iteration_step in range(config.max_steps):
        print("=" * 10 + f" Step {policy_iteration_step} " + "=" * 10)
        metrics = {"step": policy_iteration_step}

        sampling_path = (
            training_client.save_weights_for_sampler(
                name=f"{policy_iteration_step:06d}"
            )
            .result()
            .path
        )
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_path
        )

        # collect episodes with parallel environments
        print(f"🎲 Start collecting episodes at step {policy_iteration_step}")
        st = time.time()
        episodes_buffer = []
        while True:
            batch_episodes = await asyncio.gather(
                *[collect_episode(sampling_client, env) for env in envs]
            )
            batch_episodes = [x for x in batch_episodes if x != []]
            episodes_buffer.extend(batch_episodes)
            if sum([len(ep) for ep in episodes_buffer]) >= config.batch_size:
                break
        metrics["time/sample"] = time.time() - st
        metrics["sampler/episode_return"] = np.mean(
            [
                sum(transition["reward"] for transition in episode)
                for episode in episodes_buffer
            ]
        )
        metrics["sampler/num_turns_per_episode"] = np.mean(
            [len(episode) for episode in episodes_buffer]
        )
        metrics["sampler/action_num_tokens"] = np.mean(
            [
                sum(len(transition["act_tokens"]) for transition in episode)
                for episode in episodes_buffer
            ]
        )
        metrics["sampler/num_episodes"] = len(episodes_buffer)

        # print at most two episodes for debugging purposes
        for n, episode in enumerate(episodes_buffer):
            print(f"----- episode {n} -----")
            for t, transition in enumerate(episode):
                obs = tokenizer.decode(transition["obs_tokens"])
                act = tokenizer.decode(transition["act_tokens"])
                obs = obs[:196] + "\n...\n" + obs[-200:] if len(obs) > 396 else obs
                act = act[:196] + "\n...\n" + act[-200:] if len(act) > 396 else act
                print(f"turn={t+1}")
                print(colored(obs, "blue"))
                print(colored(act, "light_red", attrs=["bold"]))
                print(
                    colored(
                        "reward=" + str(transition["reward"]),
                        "light_magenta",
                        attrs=["bold"],
                    )
                )
            if n > 0:
                break

        # prepare transitions
        transitions = []
        for episode in episodes_buffer:
            # One transition typically consists of (s, a, r).
            # Here we augment it with a Monte Carlo return to
            # serve as the advantage estimation.
            rewards = [transition["reward"] for transition in episode]
            # Compute returns
            cur = 0.0
            for i in reversed(range(len(rewards))):
                cur = rewards[i] + config.gamma * cur
                episode[i]["return"] = cur
            transitions.extend(episode)

        # return batch normalization (https://arxiv.org/pdf/2510.01051#page=5.73 shows it's effective)
        if config.use_rebn:
            returns = torch.tensor(
                [transition["return"] for transition in transitions]
            ).float()
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            for i, transition in enumerate(transitions):
                transition["return"] = returns[i].item()

        # subsample to make a constant batch size
        if len(transitions) > config.batch_size:
            transitions = np.random.choice(
                transitions, config.batch_size, replace=False
            )

        # prepare training datums compatible with Tinker API
        training_datums = []
        for transition in transitions:
            ob_len_m1 = len(transition["obs_tokens"]) - 1  # -1 due to shifting
            tokens = transition["obs_tokens"] + transition["act_tokens"]

            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            all_logprobs = [0.0] * ob_len_m1 + transition["act_logprobs"]
            all_advantages = [0.0] * ob_len_m1 + [transition["return"]] * (
                len(input_tokens) - ob_len_m1
            )
            assert (
                len(input_tokens)
                == len(target_tokens)
                == len(all_logprobs)
                == len(all_advantages)
            ), f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                },
            )
            training_datums.append(datum)

        # training step
        print(f"🎈 Start training at step {policy_iteration_step}")
        st = time.time()
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="importance_sampling"
        )
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        _ = optim_step_future.result()
        metrics["time/train"] = time.time() - st

        # compute policy entropy and sampler-learner difference
        act_token_logprobs = []
        act_token_diffs = []
        for i in range(config.batch_size):
            transition = transitions[i]
            train_output = fwd_bwd_result.loss_fn_outputs[i]
            act_token_logprobs.extend(transition["act_logprobs"])
            act_token_diffs.append(
                torch.tensor(transition["act_logprobs"])
                - train_output["logprobs"].to_torch()[
                    -len(transition["act_logprobs"]) :
                ]
            )
        act_token_diffs = torch.cat(act_token_diffs)
        kl_sample_train_v1 = act_token_diffs.mean().item()
        kl_sample_train_v2 = 0.5 * (act_token_diffs**2).mean().item()
        metrics["sampler/token_entropy"] = (
            -torch.tensor(act_token_logprobs).mean().item()
        )
        metrics["train/kl_sample_train_v1"] = kl_sample_train_v1
        metrics["train/kl_sample_train_v2"] = kl_sample_train_v2
        metrics.update(**{f"train/{k}": v for k, v in fwd_bwd_result.metrics.items()})

        pprint.pprint(metrics)
        wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
