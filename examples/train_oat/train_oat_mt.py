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

"""
Entry script of using OAT to RL-tune LLM agents on GEM environments.
"""

import functools
import json
import logging
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch.distributed as dist
import tree
import vllm
from oat.algorithms.ppo import PPOArgs
from oat.algorithms.ppo_multiturn import PPOMultiTurnActor, PPOMultiTurnLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import Transition
from oat.utils.ops import masked_sum
from torch.utils.data import Dataset

import gem
from gem.utils.parsing import extract_last_boxed_answer
from gem.wrappers.wrapper_factory import get_wrapper_fns

""" +=========================================+ """
""" 1. Defining constants used in our training. """
""" +=========================================+ """

# Invalid action to be sent to the env to trigger format error penalty.
INVALID_ACTION = "<｜INVALID_ACTION｜>"


def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_no_template(observation: str) -> str:
    return observation


def apply_qwen3_general_template(question: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_code_template(question: str) -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {question}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )


TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
}


""" +=================================================+ """
""" 2. Defining extra arguments/structure for training. """
""" +=================================================+ """


@dataclass
class Args(PPOArgs):
    # Environment settings
    env_id: str = "rg:leg_counting"
    num_env: int = 1
    wrappers: str = ""
    async_env: bool = False

    # Algorithm settings
    length_norm_constant: Optional[int] = None

    # Template settings
    prompt_template: Literal["qwen3_game", "no", "qwen3_general", "code"] = "qwen3_game"

    # Reward settings
    gamma: float = 1.0  # Discount factor for Monte Carlo returns
    whiten_adv: bool = True  # Return batch normalization

    # online evaluation settings
    eval_envs: str = None  # 'eval:AIME24|eval:MATH500'. See gem.envs
    eval_wrappers: str = ""
    eval_prompt_templates: str = "no"
    eval_async_env: bool = False
    eval_n: int = 1  # number of episodes to average for each env

    # Misc settings
    dump_experience_every: int = 1  # Dump experience data

    # Episode collection logic
    keep_generation_failed: bool = False  # Keep episodes with generation failures


""" +=======================================+ """
""" 3. Defining actor to collect experiences. """
""" +=======================================+ """


class Actor(PPOMultiTurnActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.args.seed += 233 ** (actor_id + 1)
        self.game_state_save_path = os.path.join(self.save_path, "game_state")
        if actor_id == 0:
            os.makedirs(self.game_state_save_path, exist_ok=True)
        self.args: Args = self.args
        args = self.args
        self.oracle = None

        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            n=1,
            logprobs=True,
        )

        self.eval_sampling_params = vllm.SamplingParams(
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
            n=1,
            logprobs=True,
        )

        self.step_count = 0

        # Get environment wrappers.
        wrappers = get_wrapper_fns(self.args.wrappers, tokenizer=self.tokenizer)

        # Instantiate vectorized environment.
        self.env = gem.make_vec(
            [self.args.env_id] * self.args.num_env,
            vec_kwargs=[{"seed": self.args.seed + j} for j in range(self.args.num_env)],
            wrappers=wrappers,
            async_mode=self.args.async_env,
        )
        # inistantiate eval vectorized environments
        self.eval_envs = {}
        for i, eval_env_id in enumerate(self.args.eval_envs):
            wrappers = get_wrapper_fns(
                self.args.eval_wrappers[i], tokenizer=self.tokenizer
            )
            vec_env = gem.make_vec(
                [eval_env_id] * self.args.eval_batch_size,
                wrappers=wrappers,
                async_mode=self.args.eval_async_env,
            )
            self.eval_envs[eval_env_id] = vec_env

    def collect_experience(self):
        logging.info(
            f"Actor-{self.actor_id} starting to collect experiences at step {self.step_count}"
        )
        env, min_steps = self.env, self.args.rollout_batch_size_per_device
        obs, _ = env.reset()
        done = False
        episodes = [[] for _ in range(env.num_envs)]
        finished_episodes = []
        finished_episodes_tool_uses = []
        finished_episodes_tool_success = []
        num_generation_failed = 0
        while True:
            action, extra = self.agent_act(obs, self.args.prompt_template)  # type: ignore
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(env.num_envs):
                if extra[i]["generation_failed"]:
                    num_generation_failed += 1
                    if self.args.keep_generation_failed:
                        episodes[i][-1].rewards += reward[i]
                        episodes[i][-1].done = True
                        finished_episodes.append(deepcopy(episodes[i]))
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                            if done[i]
                            else info[i].get("tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                            if done[i]
                            else info[i].get("tool_success_counter", 0)
                        )
                    episodes[i].clear()
                    if not done[i]:
                        next_obs[i] = env.envs[i].reset()[0]
                else:
                    transition = Transition(
                        obs=obs[i],
                        action=action[i],
                        rewards=reward[i],
                        done=done[i],
                        prompt=extra[i]["formatted_observation"],
                        prompt_ids=extra[i]["prompt_ids"],
                        response=extra[i]["response"],
                        response_ids=extra[i]["response_ids"],
                        response_logprobs=extra[i]["response_logprobs"],
                        response_is_truncated=extra[i]["response_is_truncated"],
                        action_is_formatted=extra[i]["action_is_formatted"],
                        info={},
                    )
                    episodes[i].append(transition)
                    if done[i]:
                        finished_episodes.append(deepcopy(episodes[i]))
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                        )
                        episodes[i].clear()

            obs = next_obs
            if len(tree.flatten(finished_episodes)) >= min_steps:
                break

        info = {
            "actor/num_generation_failed": num_generation_failed,
            "actor/prop_generation_failed": (
                num_generation_failed / len(finished_episodes)
                if self.args.keep_generation_failed
                else num_generation_failed
                / (len(finished_episodes) + num_generation_failed)
            ),
            "actor/num_tool_uses": np.mean(finished_episodes_tool_uses),
            "actor/num_tool_success": np.mean(finished_episodes_tool_success),
        }
        if self.step_count % self.args.dump_experience_every == 0:
            _to_dump = {}
            for i, ep in enumerate(finished_episodes):
                key = f"episode{i}"
                _to_dump[key] = []
                for transition in ep:
                    _to_dump[key].append(transition.format())
            with open(
                os.path.join(
                    self.game_state_save_path,
                    f"actor{self.actor_id}_step{self.step_count}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    _to_dump,
                    f,
                    indent=4,
                )
        self.step_count += 1
        return finished_episodes, info

    def agent_act(
        self,
        vec_observation: List[str],
        prompt_template: str,
    ) -> Tuple[str, dict]:
        """Use the current LLM as a policy to act.

        Args:
            vec_observation: Vectorized observation from TextArena environment.

        Returns:
            Tuple[str, dict]: Action and extra data.

        """
        formatted_observations = []
        for observation in vec_observation:
            observation = TEMPLATE_FACTORY[prompt_template](observation)
            if self.args.apply_chat_template:
                observation = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": observation}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_observations.append(observation)

        sampling_params = (
            self.eval_sampling_params if self.eval_mode else self.sampling_params
        )

        # Subsample to remove observations that exceed max model length
        idss = self.tokenizer(formatted_observations).input_ids
        exceeds_lengths = [len(ids) >= self.args.max_model_len for ids in idss]
        sub_formatted_observations = [
            o for o, e in zip(formatted_observations, exceeds_lengths) if not e
        ]

        # Generate
        sub_outputs = self.generate(sub_formatted_observations, sampling_params)

        executable_actions = []
        extras = []
        sub_i = 0
        for i, exceeds_length in enumerate(exceeds_lengths):
            if exceeds_length:
                # if prompt exceeds max model length we skipped the generation
                executable_actions.append(INVALID_ACTION)
                extras.append({"generation_failed": True})
            else:
                raw_action = sub_outputs[sub_i].outputs[0].text
                prompt_token_ids = sub_outputs[sub_i].prompt_token_ids
                token_ids = sub_outputs[sub_i].outputs[0].token_ids
                response_logprobs = sub_outputs[sub_i].outputs[0].logprobs
                response_logprobs = [
                    item[token_ids[i]].logprob
                    for i, item in enumerate(response_logprobs)
                ]
                response_is_truncated = (
                    sub_outputs[sub_i].outputs[0].finish_reason == "length"
                )

                # Valid extraction = proper eos + proper format
                # Only used for metric logging
                extracted_action = (
                    INVALID_ACTION
                    if response_is_truncated
                    else self.extract_action(raw_action)
                )
                executable_actions.append(
                    INVALID_ACTION if response_is_truncated else raw_action
                )
                extras.append(
                    {
                        "formatted_observation": formatted_observations[i],
                        "prompt_ids": prompt_token_ids,
                        "response": raw_action,
                        "response_ids": token_ids,
                        "response_logprobs": response_logprobs,
                        "response_is_truncated": response_is_truncated,
                        "action_is_formatted": extracted_action != INVALID_ACTION,
                        "generation_failed": False,
                        "generation_max_length_reached": (
                            len(prompt_token_ids) + len(token_ids)
                            >= self.args.max_model_len
                        ),
                    }
                )
                sub_i += 1
        return executable_actions, extras  # type: ignore

    def extract_action(self, text: str) -> str:
        """
        Extract and format the actual action from the model's output.

        This method handles different template formats and ensures the action
        is properly formatted for the environment.

        Args:
            text: Raw text output from the model

        Returns:
            Cleaned and formatted action string ready for the environment
        """
        if not text:
            return ""  # Handle empty text case

        try:
            formatted_action = None
            if self.args.prompt_template in ["qwen3_game", "qwen3_general"] or (
                self.args.prompt_template == "no"
                and "qwen" in self.args.pretrain.lower()
            ):
                formatted_action = extract_last_boxed_answer(text)
                if formatted_action is None:
                    formatted_action = text.strip()
            elif self.args.prompt_template == "code":
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
                if not code_blocks:
                    formatted_action = None
                else:
                    formatted_action = code_blocks[-1].strip()
            else:
                raise NotImplementedError

            if formatted_action is None:
                formatted_action = INVALID_ACTION

            return formatted_action

        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            # Return invalid action if extraction fails.
            return INVALID_ACTION

    def run_eval_episode(
        self, eval_env_id, eval_prompt_template, batch_size
    ) -> List[Transition]:
        def get_attr_from_wrapper(env, attr):
            if hasattr(env, attr):
                return getattr(env, attr)

            if hasattr(env, "env"):
                return get_attr_from_wrapper(env.env, attr)

            raise ValueError(f"Cannot find {attr} in env.")

        vec_env = self.eval_envs[eval_env_id]
        try:
            dataset = get_attr_from_wrapper(vec_env.envs[0], "dataset")
        except ValueError:
            dataset = DummyPromptDataset(size=1)

        finished_episodes = []
        if batch_size > len(dataset):
            logging.info(
                f"eval batch size {batch_size} is larger than dataset size {len(dataset)}, set batch size to {len(dataset)}"
            )
            batch_size = len(dataset)

        for i in range(0, len(dataset), batch_size):
            n_parallel = min(batch_size, len(dataset) - i)
            _kwargs_map = {
                f"env{j}_kwargs": {"idx": i + j}
                for j in range(min(n_parallel, len(dataset) - i))
            }
            obs, info = vec_env.reset(**_kwargs_map)
            obs, info = obs[:n_parallel], info[:n_parallel]
            episodes = [[] for _ in range(len(obs))]
            done = np.array([False] * n_parallel)
            while not all(done):
                action, extra = self.agent_act(obs, eval_prompt_template)
                # distrubte action based on done mask
                action_iter = iter(action)
                next_obs, reward, terminated, truncated, info = vec_env.step(
                    {i: next(action_iter) for i, d in enumerate(done) if not d}
                )
                obs_len = [len(self.tokenizer.encode(o)) for o in next_obs]
                obs_exceeds_max_len = np.array(
                    [l >= self.args.max_model_len for l in obs_len]
                )
                # distribute cur_done to done
                cur_done = terminated | truncated | obs_exceeds_max_len
                iter_idx = 0
                for i in range(len(done)):
                    if not done[i]:
                        episodes[i].append(
                            {
                                "obs": obs[iter_idx],
                                "action": action[iter_idx],
                                "reward": reward[iter_idx],
                                "next_obs": next_obs[iter_idx],
                                "done": cur_done[iter_idx],
                                "info": info[iter_idx],
                            }
                        )
                        iter_idx += 1
                done[~done] = cur_done
                obs = [o for o, d in zip(next_obs, cur_done) if not d]

            finished_episodes.extend(episodes)
        return finished_episodes


class DummyPromptDataset(Dataset):
    """Empty dataset to satisfy OAT's requirements without actually loading data."""

    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        del idx
        return "", "", ""


""" +====================================+ """
""" 4. Defining learner update the policy. """
""" +====================================+ """


class Learner(PPOMultiTurnLearner):
    def _init(self, args: Args, actors: List[Actor]) -> None:
        """
        Initialize the learner.
        """
        # Call parent's _init but then override prepare_data
        super()._init(args, actors)
        self.args = args

        # Masked sum is the correct implementation!
        # Oat by default uses Dr.GRPO: https://arxiv.org/pdf/2503.20783
        self.masked_aggregator = functools.partial(
            masked_sum,
            constant_normalizer=args.length_norm_constant or args.generate_max_length,
        )

    def prepare_data(self, strategy, tokenizer):
        """
        Override the data preparation to avoid loading external datasets.
        Instead, create dummy datasets just to keep OAT's infrastructure happy.
        """
        # Create dummy dataset that satisfies OAT's requirements
        # but doesn't actually load any data
        # Used to control the training episode, set a large number.
        self.prompts_dataset = DummyPromptDataset(size=int(1e9))
        self.eval_prompts_dataset = DummyPromptDataset(size=1)  # no use currently

        # Create the dataloaders
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            shuffle=False,  # No need to shuffle dummy data
        )
        self.eval_prompts_dataloader = strategy.setup_dataloader(
            self.eval_prompts_dataset, batch_size=1, shuffle=False, drop_last=False
        )

    def evaluate(self, _unused_dataloader, steps):
        """Online evaluation on TIR environments."""
        # NOTE: Evaluate all envs specified in args.eval_envs, report avg@args.eval_n
        # NOTE: prompt_template is needed for concat wrapper
        del _unused_dataloader
        assert not self.pi_beta_lags_behind, "pi beta lags behind for evaluation"
        assert (
            self.args.eval_n % len(self.actors) == 0
        ), "args.eval_n must be divisible by number of actors"
        self._pre_evaluate()
        self.strategy.print(f"Starting evaluation at {steps} steps")
        eval_env_ids = self.args.eval_envs
        eval_prompt_templates = self.args.eval_prompt_templates

        t0 = time.time()
        futs = []
        episodes = []

        metrics = {
            f"eval/{env_id}/{metric}": 0.0
            for env_id in eval_env_ids
            for metric in [
                "accuracy",
                "elapse",
                "response_tok_len",
                "mean_episode_len",
                "num_tool_success",
            ]
        }

        for eval_env_id, eval_prompt_template in zip(
            eval_env_ids, eval_prompt_templates
        ):
            episodes.clear()
            # assign task and wait for results
            n_actor = len(self.actors)
            for _ in range(self.args.eval_n // n_actor):
                if self.strategy.is_rank_0():
                    futs += [
                        actor.futures.run_eval_episode(
                            eval_env_id,
                            eval_prompt_template,
                            self.args.eval_batch_size,
                        )
                        for actor in self.actors
                    ]
                    for fut in futs:
                        episodes.extend(fut.result())
                    futs.clear()

            run_elapse = time.time() - t0
            t0 = time.time()
            metrics.update(
                {
                    f"eval/{eval_env_id}/elapse": run_elapse,
                    f"eval/{eval_env_id}/response_tok_len": np.mean(
                        [
                            sum([len(self.tokenizer.encode(t["action"])) for t in ep])
                            for ep in episodes
                        ]
                    ),
                    f"eval/{eval_env_id}/accuracy": np.mean(
                        [sum([t["reward"] for t in ep]) for ep in episodes]
                    ),
                    f"eval/{eval_env_id}/mean_episode_len": np.mean(
                        [len(ep) for ep in episodes]
                    ),
                    f"eval/{eval_env_id}/num_tool_success": np.mean(
                        [
                            ep[-1]["info"].get("tool_success_counter", 0)
                            + ep[-1]["info"].get("prev_ep_tool_success_counter", 0)
                            for ep in episodes
                        ]
                    ),
                }
            )
            # save the results
            transitions = [t for ep in episodes for t in ep]
            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            pd.DataFrame(
                {
                    "obs": [t["obs"] for t in transitions],
                    "action": [t["action"] for t in transitions],
                    "reward": [t["reward"] for t in transitions],
                    "done": [t["done"] for t in transitions],
                    "next_obs": [t["next_obs"] for t in transitions],
                    "info": [t["info"] for t in transitions],
                }
            ).to_json(
                os.path.join(eval_res_path, f"{steps}_{eval_env_id}.json"),
                orient="records",
                indent=4,
            )

        dist.barrier()
        metrics = self.strategy.broadcast(metrics)
        metrics["eval/average/accuracy"] = np.mean(
            [metrics[f"eval/{env_id}/accuracy"] for env_id in eval_env_ids]
        )
        metrics["eval/average/mean_episode_len"] = np.mean(
            [metrics[f"eval/{env_id}/mean_episode_len"] for env_id in eval_env_ids]
        )
        metrics["eval/average/response_tok_len"] = np.mean(
            [metrics[f"eval/{env_id}/response_tok_len"] for env_id in eval_env_ids]
        )
        metrics["eval/average/elapse"] = np.mean(
            [metrics[f"eval/{env_id}/elapse"] for env_id in eval_env_ids]
        )
        self._post_evaluate()
        return metrics


def train(args: Args):
    """
    Reinforcement learning starts here.

    Args:
        args: Configuration arguments for the run
    """
    # Define a distributed program that composes Actors and Learners
    program, local_resources = get_program(args, learner_cls=Learner, actor_cls=Actor)

    # Launch the program
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    # Get default arguments and customize them
    args: Args = get_default_args(Args)

    # Customization
    args.algo = "PPO"
    args.eval_batch_size = 32

    # CRITICAL: Disable oracle and dataset loading
    args.oracle = ""  # Empty string for no external oracle
    args.prompt_data = ""  # Don't load any dataset
    args.rollout_batch_size = args.rollout_batch_size_per_device * args.gpus

    # setup evaluation hps
    def _validate_eval_hp(hp):
        hp = hp.split("|")
        if len(hp) == 1:
            hp = hp * len(args.eval_envs)
        else:
            assert len(hp) == len(
                args.eval_envs
            ), "eval_wrappers/eval_prompt_templates should be either a string or a list of the same length as eval_envs"
        return hp

    if args.eval_envs:
        args.eval_envs = args.eval_envs.split("|")
        assert isinstance(args.eval_envs, list)
        assert len(args.eval_envs) == len(
            set(args.eval_envs)
        ), "eval_envs should be unique"
        args.eval_wrappers = _validate_eval_hp(args.eval_wrappers)
        args.eval_prompt_templates = _validate_eval_hp(args.eval_prompt_templates)
    else:
        logging.info(
            "No eval_envs specified, set `args.eval_steps` to -1,skipping evaluation."
        )
        args.eval_envs = []
        args.eval_steps = -1

    if "concat_chat" in args.wrappers:
        assert (
            args.prompt_template == "no"
        ), "chat template is applied on env side already"
    args = default_args_validation(args)

    # Let's go
    train(args)
