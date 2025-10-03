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
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import tree
import vllm
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TransitionData
from oat.utils.ops import masked_mean, masked_sum
from torch.utils.data import Dataset

import gem
from gem.utils.parsing import extract_last_boxed_answer
from gem.wrappers.wrapper_factory import get_wrapper_fns

""" +=========================================+ """
""" 1. Defining constants used in our training. """
""" +=========================================+ """

from examples.train_oat import (
    INVALID_ACTION,
    TEMPLATE_FACTORY,
)

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

    # Template settings
    prompt_template: Literal["qwen3_game", "no", "qwen3_general", "code"] = "qwen3_game"

    # Reward settings
    gamma: float = 1.0  # Discount factor for Monte Carlo returns
    norm_return: bool = False

    # Evaluation settings
    eval_prompt_template: Literal["qwen3_general"] = "qwen3_general"
    eval_data: Optional[str] = "./data"
    eval_input_key: str = "input"
    eval_output_key: str = "answer"
    eval_split: str = "all"

    # Misc settings
    dump_experience_every: int = 1  # Dump experience data

    # Episode collection logic
    keep_generation_failed: bool = False  # Keep episodes with generation failures

    critic_type2: Literal["grpo", "drgrpo", "rloo", "ep_level", "none"] = "none"


@dataclass
class Transition:
    obs: str
    action: str
    reward: float
    done: bool

    prompt: str
    prompt_ids: list
    response: str
    response_ids: list
    response_logprobs: list

    response_is_truncated: bool
    action_is_formatted: bool

    def format(self):
        return {
            "obs": self.obs,
            "action": self.action,
            "reward": self.reward,
            "done": int(self.done),
            "prompt": self.prompt,
            "response": self.response,
        }


""" +=======================================+ """
""" 3. Defining actor to collect experiences. """
""" +=======================================+ """


class Actor(PPOActor):
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

    def step(
        self, prompts=None, formatted_prompts=None, references=None
    ) -> List[TransitionData]:
        """Each actor.step handles the interaction between agent and environment to collect experiences."""
        # The provided parameters are ignored since we generate prompts from the environment
        del prompts, formatted_prompts, references

        logging.info(
            f"Actor-{self.actor_id} starting to collect experiences at step {self.step_count}"
        )
        info = {}

        # Play multiple episodes to generate transitions
        all_transitions = []

        finished_groups, collection_info = self.collect_experience(
            self.env, self.args.rollout_batch_size_per_device
        )
        for group in finished_groups:
            all_transitions.extend(self.prepare_group_of_episodes(group))

        # logging infos
        flattened_episodes = []
        for group in finished_groups:
            flattened_episodes.extend(group)
        info["actor/num_transitions"] = len(all_transitions)
        info["actor/mean_episode_len"] = np.mean([len(ep) for ep in flattened_episodes])
        info["actor/mean_episode_return"] = np.mean(
            [
                sum(transition.reward for transition in episode)
                for episode in flattened_episodes
            ]
        )
        info["actor/mean_episode_success"] = np.mean(
            [episode[-1].reward == 1 for episode in flattened_episodes]
        )  # NOTE: assuming success reward is always 1

        # update collection info
        info.update(
            {k.replace("actor/", "actor/"): v for k, v in collection_info.items()}
        )

        # Subsample transitions if they exceed the batch size
        if len(all_transitions) > self.args.rollout_batch_size_per_device:
            subsample_indices = np.random.choice(
                len(all_transitions),
                self.args.rollout_batch_size_per_device,
                replace=False,
            )
            all_transitions = [all_transitions[si] for si in subsample_indices]
        logging.info(f"Actor finished collecting {len(all_transitions)} transitions")

        for transition in all_transitions:
            transition.info.update(**info)

        self.step_count += 1
        # Serialize and return the transitions
        handle = self.ipc_client.serialize_ipc(all_transitions)
        return handle  # type: ignore

    def collect_experience(self, env, min_steps: int):
        if self.args.num_samples == 1:
            episodes, info = self.collect_experience_single(env, min_steps)
            episode_groups = [[ep] for ep in episodes]
        else:
            episode_groups, info = self.collect_experience_multiple(
                env, min_steps, self.args.num_samples
            )
        return episode_groups, info

    def collect_experience_single(self, env, min_steps: int):
        obs, _ = env.reset()
        done = False
        episodes = [[] for _ in range(env.num_envs)]
        finished_episodes = []
        finished_episodes_tool_uses = []
        finished_episodes_tool_success = []
        num_generation_failed = 0
        while True:
            action, extra = self.agent_act(obs)  # type: ignore
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(env.num_envs):
                if extra[i]["generation_failed"]:
                    num_generation_failed += 1
                    if self.args.keep_generation_failed:
                        episodes[i][-1].reward += reward[i]
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
                        reward=reward[i],
                        done=done[i],
                        prompt=extra[i]["formatted_observation"],
                        prompt_ids=extra[i]["prompt_ids"],
                        response=extra[i]["response"],
                        response_ids=extra[i]["response_ids"],
                        response_logprobs=extra[i]["response_logprobs"],
                        response_is_truncated=extra[i]["response_is_truncated"],
                        action_is_formatted=extra[i]["action_is_formatted"],
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

        return finished_episodes, info

    def collect_experience_multiple(self, env, min_steps: int, num_samples: int):
        start_time = time.time()
        # If env has get_state and set_state methods then use these, otherwise deepcopy the env
        env_has_getset_state = hasattr(env.envs[0], "get_state") and hasattr(
            env.envs[0], "set_state"
        )
        logging.info(f"Actor-{self.actor_id}: {env_has_getset_state=}")

        # for in-progress episodes
        episodes = [[] for _ in range(env.num_envs)]
        envs_in_progress = [None for _ in range(env.num_envs)]
        ids_in_progress = [None for _ in range(env.num_envs)]
        initial_obs_in_progress = [None for _ in range(env.num_envs)]
        finished_episodes_groups = {}

        # queues
        env_queue = deque()
        id_queue = deque()
        initial_obs_queue = deque()

        # for finished groups
        finished_groups = []
        finished_groups_ids = []
        finished_groups_envs = []
        finished_groups_num_transitions = 0

        # for logging
        finished_episodes_tool_uses = []
        finished_episodes_tool_success = []
        num_finished_episodes = 0
        num_generation_failed = 0
        max_ep_length = 0

        def get_env_for_storing(env_i, apply_deepcopy=True):
            if env_has_getset_state:
                state = env_i.get_state()
            else:
                state = env_i
            return deepcopy(state) if apply_deepcopy else state

        def set_env(envs, i, state, apply_deepcopy=True):
            state_ = deepcopy(state) if apply_deepcopy else state
            if env_has_getset_state:
                envs[i].set_state(state_)
            else:
                envs[i] = state_

        def top_up_queue(env_to_use):
            if len(env_queue) == 0:
                while True:
                    id = str(uuid.uuid4())
                    if (
                        (id not in id_queue)
                        and (id not in ids_in_progress)
                        and (id not in finished_groups_ids)
                    ):
                        break
                while True:
                    initial_obs, _ = env_to_use.reset()
                    if (
                        (env_to_use not in env_queue)
                        and (env_to_use not in envs_in_progress)
                        and (env_to_use not in finished_groups_envs)
                    ):
                        break

                finished_episodes_groups[id] = []
                for j in range(num_samples):
                    # env_queue.append(deepcopy(env_to_use))
                    env_queue.append(
                        get_env_for_storing(env_to_use, apply_deepcopy=True)
                    )
                    id_queue.append(id)
                    initial_obs_queue.append(initial_obs)

        def move_finished_group(id):
            assert (
                len(finished_episodes_groups[id]) <= num_samples
            ), f"{num_samples=}\n{len(finished_episodes_groups[id])=}\n{id=}\n{finished_episodes_groups=}"
            if len(finished_episodes_groups[id]) == num_samples:
                finished_group = finished_episodes_groups.pop(id)
                finished_groups.append(finished_group)
                finished_groups_ids.append(deepcopy(id))
                num_transitions_added = sum([len(ep) for ep in finished_group])
            else:
                num_transitions_added = 0
            return num_transitions_added

        def update_metrics(info_i, done_i):
            finished_episodes_tool_uses.append(
                info_i.get("prev_ep_tool_use_counter", 0)
                if done_i
                else info_i.get("tool_use_counter", 0)
            )
            finished_episodes_tool_success.append(
                info_i.get("prev_ep_tool_success_counter", 0)
                if done_i
                else info_i.get("tool_success_counter", 0)
            )

        # Set initial state of each environment
        obs, _ = env.reset()
        for i in range(env.num_envs):
            top_up_queue(env.envs[i])
            envs_in_progress[i] = env_queue.popleft()
            ids_in_progress[i] = id_queue.popleft()
            initial_obs_in_progress[i] = initial_obs_queue.popleft()
            # env.envs[i] = deepcopy(envs_in_progress[i])
            set_env(env.envs, i, envs_in_progress[i], apply_deepcopy=True)
            obs[i] = deepcopy(initial_obs_in_progress[i])

        # Collect episodes
        while True:
            action, extra = self.agent_act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(env.num_envs):

                if extra[i]["generation_failed"]:
                    num_generation_failed += 1
                    if self.args.keep_generation_failed:
                        episodes[i][-1].reward += reward[i]
                        episodes[i][-1].done = True
                        finished_episodes_groups[ids_in_progress[i]].append(
                            deepcopy(episodes[i])
                        )
                        num_finished_episodes += 1
                        finished_groups_num_transitions += move_finished_group(
                            ids_in_progress[i]
                        )
                        max_ep_length = max(max_ep_length, len(episodes[i]))
                        update_metrics(info[i], done[i])
                        top_up_queue(env.envs[i])
                        envs_in_progress[i] = env_queue.popleft()
                        ids_in_progress[i] = id_queue.popleft()
                        initial_obs_in_progress[i] = initial_obs_queue.popleft()
                    episodes[i].clear()
                    # env.envs[i] = deepcopy(envs_in_progress[i])
                    set_env(env.envs, i, envs_in_progress[i], apply_deepcopy=True)
                    next_obs[i] = deepcopy(initial_obs_in_progress[i])

                else:
                    transition = Transition(
                        obs=obs[i],
                        action=action[i],
                        reward=reward[i],
                        done=done[i],
                        prompt=extra[i]["formatted_observation"],
                        prompt_ids=extra[i]["prompt_ids"],
                        response=extra[i]["response"],
                        response_ids=extra[i]["response_ids"],
                        response_logprobs=extra[i]["response_logprobs"],
                        response_is_truncated=extra[i]["response_is_truncated"],
                        action_is_formatted=extra[i]["action_is_formatted"],
                    )
                    episodes[i].append(transition)
                    if done[i]:
                        finished_episodes_groups[ids_in_progress[i]].append(
                            deepcopy(episodes[i])
                        )
                        num_finished_episodes += 1
                        finished_groups_num_transitions += move_finished_group(
                            ids_in_progress[i]
                        )
                        max_ep_length = max(max_ep_length, len(episodes[i]))
                        update_metrics(info[i], done[i])
                        top_up_queue(env.envs[i])
                        envs_in_progress[i] = env_queue.popleft()
                        ids_in_progress[i] = id_queue.popleft()
                        initial_obs_in_progress[i] = initial_obs_queue.popleft()
                        episodes[i].clear()
                        # env.envs[i] = deepcopy(envs_in_progress[i])
                        set_env(env.envs, i, envs_in_progress[i], apply_deepcopy=True)
                        next_obs[i] = deepcopy(initial_obs_in_progress[i])

                if finished_groups_num_transitions >= min_steps:
                    break
            if finished_groups_num_transitions >= min_steps:
                break

            #     print(f"{x=}, {i=}, {finished_groups_num_transitions=}")
            # x += 1
            # assert x <= 30, f"{x=}, {finished_groups_num_transitions=}, {min_steps=}"

            obs = next_obs

        info = {
            "actor/num_generation_failed": num_generation_failed,
            "actor/prop_generation_failed": (
                num_generation_failed / num_finished_episodes
                if self.args.keep_generation_failed
                else num_generation_failed
                / (num_finished_episodes + num_generation_failed)
            ),
            "actor/num_tool_uses": np.mean(finished_episodes_tool_uses),
            "actor/num_tool_success": np.mean(finished_episodes_tool_success),
            "actor/num_groups": len(finished_groups),
        }
        if self.step_count % self.args.dump_experience_every == 0:
            _to_dump = {}
            for n, group in enumerate(finished_groups):
                for i, ep in enumerate(group):
                    key = f"group{n}_episode{i}"
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

        return finished_groups, info

    def agent_act(self, vec_observation: List[str]) -> Tuple[str, dict]:
        """Use the current LLM as a policy to act.

        Args:
            vec_observation: Vectorized observation from TextArena environment.

        Returns:
            Tuple[str, dict]: Action and extra data.

        """
        formatted_observations = []
        for observation in vec_observation:
            observation = TEMPLATE_FACTORY[self.args.prompt_template](observation)
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

    def prepare_group_of_episodes(
        self, group: Sequence[Transition]
    ) -> List[TransitionData]:
        """
        Prepare language transitions (sequence of tokens to sequence of tokens).

        Args:
            group: List of episodes in the group

        Returns:
            List of transitions
        """
        if self.args.critic_type2 in ["grpo", "drgrpo", "rloo", "ep_level"]:
            return self.prepare_group_of_episodes_episode_level(group)
        elif self.args.critic_type2 in ["none"]:
            return self.prepare_group_of_episodes_transition_level(group)
        else:
            raise NotImplementedError(f"Unknown arg: {self.args.critic_type2=}")

    def prepare_group_of_episodes_episode_level(
        self, group: Sequence[Transition]
    ) -> List[TransitionData]:
        if self.args.critic_type2 in ["grpo", "drgrpo", "rloo"]:
            assert (
                self.args.num_samples > 1
            ), f"{self.args.critic_type2=} requires num_samples > 1, got {self.args.num_samples=}"
        group_rewards_ep_level = [sum(t.reward for t in episode) for episode in group]
        # Normalize at episode level
        if self.args.critic_type2 == "grpo":
            mean = np.mean(group_rewards_ep_level)
            std = np.std(group_rewards_ep_level) + 1e-9
            group_returns_ep_level_normalized = [
                (r - mean) / std for r in group_rewards_ep_level
            ]
        elif self.args.critic_type2 == "drgrpo":
            mean = np.mean(group_rewards_ep_level)
            group_returns_ep_level_normalized = [
                r - mean for r in group_rewards_ep_level
            ]
        elif self.args.critic_type2 == "rloo":
            group_returns_ep_level_normalized = []
            for i, r in enumerate(group_rewards_ep_level):
                leave_one_out = [
                    group_rewards_ep_level[j]
                    for j in range(len(group_rewards_ep_level))
                    if j != i
                ]
                group_returns_ep_level_normalized.append(r - np.mean(leave_one_out))
        elif self.args.critic_type2 == "ep_level":
            group_returns_ep_level_normalized = group_rewards_ep_level
        else:
            raise NotImplementedError(f"Unknown arg: {self.args.critic_type2=}")

        # Collect tranisitions
        transitions = []
        for episode, ep_reward in zip(group, group_returns_ep_level_normalized):
            # Distribute turn-based returns to token-level returns
            for i, step_data in enumerate(episode):
                dense_rewards = self.compute_token_level_rewards(
                    step_data.response_ids, ep_reward
                )
                # Add transition data
                transitions.append(
                    TransitionData(
                        prompt=step_data.prompt,
                        prompt_ids=step_data.prompt_ids,
                        response=step_data.response,
                        response_ids=step_data.response_ids,
                        # response_logprobs=None,  # Re-calculated on learner side.
                        response_logprobs=step_data.response_logprobs,
                        rewards=dense_rewards,
                        loss_mask=(
                            not step_data.response_is_truncated
                            if self.args.ignore_no_eos
                            else True
                        ),
                        info={
                            "actor/action_is_formatted": step_data.action_is_formatted,
                            "actor/step_reward": step_data.reward,
                            "actor/discount_factor": self.args.gamma,
                            "actor/discounted_step_return": ep_reward,
                            "actor/response_is_truncated": step_data.response_is_truncated,
                            "actor/timestamp": time.time_ns(),
                        },
                    )
                )
        return transitions

    def prepare_group_of_episodes_transition_level(
        self, group: Sequence[Transition]
    ) -> List[TransitionData]:

        # Compute the returns
        group_returns = (
            []
        )  # List (episodes) of arrays (return per transition in episode)
        for episode in group:
            rewards = [t.reward for t in episode]
            returns = np.zeros_like(rewards, dtype=np.float32)
            cur = 0.0
            for i in reversed(range(len(rewards))):
                cur = rewards[i] + self.args.gamma * cur
                returns[i] = cur
            group_returns.append(returns)

        # Collect tranisitions
        transitions = []
        for episode, returns in zip(group, group_returns):
            # Distribute turn-based returns to token-level returns
            for i, step_data in enumerate(episode):
                dense_rewards = self.compute_token_level_rewards(
                    step_data.response_ids, returns[i]
                )
                # Add transition data
                transitions.append(
                    TransitionData(
                        prompt=step_data.prompt,
                        prompt_ids=step_data.prompt_ids,
                        response=step_data.response,
                        response_ids=step_data.response_ids,
                        # response_logprobs=None,  # Re-calculated on learner side.
                        response_logprobs=step_data.response_logprobs,
                        rewards=dense_rewards,
                        loss_mask=(
                            not step_data.response_is_truncated
                            if self.args.ignore_no_eos
                            else True
                        ),
                        info={
                            "actor/action_is_formatted": step_data.action_is_formatted,
                            "actor/step_reward": step_data.reward,
                            "actor/discount_factor": self.args.gamma,
                            "actor/discounted_step_return": returns[i],
                            "actor/response_is_truncated": step_data.response_is_truncated,
                            "actor/timestamp": time.time_ns(),
                        },
                    )
                )
        return transitions

    def compute_token_level_rewards(
        self, token_ids: List[int], discounted_reward: float
    ) -> List[float]:
        # Initialize all tokens with zero reward
        dense_rewards = [0.0] * len(token_ids)
        # Last token gets full discounted reward
        dense_rewards[-1] = discounted_reward
        return dense_rewards

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


class Learner(PPOLearner):
    def _init(self, args: Args, actors: List[Actor]) -> None:
        """
        Initialize the learner.
        """
        # Call parent's _init but then override prepare_data
        super()._init(args, actors)
        self.args = args

        # Masked sum is the correct implementation!
        # Oat by default uses Dr.GRPO: https://arxiv.org/pdf/2503.20783
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
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

        # Create the dataloaders
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            shuffle=False,  # No need to shuffle dummy data
        )

    def process_feedback_data(self, data_list: List[TransitionData]):
        """Process collected feedback data, adding it to buffer."""

        logging.info("adding data into buffer")

        # Add to buffer
        self.pi_buffer.extend(data_list)

        # Also add to all_buffer if we're tracking all data
        if self.args.dump_all_buffer:
            self.all_buffer.extend(data_list)

        # Update query step (for tracking progress)
        self.query_step += len(data_list)

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        # Return without baseline
        advantages = rewards.sum(-1)
        if self.args.norm_return:
            local_sum = advantages.sum()
            local_square_sum = (advantages**2).sum()
            local_num = torch.tensor(
                [advantages.numel()], dtype=torch.float32, device=advantages.device
            )

            global_sum = self.strategy.all_reduce(local_sum, op="sum")
            global_square_sum = self.strategy.all_reduce(local_square_sum, op="sum")
            global_num = self.strategy.all_reduce(local_num, op="sum")

            mean_adv = global_sum / global_num
            std_adv = torch.sqrt(global_square_sum / global_num - mean_adv**2)
            advantages = (advantages - mean_adv) / (std_adv + 1e-9)
        return advantages


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
    if "concat_chat" in args.wrappers:
        assert (
            args.prompt_template == "no"
        ), "chat template is applied on env side already"
    args = default_args_validation(args)

    # Let's go
    train(args)
