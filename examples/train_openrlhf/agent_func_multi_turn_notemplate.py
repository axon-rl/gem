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

import logging
from typing import Any, Dict

import torch
from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase

import gem

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


# TEMPLATE_FACTORY = {
#     "qwen3_game": apply_qwen3_game_template,
#     "no": apply_no_template,
#     "qwen3_general": apply_qwen3_general_template,
#     "code": apply_code_template,
# }

import time


# A simple n-step random environment
class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.env = gem.make("game:GuessTheNumber-v0", seed=time.time_ns())
        self.max_steps = self.env.max_turns

    async def reset(self, states: dict, **kwargs):
        """Initialize the environment and return initial observation

        Args:
            states: Dictionary containing prompt and label

        Returns:
            str: Initial observation text
        """
        import random

        seed = kwargs.get("seed", time.time_ns())
        random.seed(seed)
        # Reset the environment to generate the first observation
        observation, info = self.env.reset(random.randint(1, 10000000))
        # Reset the environment to generate the first observation
        self.step_idx = 0
        return {
            "observation": states["observation"]
        }  # Return original text observation

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Execute one step of verification and return environment feedback

        Args:
            states: Dictionary containing observation_text, action_text, and label

        Returns:
            Dict[str, Any]: A dictionary containing:
                - rewards: Reward value for advantage calculation
                - scores: Reward value for dynamic filtering
                - environment_feedback: The environment feedback text
                - done: Boolean indicating if the episode is complete
                - sampling_params: Parameters for vLLM sampling
                - extra_logs: Additional logging information
        """
        print(f"step_idx: {self.step_idx}, max_steps: {self.max_steps}")

        observation_text = states["observation_text"]
        action_text = states["action_text"]
        label = states["label"]

        # apply action and receive next observation, reward
        # and whether the episode has ended
        next_observation, reward, terminated, truncated, info = self.env.step(
            action_text
        )

        if reward < -1:
            reward = -1.0

        logger.info(
            {
                "INFO": "##INFO##",
                "action": action_text,
                "observation": next_observation,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "step_idx": self.step_idx,
            }
        )

        # Check if episode is done
        done = terminated or truncated
        self.step_idx += 1

        next_observation = f"\nUser: {next_observation}\n"

        if not done:
            next_observation += (
                "\nEnter your next guess within \\boxed{{}}.\nAssistant: "
            )
            # next_observation += f"\n{info['suffix']}\n"

        return {
            "rewards": torch.tensor(reward),  # Rewards for advantage calculation
            "scores": torch.tensor(reward),  # Scores for dynamic filtering (0-1 reward)
            "environment_feedback": next_observation,  # Environment feedback text
            "done": done,  # Boolean indicating if the episode is complete
            "sampling_params": states.get(
                "sampling_params", None
            ),  # Parameters for vLLM sampling in next step
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(self.step_idx),
            },  # Additional logging information
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        super().__init__(
            AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue
        )

    async def execute(self, prompt, label, sampling_params):
        # You could override the execute function of AgentExecutorBase to add custom agent running logic
        return await super().execute(prompt, label, sampling_params)
