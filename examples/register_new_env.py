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

import random
import string

import gem
from gem.core import Env
from gem.envs.registration import register
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer


def demo():
    """+=========================================+"""
    """ Use case 1: add new math/code/qa datasets.  """
    """ +=========================================+ """
    print("=" * 10, "case 1", "=" * 10)
    # Use math as an example
    register(
        "math:GSM8K-Example",
        "gem.envs.math_env:MathEnv",
        dataset_name="axon-rl/GSM-8k",
        question_key="problem",
        answer_key="answer",
    )

    env = gem.make("math:GSM8K-Example")
    obs, _ = env.reset()
    print(obs)  # i.e., question

    action = "\\boxed{42}"
    next_obs, reward, _, _, _ = env.step(action)
    print(next_obs, reward)

    """ +=======================================+ """
    """ Use case 2: implement a new environment.  """
    """ +=======================================+ """

    print("=" * 10, "case 2", "=" * 10)
    register(
        "game:ReverseString",
        "examples.register_new_env:ReverseStringEnv",
    )

    env = gem.make("game:ReverseString")
    obs, _ = env.reset()
    print(obs)  # i.e., question

    action = "\\boxed{" + env.gt_str[::-1] + "}"  # hack to get the gt
    next_obs, reward, _, _, _ = env.step(action)
    print(next_obs, reward)


class ReverseStringEnv(Env):
    def __init__(self, str_len: int = 5):
        super().__init__()
        self.str_len = str_len

    def _get_instructions(self) -> str:
        return (
            "You are tasked to reverse a given string.\n"
            "You may provide your response in any manner. Only the content wrapped inside \\boxed{} will be considered as your final answer.\n"
            f"Please reverse the string: {self.gt_str}.\n"
        )

    def reset(self, seed=None):
        super().reset(seed)
        characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
        self.gt_str = "".join(random.choices(characters, k=self.str_len))
        return self._get_instructions(), {}

    def step(self, action):
        clean_action = extract_last_boxed_answer(action)
        if clean_action is None:
            reward = 0
        else:
            reward = float(clean_action[::-1] == self.gt_str)
        return TERMINAL_STATE, reward, True, True, {}


if __name__ == "__main__":
    demo()
