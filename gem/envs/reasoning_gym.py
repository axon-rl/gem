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

"""Reasoning Gym environments (https://github.com/open-thought/reasoning-gym)."""

import random
from typing import Any, Optional, SupportsFloat, Tuple

import reasoning_gym as rg

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer


class ReasoningGymEnv(Env):
    """Built upon a dataset, serving as a single-turn env (contextual bandits)."""

    def __init__(self, name: str, size: int = 500, seed: int = 42) -> None:
        super().__init__()
        self.idx = 0
        self.name = name
        self.size = size
        self.seed = seed
        self.dataset = rg.create_dataset(name, size=size, seed=seed)
        self.dataset_iter = iter(self.dataset)
        self.reward_fn = self.dataset.score_answer

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        clean_action = extract_last_boxed_answer(action)
        reward = self.reward_fn(answer=clean_action, entry=self.data)
        return TERMINAL_STATE, reward, True, True, {}

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        if seed is not None:
            data = random.choice(self.dataset)
            if (self.idx + 1) % self.size == 0:
                self.dataset = rg.create_dataset(
                    self.name, size=self.size, seed=self.seed + self.idx
                )
        else:
            try:
                data = next(self.dataset_iter)
            except StopIteration:
                # reset dataset with a new but deterministic seed
                self.dataset = rg.create_dataset(
                    self.name, size=self.size, seed=self.seed + self.idx
                )
                self.dataset_iter = iter(self.dataset)
                data = next(self.dataset_iter)

        question = data["question"]
        self.idx += 1
        self.data = data
        return question, {}
