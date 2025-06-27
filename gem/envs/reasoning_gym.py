"""Reasoning Gym environments (https://github.com/open-thought/reasoning-gym)."""

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
        del seed

        try:
            data = next(self.dataset_iter)
        except StopIteration:
            # reset dataset with a new but deterministic seed
            self.dataset = rg.create_dataset(
                self.name, size=self.size, seed=self.seed + self.idx
            )
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)

        self.idx += 1
        self.data = data
        return self.data["question"], {}

    def get_initial_state(self) -> dict[str, Any]:
        return self.data

    def reset_to_initial_state(self, initial_state: dict[str, Any]) -> Tuple[str, dict[str, Any]]:
        assert isinstance(initial_state, dict), f"Incorrect initial state format: {type(initial_state)=}, {initial_state=}"
        self.data = initial_state
        return self.data["question"], {}
