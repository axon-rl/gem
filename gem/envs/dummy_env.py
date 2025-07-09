"""Dummy env for testing python tool."""

import logging
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE

logger = logging.getLogger(__name__)


class DummyEnv(Env):
    """Built upon a dataset, serving as a single-turn env (contextual bandits)."""

    def __init__(
        self,
        seed: int = 0,
        **_,
    ):
        super().__init__()
        self.seed = seed

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        return TERMINAL_STATE, 0.0, True, True, {}

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        del seed
        first_obs = "Write a short example python code block to print something."
        return first_obs, {}

    def sample_random_action(self) -> str:
        """Sample a random action."""
        return "```python\nprint('Hello, World!')\n```"
