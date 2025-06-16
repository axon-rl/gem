from enum import Enum
from typing import Callable, Sequence, TypeVar, Union

import numpy as np

from gem.core import Env

ArrayType = TypeVar("ArrayType")


class AutoresetMode(Enum):
    """Enum representing the different autoreset modes, next step and same step."""

    NEXT_STEP = "NextStep"
    SAME_STEP = "SameStep"


class VectorEnv(Env):
    """Defaults to NEXT_STEP AutoresetMode, see https://farama.org/Vector-Autoreset-Mode."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        autoreset_mode: Union[str, AutoresetMode] = AutoresetMode.SAME_STEP,
    ) -> None:
        super().__init__()
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(env_fns)
        self.autoreset_mode = autoreset_mode

        # Initialize attributes used in `step` and `reset`
        self._env_obs = [None for _ in range(self.num_envs)]
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)
        self._env_infos = [{} for _ in range(self.num_envs)]
