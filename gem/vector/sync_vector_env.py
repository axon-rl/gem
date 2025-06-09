"""Synchronous (for loop) vectorized environment execution."""

from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from gem.core import ActType, Env, ObsType

ArrayType = TypeVar("ArrayType")


class AutoresetMode(Enum):
    """Enum representing the different autoreset modes, next step and same step."""

    NEXT_STEP = "NextStep"
    SAME_STEP = "SameStep"


class SyncVectorEnv(Env):
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
        self._observations = [None]
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    def step(self, actions: Sequence[ActType]) -> Tuple[
        Sequence[ObsType],
        ArrayType,
        ArrayType,
        ArrayType,
        dict[str, Any],
    ]:
        for i, action in enumerate(actions):
            if self.autoreset_mode == AutoresetMode.NEXT_STEP:
                if self._autoreset_envs[i]:
                    self._env_obs[i], env_info = self.envs[i].reset()
                    self._rewards[i] = 0.0
                    self._terminations[i] = False
                    self._truncations[i] = False
                else:
                    (
                        self._env_obs[i],
                        self._rewards[i],
                        self._terminations[i],
                        self._truncations[i],
                        env_info,
                    ) = self.envs[i].step(action)
            elif self.autoreset_mode == AutoresetMode.SAME_STEP:
                (
                    self._env_obs[i],
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    env_info,
                ) = self.envs[i].step(action)

                if self._terminations[i] or self._truncations[i]:
                    self._env_obs[i], env_info = self.envs[i].reset()
            else:
                raise ValueError

            del env_info

        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._env_obs),
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            {},
        )

    def reset(
        self, seed: Optional[Union[int, Sequence[int]]] = None
    ) -> Tuple[Sequence[ObsType], dict[str, Any]]:
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert (
            len(seed) == self.num_envs
        ), f"If seeds are passed as a list the length must match num_envs={self.num_envs} but got length={len(seed)}."

        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            self._env_obs[i], env_info = env.reset(seed=single_seed)
            del env_info  # TODO: Ignore info for now, because most envs do not need extra info.

        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        return deepcopy(self._env_obs), {}
