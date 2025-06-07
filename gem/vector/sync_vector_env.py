"""Vectorized environments for parallel data generation."""

from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from gem.core import ActType, Env, ObsType

ArrayType = TypeVar("ArrayType")


class SyncVectorEnv(Env):
    """Defaults to SAME_STEP AutoresetMode, see https://farama.org/Vector-Autoreset-Mode."""

    def __init__(self, env_fns: Sequence[Callable[[], Env]]) -> None:
        super().__init__()
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(env_fns)

        # Initialize attributes used in `step` and `reset`
        self._env_obs = [None for _ in range(self.num_envs)]
        self._observations = [None]
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

    def step(self, actions: Sequence[ActType]) -> Tuple[
        Sequence[ObsType],
        ArrayType,
        ArrayType,
        ArrayType,
        dict[str, Any],
    ]:
        for i, action in enumerate(actions):
            (
                self._env_obs[i],
                self._rewards[i],
                self._terminations[i],
                self._truncations[i],
                env_info,
            ) = self.envs[i].step(action)

            if self._terminations[i] or self._truncations[i]:
                self._env_obs[i], env_info = self.envs[i].reset()

            del env_info

        return (
            self._env_obs,
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

        return self._env_obs, {}
