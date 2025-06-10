import asyncio
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from gem.core import ActType, Env, ObsType
from gem.vector.sync_vector_env import AutoresetMode, ArrayType


class AsyncVectorEnv(Env):
    """Asynchronous vectorized environment execution."""

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


    async def _execute_step_in_thread(
        self, env_idx: int, action: ActType
    ) -> Tuple[ObsType, float, bool, bool]:
        """Helper to run a single environment's step (and potential reset) in a thread."""
        current_env = self.envs[env_idx]
        
        obs: Optional[ObsType] = None 
        reward: float = 0.0
        terminated: bool = False
        truncated: bool = False

        if self.autoreset_mode == AutoresetMode.NEXT_STEP:
            # If the environment is set to autoreset, check if it needs resetting
            if self._autoreset_envs[env_idx]:
                obs_reset, _ = await asyncio.to_thread(current_env.reset)
                obs = obs_reset
            else:
                # Standard step
                obs_step, r_step, term_step, trunc_step, _ = await asyncio.to_thread(
                    current_env.step, action
                )
                obs, reward, terminated, truncated = obs_step, r_step, term_step, trunc_step
        
        elif self.autoreset_mode == AutoresetMode.SAME_STEP:
            # Standard step
            obs_step, r_step, term_step, trunc_step, _ = await asyncio.to_thread(
                current_env.step, action
            )
            obs, reward, terminated, truncated = obs_step, r_step, term_step, trunc_step

            # If terminated or truncated, reset immediately in the same step
            if terminated or truncated:
                obs_reset, _ = await asyncio.to_thread(current_env.reset)
                obs = obs_reset
        else:
                raise ValueError
        
        return obs, reward, terminated, truncated

    async def step(self, actions: Sequence[ActType]) -> Tuple[
        Sequence[ObsType],
        ArrayType,
        ArrayType,
        ArrayType,
        dict[str, Any],
    ]:
        """Steps all sub-environments asynchronously with the provided actions."""
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Number of actions ({len(actions)}) does not match number of "
                f"environments ({self.num_envs})."
            )

        # Step environments in parallel using asyncio and gather
        step_tasks = [
            self._execute_step_in_thread(i, act) for i, act in enumerate(actions)
        ]
        results: Sequence[Tuple[ObsType, float, bool, bool]] = await asyncio.gather(*step_tasks)

        # Unpack results
        for i, (obs_val, r_val, term_val, trunc_val) in enumerate(results):
            self._env_obs[i] = obs_val
            self._rewards[i] = r_val
            self._terminations[i] = term_val
            self._truncations[i] = trunc_val
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._env_obs),
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            {},
        )

    async def _execute_reset_in_thread(self, env_idx: int, seed_val: Optional[int]) -> ObsType:
        """Helper to run a single environment's reset method in a separate thread."""
        obs, info = await asyncio.to_thread(self.envs[env_idx].reset, seed=seed_val)
        del info
        return obs

    async def reset(
        self, seed: Optional[Union[int, Sequence[int]]] = None
    ) -> Tuple[Sequence[ObsType], dict[str, Any]]:
        """Resets all sub-environments asynchronously."""
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert (
            len(seed) == self.num_envs
        ), f"If seeds are passed as a list the length must match num_envs={self.num_envs} but got length={len(seed)}."

        reset_tasks = [
            self._execute_reset_in_thread(i, s) for i, s in enumerate(seed)
        ]
        all_new_obs: Sequence[ObsType] = await asyncio.gather(*reset_tasks)

        for i, obs_val in enumerate(all_new_obs):
            self._env_obs[i] = obs_val
        self._rewards.fill(0.0)
        self._terminations.fill(False)
        self._truncations.fill(False)
        self._autoreset_envs.fill(False)

        return deepcopy(self._env_obs), {}
