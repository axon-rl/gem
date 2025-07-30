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

"""Asynchronous (thread pool) vectorized environment execution."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from gem.core import ActType, ObsType
from gem.vector.vector_env import ArrayType, AutoresetMode, VectorEnv


def step_reset_env(action, env, autoreset_mode, autoreset_env):
    if autoreset_mode == AutoresetMode.NEXT_STEP:
        if autoreset_env:
            obs, info = env.reset()
            reward, terminated, truncated = 0.0, False, False
        else:
            obs, reward, terminated, truncated, info = env.step(action)
    elif autoreset_mode == AutoresetMode.SAME_STEP:
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    else:
        raise ValueError
    return obs, reward, terminated, truncated, info


class AsyncVectorEnv(VectorEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_pool_executer = ThreadPoolExecutor(max_workers=self.num_envs)

    def step(self, actions: Sequence[ActType]) -> Tuple[
        Sequence[ObsType],
        ArrayType,
        ArrayType,
        ArrayType,
        dict[str, Any],
    ]:
        results = list(
            self.thread_pool_executer.map(
                lambda args: step_reset_env(*args),
                list(
                    zip(
                        actions,
                        self.envs,
                        [self.autoreset_mode] * self.num_envs,
                        self._autoreset_envs,
                    )
                ),
            )
        )
        for i, (obs, reward, terminated, truncated, info) in enumerate(results):
            self._env_obs[i] = obs
            self._rewards[i] = reward
            self._terminations[i] = terminated
            self._truncations[i] = truncated
            self._env_infos[i] = info
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._env_obs),
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            deepcopy(self._env_infos),
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
            self._env_obs[i], self._env_infos[i] = env.reset(seed=single_seed)

        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        return deepcopy(self._env_obs), deepcopy(self._env_infos)
