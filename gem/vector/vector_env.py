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
        env_ids: Sequence[str],
        env_fns: Sequence[Callable[[], Env]],
        autoreset_mode: Union[str, AutoresetMode] = AutoresetMode.SAME_STEP,
    ) -> None:
        super().__init__()
        self.env_ids = env_ids
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(env_fns)
        self.autoreset_mode = autoreset_mode

        # Initialize attributes used in `step` and `reset`
        self._env_obs = [None for _ in range(self.num_envs)]
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._env_infos = [{} for _ in range(self.num_envs)]
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)
