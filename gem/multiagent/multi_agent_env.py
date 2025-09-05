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

import abc
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, TypeVar

from gem.core import Env
from gem.utils import seeding

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
AgentID = TypeVar("AgentID", bound=str)


class MultiAgentEnv(Env):
    def __init__(self):
        self._agents: List[str] = []
        self._possible_agents: List[str] = []
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.rewards: Dict[str, float] = {}
        self.infos: Dict[str, Dict[str, Any]] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.observation_spaces: Dict[str, Any] = {}
        self.action_spaces: Dict[str, Any] = {}

    @property
    def agents(self) -> List[str]:
        return self._agents

    @agents.setter
    def agents(self, value: List[str]) -> None:
        self._agents = value

    @property
    def possible_agents(self) -> List[str]:
        return self._possible_agents

    @possible_agents.setter
    def possible_agents(self, value: List[str]) -> None:
        self._possible_agents = value

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def observation_space(self, agent: str) -> Optional[Any]:
        return self.observation_spaces.get(agent)

    def action_space(self, agent: str) -> Optional[Any]:
        return self.action_spaces.get(agent)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            seeding.set_seed(seed)

        self.agents = self.possible_agents.copy()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

    def _accumulate_rewards(self) -> None:
        for agent in self.agents:
            if agent in self.rewards:
                if agent not in self._cumulative_rewards:
                    self._cumulative_rewards[agent] = 0.0
                self._cumulative_rewards[agent] += self.rewards[agent]

    def _clear_rewards(self) -> None:
        self.rewards = {agent: 0.0 for agent in self.agents}

    def _was_dead_step(self, action: Optional[Any]) -> bool:
        return action is None

    @abc.abstractmethod
    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def render(self) -> Optional[Any]:
        return None

    def state(self) -> Any:
        raise NotImplementedError
