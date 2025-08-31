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

from typing import Any, Dict, List, Optional, Tuple

from gem.core import EnvWrapper
from gem.multiagent.aec_env import AECEnv
from gem.multiagent.parallel_env import ParallelEnv


class AgentSelector:
    def __init__(self, agents: List[str]):
        self.agents = agents.copy()
        self._current_index = 0
        self.selected = agents[0] if agents else None

    def next(self) -> Optional[str]:
        if not self.agents:
            return None
        self._current_index = (self._current_index + 1) % len(self.agents)
        self.selected = self.agents[self._current_index]
        return self.selected

    def reset(self) -> Optional[str]:
        self._current_index = 0
        self.selected = self.agents[0] if self.agents else None
        return self.selected

    def is_first(self) -> bool:
        return self._current_index == 0

    def is_last(self) -> bool:
        if not self.agents:
            return True
        return self._current_index == len(self.agents) - 1

    def reinit(self, agents: List[str]) -> None:
        self.agents = agents.copy()
        self._current_index = 0
        self.selected = agents[0] if agents else None

    def remove_agent(self, agent: str) -> None:
        if agent not in self.agents:
            return

        current_agent = self.selected
        agent_index = self.agents.index(agent)

        self.agents.remove(agent)

        if not self.agents:
            self.selected = None
            self._current_index = 0
            return

        if current_agent == agent:
            if agent_index < len(self.agents):
                self._current_index = agent_index
            else:
                self._current_index = 0
            self.selected = self.agents[self._current_index]
        elif agent_index < self._current_index:
            self._current_index -= 1

    def agent_order(self) -> List[str]:
        return self.agents.copy()

    def __len__(self) -> int:
        return len(self.agents)


class AECToParallelWrapper(ParallelEnv, EnvWrapper):
    def __init__(self, aec_env: AECEnv):
        ParallelEnv.__init__(self)
        EnvWrapper.__init__(self, aec_env)

        self.aec_env = aec_env
        self.possible_agents = aec_env.possible_agents.copy()
        self.agents = aec_env.agents.copy()
        self.observation_spaces = aec_env.observation_spaces.copy()
        self.action_spaces = aec_env.action_spaces.copy()

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        first_obs, first_info = self.aec_env.reset(seed)

        self.agents = self.aec_env.agents.copy()

        observations = {}
        infos = {}

        if self.aec_env.agent_selection:
            observations[self.aec_env.agent_selection] = first_obs
            infos[self.aec_env.agent_selection] = first_info

        for agent in self.agents:
            if agent not in observations:
                observations[agent] = self.aec_env.observe(agent)
                infos[agent] = self.aec_env.infos.get(agent, {})

        return observations, infos

    def step(self, actions: Dict[str, str]) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        self._validate_actions(actions)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        agents_to_step = self.agents.copy()

        for agent in agents_to_step:
            if agent in actions:
                self.aec_env.agent_selection = agent
                obs_before = self.aec_env.observe(agent)

                self.aec_env.step(actions[agent])

                observations[agent] = self.aec_env.observe(agent)
                rewards[agent] = self.aec_env._cumulative_rewards.get(agent, 0.0)
                terminations[agent] = self.aec_env.terminations.get(agent, False)
                truncations[agent] = self.aec_env.truncations.get(agent, False)
                infos[agent] = self.aec_env.infos.get(agent, {})

                self.aec_env._cumulative_rewards[agent] = 0.0

        self.agents = [
            agent
            for agent in self.agents
            if not (terminations.get(agent, False) or truncations.get(agent, False))
        ]

        return observations, rewards, terminations, truncations, infos


class ParallelToAECWrapper(AECEnv, EnvWrapper):
    def __init__(self, parallel_env: ParallelEnv):
        AECEnv.__init__(self)
        EnvWrapper.__init__(self, parallel_env)

        self.parallel_env = parallel_env
        self.possible_agents = parallel_env.possible_agents.copy()
        self.agents = parallel_env.agents.copy()
        self.observation_spaces = parallel_env.observation_spaces.copy()
        self.action_spaces = parallel_env.action_spaces.copy()

        self._action_buffer: Dict[str, Any] = {}
        self._observations: Dict[str, str] = {}
        self._rewards: Dict[str, float] = {}
        self._terminations: Dict[str, bool] = {}
        self._truncations: Dict[str, bool] = {}
        self._infos: Dict[str, Dict] = {}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.selected

    def observe(self, agent: str) -> str:
        return self._observations.get(agent, "")

    def step(self, action: Optional[str]) -> None:
        if self.agent_selection is None:
            return

        if action is not None:
            self._action_buffer[self.agent_selection] = action

        if self._agent_selector.is_last():
            actions = {}
            for agent in self.agents:
                if agent in self._action_buffer:
                    actions[agent] = self._action_buffer[agent]
                else:
                    actions[agent] = None

            (
                self._observations,
                self._rewards,
                self._terminations,
                self._truncations,
                self._infos,
            ) = self.parallel_env.step(actions)

            for agent in self.agents:
                self.rewards[agent] = self._rewards.get(agent, 0.0)
                self._cumulative_rewards[agent] += self.rewards[agent]
                self.terminations[agent] = self._terminations.get(agent, False)
                self.truncations[agent] = self._truncations.get(agent, False)
                self.infos[agent] = self._infos.get(agent, {})

            self._action_buffer.clear()

            self.agents = [
                agent
                for agent in self.agents
                if not (self.terminations[agent] or self.truncations[agent])
            ]

            if self.agents:
                self._agent_selector.reinit(self.agents)
                self.agent_selection = self._agent_selector.selected
            else:
                self.agent_selection = None
        else:
            self._agent_selector.next()
            self.agent_selection = self._agent_selector.selected

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        observations, infos = self.parallel_env.reset(seed)

        self.agents = self.parallel_env.agents.copy()
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.selected

        self._observations = observations
        self._infos = infos
        self._action_buffer.clear()

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        first_agent = self.agent_selection
        return self._observations.get(first_agent, ""), self._infos.get(first_agent, {})


def aec_to_parallel(aec_env: AECEnv) -> ParallelEnv:
    return AECToParallelWrapper(aec_env)


def parallel_to_aec(parallel_env: ParallelEnv) -> AECEnv:
    return ParallelToAECWrapper(parallel_env)
