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

from typing import Any, Dict, Optional, Tuple

import pytest

from gem.multiagent.aec_env import AECEnv, AECIterable
from gem.multiagent.multi_agent_env import MultiAgentEnv
from gem.multiagent.utils import AgentSelector


class SimpleAECEnv(AECEnv):

    def __init__(self):
        super().__init__()
        self.possible_agents = ["agent1", "agent2", "agent3"]
        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.selected

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.step_count = 0
        self.max_steps = 10

    def observe(self, agent: str) -> str:
        return f"Observation for {agent} at step {self.step_count}"

    def step(self, action: Optional[str]) -> None:
        if self.agent_selection is None:
            return

        current_agent = self.agent_selection

        if not self._was_dead_step(action):
            self.rewards[current_agent] = 1.0 if action == "good" else 0.0
            self._cumulative_rewards[current_agent] += self.rewards[current_agent]

            if action == "terminate":
                self.terminations[current_agent] = True

        next_agent = self._agent_selector.next()
        self.agent_selection = next_agent

        if self._agent_selector.is_first():
            self.step_count += 1

            if self.step_count >= self.max_steps:
                for agent in self.agents:
                    self.truncations[agent] = True

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        MultiAgentEnv.reset(self, seed)

        self.agents = self.possible_agents.copy()
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.selected

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.step_count = 0

        return self.observe(self.agent_selection), {}


class TestAECEnv:

    def test_initialization(self):
        env = SimpleAECEnv()

        assert env.agent_selection == "agent1"
        assert len(env.agents) == 3
        assert env.step_count == 0

    def test_observe(self):
        env = SimpleAECEnv()

        obs = env.observe("agent1")
        assert "agent1" in obs
        assert "step 0" in obs

        obs = env.observe("agent2")
        assert "agent2" in obs

    def test_last(self):
        env = SimpleAECEnv()
        env.reset()

        obs, reward, terminated, truncated, info = env.last()

        assert "agent1" in obs
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert isinstance(info, dict)

        obs, reward, terminated, truncated, info = env.last(observe=False)
        assert obs is None

    def test_last_no_agent_selected(self):
        env = SimpleAECEnv()
        env.agent_selection = None

        with pytest.raises(ValueError, match="No agent selected"):
            env.last()

    def test_step_with_actions(self):
        env = SimpleAECEnv()
        env.reset()

        env.step("good")
        assert env.agent_selection == "agent2"

        assert env._cumulative_rewards["agent1"] == 1.0

        env.step("bad")
        assert env.agent_selection == "agent3"
        assert env._cumulative_rewards["agent2"] == 0.0

        env.step("good")
        assert env.agent_selection == "agent1"
        assert env.step_count == 1

    def test_termination(self):
        env = SimpleAECEnv()
        env.reset()

        env.step("terminate")
        assert env.terminations["agent1"] is True

        assert env.terminations["agent2"] is False
        assert env.terminations["agent3"] is False

    def test_truncation(self):
        env = SimpleAECEnv()
        env.reset()
        env.max_steps = 2

        for _ in range(6):
            env.step("action")

        assert all(env.truncations.values())

    def test_dead_step(self):
        env = SimpleAECEnv()
        env.reset()

        env.terminations["agent1"] = True

        assert env._was_dead_step(None) is True
        env.step(None)

        assert env.agent_selection == "agent2"
        assert env._cumulative_rewards["agent1"] == 0.0

    def test_agent_iter(self):
        env = SimpleAECEnv()
        env.reset()

        agents_seen = []
        for i, agent in enumerate(env.agent_iter(max_iter=9)):
            agents_seen.append(agent)
            obs, reward, terminated, truncated, info = env.last()

            if terminated or truncated:
                action = None
            else:
                action = "action"

            env.step(action)

            if i >= 8:
                break

        assert agents_seen == ["agent1", "agent2", "agent3"] * 3

    def test_agent_iter_with_termination(self):
        env = SimpleAECEnv()
        env.reset()

        for agent in env.agents:
            env.terminations[agent] = True

        agents_seen = list(env.agent_iter())
        assert len(agents_seen) == 0

    def test_reset(self):
        env = SimpleAECEnv()
        env.reset()

        env.step("good")
        env.step("bad")
        env.terminations["agent1"] = True
        env.step_count = 5

        obs, info = env.reset()

        assert env.agent_selection == "agent1"
        assert env.step_count == 0
        assert all(not term for term in env.terminations.values())
        assert all(not trunc for trunc in env.truncations.values())
        assert all(reward == 0.0 for reward in env._cumulative_rewards.values())
        assert "agent1" in obs


class TestAECIterable:

    def test_basic_iteration(self):
        env = SimpleAECEnv()
        env.reset()

        iterator = AECIterable(env, max_iter=6)
        agents = list(iterator)

        assert len(agents) == 6
        assert agents == ["agent1", "agent2", "agent3", "agent1", "agent2", "agent3"]

    def test_max_iter_limit(self):
        env = SimpleAECEnv()
        env.reset()

        iterator = AECIterable(env, max_iter=2)
        agents = list(iterator)

        assert len(agents) == 2
        assert agents == ["agent1", "agent2"]

    def test_no_agents(self):
        env = SimpleAECEnv()
        env.reset()
        env.agents = []

        iterator = AECIterable(env, max_iter=10)
        agents = list(iterator)

        assert len(agents) == 0

    def test_all_terminated(self):
        env = SimpleAECEnv()
        env.reset()

        for agent in env.agents:
            env.terminations[agent] = True

        iterator = AECIterable(env, max_iter=10)
        agents = list(iterator)

        assert len(agents) == 0
