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

"""Tests for environment conversions between AEC and Parallel."""

from typing import Any, Dict, Optional, Tuple

import pytest

from gem.multiagent.aec_env import AECEnv
from gem.multiagent.utils import AgentSelector
from gem.multiagent.utils import (
    AECToParallelWrapper,
    ParallelToAECWrapper,
    aec_to_parallel,
    parallel_to_aec,
)
from gem.multiagent.multi_agent_env import MultiAgentEnv
from gem.multiagent.parallel_env import ParallelEnv


class MockAECEnv(AECEnv):
    """Mock AEC environment for testing."""

    def __init__(self):
        super().__init__()
        self.possible_agents = ["agent1", "agent2"]
        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.selected

        self.observation_spaces = {agent: "obs_space" for agent in self.possible_agents}
        self.action_spaces = {agent: "action_space" for agent in self.possible_agents}

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.step_count = 0

    def observe(self, agent: str) -> str:
        return f"obs_{agent}_{self.step_count}"

    def step(self, action: Optional[str]) -> None:
        if self.agent_selection and action:
            self.rewards[self.agent_selection] = 1.0 if action == "good" else 0.0
            self._cumulative_rewards[self.agent_selection] += self.rewards[
                self.agent_selection
            ]

            if action == "terminate":
                self.terminations[self.agent_selection] = True

        self._agent_selector.next()
        self.agent_selection = self._agent_selector.selected

        if self._agent_selector.is_first():
            self.step_count += 1

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        # Call MultiAgentEnv.reset directly, not AECEnv.reset
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


class MockParallelEnv(ParallelEnv):
    """Mock Parallel environment for testing."""

    def __init__(self):
        super().__init__()
        self.possible_agents = ["agent1", "agent2"]
        self.agents = self.possible_agents.copy()

        self.observation_spaces = {agent: "obs_space" for agent in self.possible_agents}
        self.action_spaces = {agent: "action_space" for agent in self.possible_agents}

        self.step_count = 0

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

        for agent, action in actions.items():
            observations[agent] = f"obs_{agent}_{self.step_count + 1}"
            rewards[agent] = 1.0 if action == "good" else 0.0
            terminations[agent] = action == "terminate"
            truncations[agent] = False
            infos[agent] = {"step": self.step_count + 1}

        self.step_count += 1

        # Update internal state
        self.terminations = terminations
        self.truncations = truncations

        # Remove dead agents
        self._remove_dead_agents()

        return observations, rewards, terminations, truncations, infos

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        # Call MultiAgentEnv.reset directly, not ParallelEnv.reset
        MultiAgentEnv.reset(self, seed)
        self.agents = self.possible_agents.copy()
        self.step_count = 0

        observations = {agent: f"obs_{agent}_0" for agent in self.agents}
        infos = {agent: {"initial": True} for agent in self.agents}

        return observations, infos


class TestAECToParallelWrapper:
    """Test AEC to Parallel conversion."""

    def test_initialization(self):
        """Test wrapper initialization."""
        aec_env = MockAECEnv()
        wrapper = AECToParallelWrapper(aec_env)

        assert wrapper.possible_agents == aec_env.possible_agents
        assert wrapper.agents == aec_env.agents
        assert wrapper.observation_spaces == aec_env.observation_spaces
        assert wrapper.action_spaces == aec_env.action_spaces

    def test_reset(self):
        """Test reset through wrapper."""
        aec_env = MockAECEnv()
        wrapper = AECToParallelWrapper(aec_env)

        observations, infos = wrapper.reset()

        assert len(observations) == 2
        assert "agent1" in observations
        assert "agent2" in observations
        assert observations["agent1"] == "obs_agent1_0"
        assert observations["agent2"] == "obs_agent2_0"

    def test_step(self):
        """Test parallel step through wrapper."""
        aec_env = MockAECEnv()
        wrapper = AECToParallelWrapper(aec_env)
        wrapper.reset()

        actions = {"agent1": "good", "agent2": "bad"}
        obs, rewards, terms, truncs, infos = wrapper.step(actions)

        assert len(obs) == 2
        assert len(rewards) == 2
        assert rewards["agent1"] == 1.0
        assert rewards["agent2"] == 0.0
        assert all(not term for term in terms.values())

    def test_step_with_termination(self):
        """Test step with agent termination."""
        aec_env = MockAECEnv()
        wrapper = AECToParallelWrapper(aec_env)
        wrapper.reset()

        actions = {"agent1": "terminate", "agent2": "good"}
        obs, rewards, terms, truncs, infos = wrapper.step(actions)

        assert terms["agent1"] is True
        assert terms["agent2"] is False
        assert len(wrapper.agents) == 1
        assert "agent2" in wrapper.agents

    def test_validate_actions(self):
        """Test action validation."""
        aec_env = MockAECEnv()
        wrapper = AECToParallelWrapper(aec_env)
        wrapper.reset()

        # Missing action
        with pytest.raises(ValueError, match="Missing actions"):
            wrapper.step({"agent1": "good"})

        # Extra action
        with pytest.raises(ValueError, match="Actions provided for non-active agents"):
            wrapper.step({"agent1": "good", "agent2": "good", "agent3": "good"})


class TestParallelToAECWrapper:
    """Test Parallel to AEC conversion."""

    def test_initialization(self):
        """Test wrapper initialization."""
        parallel_env = MockParallelEnv()
        wrapper = ParallelToAECWrapper(parallel_env)

        assert wrapper.possible_agents == parallel_env.possible_agents
        assert wrapper.agents == parallel_env.agents
        assert wrapper.observation_spaces == parallel_env.observation_spaces
        assert wrapper.action_spaces == parallel_env.action_spaces
        assert wrapper.agent_selection is not None

    def test_reset(self):
        """Test reset through wrapper."""
        parallel_env = MockParallelEnv()
        wrapper = ParallelToAECWrapper(parallel_env)

        obs, info = wrapper.reset()

        assert wrapper.agent_selection == "agent1"
        assert obs == "obs_agent1_0"
        assert isinstance(info, dict)

    def test_observe(self):
        """Test observation method."""
        parallel_env = MockParallelEnv()
        wrapper = ParallelToAECWrapper(parallel_env)
        wrapper.reset()

        obs1 = wrapper.observe("agent1")
        obs2 = wrapper.observe("agent2")

        assert obs1 == "obs_agent1_0"
        assert obs2 == "obs_agent2_0"

    def test_step_buffering(self):
        """Test action buffering in AEC mode."""
        parallel_env = MockParallelEnv()
        wrapper = ParallelToAECWrapper(parallel_env)
        wrapper.reset()

        # First agent action - should be buffered
        assert wrapper.agent_selection == "agent1"
        wrapper.step("good")

        assert "agent1" in wrapper._action_buffer
        assert wrapper._action_buffer["agent1"] == "good"
        assert wrapper.agent_selection == "agent2"

        # Second agent action - should trigger parallel step
        wrapper.step("bad")

        # Action buffer should be cleared after parallel step
        assert len(wrapper._action_buffer) == 0
        assert wrapper.agent_selection == "agent1"  # Back to first agent

    def test_last(self):
        """Test last() method."""
        parallel_env = MockParallelEnv()
        wrapper = ParallelToAECWrapper(parallel_env)
        wrapper.reset()

        # Get last for first agent
        obs, reward, terminated, truncated, info = wrapper.last()

        assert obs == "obs_agent1_0"
        assert reward == 0.0
        assert terminated is False
        assert truncated is False

    def test_full_cycle(self):
        """Test a full cycle of actions."""
        parallel_env = MockParallelEnv()
        wrapper = ParallelToAECWrapper(parallel_env)
        wrapper.reset()

        # Complete one full cycle
        wrapper.step("good")  # agent1
        wrapper.step("bad")  # agent2

        # Check that parallel step was executed
        assert parallel_env.step_count == 1

        # Check rewards were updated
        obs, reward, _, _, _ = wrapper.last()
        # Note: reward for agent1 should be available after the cycle
        assert wrapper._rewards["agent1"] == 1.0
        assert wrapper._rewards["agent2"] == 0.0

    def test_dead_step(self):
        """Test handling of dead steps."""
        parallel_env = MockParallelEnv()
        wrapper = ParallelToAECWrapper(parallel_env)
        wrapper.reset()

        # Terminate an agent
        wrapper._terminations["agent1"] = True

        # Dead step should not add to buffer
        wrapper.step(None)
        assert "agent1" not in wrapper._action_buffer
        assert wrapper.agent_selection == "agent2"


class TestConversionFunctions:
    """Test the convenience conversion functions."""

    def test_aec_to_parallel_function(self):
        """Test aec_to_parallel convenience function."""
        aec_env = MockAECEnv()
        parallel_env = aec_to_parallel(aec_env)

        assert isinstance(parallel_env, ParallelEnv)
        assert isinstance(parallel_env, AECToParallelWrapper)

        # Test it works
        observations, infos = parallel_env.reset()
        assert len(observations) == 2

    def test_parallel_to_aec_function(self):
        """Test parallel_to_aec convenience function."""
        parallel_env = MockParallelEnv()
        aec_env = parallel_to_aec(parallel_env)

        assert isinstance(aec_env, AECEnv)
        assert isinstance(aec_env, ParallelToAECWrapper)

        # Test it works
        obs, info = aec_env.reset()
        assert isinstance(obs, str)
        assert aec_env.agent_selection is not None
