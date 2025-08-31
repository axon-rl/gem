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

"""Tests for multi-agent core components."""

from typing import Any, Dict, Optional, Tuple

import pytest

from gem.multiagent.multi_agent_env import MultiAgentEnv


class TestMultiAgentEnv:
    """Test the base MultiAgentEnv class."""

    def test_initialization(self):
        """Test that MultiAgentEnv initializes correctly."""

        class SimpleMultiAgentEnv(MultiAgentEnv):
            def step(
                self, action: Any
            ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
                return "obs", 0.0, False, False, {}

        env = SimpleMultiAgentEnv()

        assert env.agents == []
        assert env.possible_agents == []
        assert env.num_agents == 0
        assert env.max_num_agents == 0
        assert isinstance(env.terminations, dict)
        assert isinstance(env.truncations, dict)
        assert isinstance(env.rewards, dict)
        assert isinstance(env.infos, dict)

    def test_agent_management(self):
        """Test agent list management."""

        class TestEnv(MultiAgentEnv):
            def step(
                self, action: Any
            ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
                return "obs", 0.0, False, False, {}

        env = TestEnv()

        # Set possible agents
        env.possible_agents = ["agent1", "agent2", "agent3"]
        assert env.max_num_agents == 3

        # Set active agents
        env.agents = ["agent1", "agent2"]
        assert env.num_agents == 2
        assert "agent1" in env.agents
        assert "agent2" in env.agents
        assert "agent3" not in env.agents

    def test_observation_and_action_spaces(self):
        """Test observation and action space methods."""

        class TestEnv(MultiAgentEnv):
            def __init__(self):
                super().__init__()
                self.observation_spaces = {
                    "agent1": "obs_space_1",
                    "agent2": "obs_space_2",
                }
                self.action_spaces = {
                    "agent1": "action_space_1",
                    "agent2": "action_space_2",
                }

            def step(
                self, action: Any
            ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
                return "obs", 0.0, False, False, {}

            def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
                super().reset(seed)
                return "obs", {}

        env = TestEnv()

        assert env.observation_space("agent1") == "obs_space_1"
        assert env.observation_space("agent2") == "obs_space_2"
        assert env.observation_space("agent3") is None

        assert env.action_space("agent1") == "action_space_1"
        assert env.action_space("agent2") == "action_space_2"
        assert env.action_space("agent3") is None

    def test_reward_accumulation(self):
        """Test reward accumulation and clearing."""

        class TestEnv(MultiAgentEnv):
            def step(
                self, action: Any
            ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
                return "obs", 0.0, False, False, {}

        env = TestEnv()
        env.agents = ["agent1", "agent2"]

        # Set initial rewards
        env.rewards = {"agent1": 1.0, "agent2": 0.5}

        # Accumulate rewards
        env._accumulate_rewards()
        assert env._cumulative_rewards["agent1"] == 1.0
        assert env._cumulative_rewards["agent2"] == 0.5

        # Accumulate again
        env.rewards = {"agent1": 0.5, "agent2": 1.0}
        env._accumulate_rewards()
        assert env._cumulative_rewards["agent1"] == 1.5
        assert env._cumulative_rewards["agent2"] == 1.5

        # Clear rewards
        env._clear_rewards()
        assert env.rewards["agent1"] == 0.0
        assert env.rewards["agent2"] == 0.0

    def test_dead_step_detection(self):
        """Test dead step detection."""

        class TestEnv(MultiAgentEnv):
            def step(
                self, action: Any
            ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
                return "obs", 0.0, False, False, {}

        env = TestEnv()

        assert env._was_dead_step(None) is True
        assert env._was_dead_step("action") is False
        assert env._was_dead_step("") is False

    def test_reset(self):
        """Test environment reset."""

        class TestEnv(MultiAgentEnv):
            def __init__(self):
                super().__init__()
                self.possible_agents = ["agent1", "agent2"]

            def step(
                self, action: Any
            ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
                return "obs", 0.0, False, False, {}

            def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
                super().reset(seed)
                return "obs", {}

        env = TestEnv()

        # Modify state
        env.agents = ["agent1"]
        env.terminations = {"agent1": True}
        env.rewards = {"agent1": 1.0}

        # Reset
        env.reset()

        # Check state is reset
        assert env.agents == ["agent1", "agent2"]
        assert env.terminations == {"agent1": False, "agent2": False}
        assert env.truncations == {"agent1": False, "agent2": False}
        assert env.rewards == {"agent1": 0.0, "agent2": 0.0}
        assert env._cumulative_rewards == {"agent1": 0.0, "agent2": 0.0}

    def test_state_not_implemented(self):
        """Test that state() raises NotImplementedError by default."""

        class TestEnv(MultiAgentEnv):
            def step(
                self, action: Any
            ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
                return "obs", 0.0, False, False, {}

        env = TestEnv()

        with pytest.raises(NotImplementedError):
            env.state()
