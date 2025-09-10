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

from gem.multiagent import MultiAgentEnv, AgentSelector


class SimpleSequentialEnv(MultiAgentEnv):
    
    def __init__(self):
        super().__init__(simultaneous=False)
        self.possible_agents = ["agent1", "agent2", "agent3"]
        self.step_count = 0
        self.max_steps = 10
        
    def observe(self, agent: str) -> str:
        return f"Observation for {agent} at step {self.step_count}"
    
    def _step_sequential(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        current = self.current_agent
        
        if action == "terminate":
            self.terminations[current] = True
        
        reward = 1.0 if action == "good" else 0.0
        self.rewards[current] = reward
        self._cumulative_rewards[current] = self._cumulative_rewards.get(current, 0.0) + reward
        
        if self._agent_selector:
            self._agent_selector.next()
            self.agent_selection = self._agent_selector.selected
            
            if self._agent_selector.is_first():
                self.step_count += 1
                if self.step_count >= self.max_steps:
                    for agent in self.agents:
                        self.truncations[agent] = True
        
        obs = self.observe(self.current_agent) if self.current_agent else ""
        return obs, reward, self.terminations[current], self.truncations[current], {}
    
    def _step_simultaneous(self, actions: Dict[str, str]):
        raise NotImplementedError("This is a sequential environment")


class SimpleSimultaneousEnv(MultiAgentEnv):
    
    def __init__(self):
        super().__init__(simultaneous=True)
        self.possible_agents = ["agent1", "agent2", "agent3"]
        self.step_count = 0
        self.max_steps = 10
        
    def observe(self, agent: str) -> str:
        return f"Step {self.step_count} observation for {agent}"
    
    def _step_simultaneous(self, actions: Dict[str, str]) -> Tuple:
        self._validate_actions(actions)
        
        observations = {}
        rewards = {}
        
        for agent, action in actions.items():
            observations[agent] = f"Result for {agent} after {action}"
            
            if action == "good":
                rewards[agent] = 1.0
            elif action == "bad":
                rewards[agent] = -1.0
            else:
                rewards[agent] = 0.0
            
            if action == "terminate":
                self.terminations[agent] = True
        
        self.step_count += 1
        
        if self.step_count >= self.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True
        
        self._remove_dead_agents()
        
        return observations, rewards, self.terminations, self.truncations, self.infos
    
    def _step_sequential(self, action: str):
        raise NotImplementedError("This is a simultaneous environment")


class TestMultiAgentEnvBase:
    
    def test_initialization_sequential(self):
        env = SimpleSequentialEnv()
        assert env.simultaneous is False
        assert env.agents == []
        assert env.possible_agents == ["agent1", "agent2", "agent3"]
        assert env.num_agents == 0
        assert env.max_num_agents == 3
    
    def test_initialization_simultaneous(self):
        env = SimpleSimultaneousEnv()
        assert env.simultaneous is True
        assert env.agents == []
        assert env.possible_agents == ["agent1", "agent2", "agent3"]
    
    def test_reset_sequential(self):
        env = SimpleSequentialEnv()
        obs, info = env.reset()
        
        assert isinstance(obs, str)
        assert isinstance(info, dict)
        assert env.agents == ["agent1", "agent2", "agent3"]
        assert env.current_agent == "agent1"
        assert all(not env.terminations[a] for a in env.agents)
    
    def test_reset_simultaneous(self):
        env = SimpleSimultaneousEnv()
        obs, info = env.reset()
        
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert len(obs) == 3
        assert all(agent in obs for agent in env.agents)
    
    def test_step_wrong_type_sequential(self):
        env = SimpleSequentialEnv()
        env.reset()
        
        with pytest.raises(ValueError, match="Sequential mode requires single action"):
            env.step({"agent1": "action"})
    
    def test_step_wrong_type_simultaneous(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        
        with pytest.raises(ValueError, match="Simultaneous mode requires dict"):
            env.step("action")
    
    def test_add_remove_agent(self):
        env = SimpleSequentialEnv()
        env.reset()
        
        env.add_agent("agent4", role="new_agent")
        assert "agent4" in env.agents
        assert "agent4" in env.possible_agents
        assert env.infos["agent4"]["role"] == "new_agent"
        
        env.remove_agent("agent4")
        assert "agent4" not in env.agents
        assert env.terminations["agent4"] is True
    
    def test_message_passing(self):
        env = SimpleSequentialEnv()
        env.reset()
        
        env.send_message("agent1", "agent2", "Hello")
        messages = env.get_messages("agent2")
        
        assert len(messages) == 1
        assert messages[0]["from"] == "agent1"
        assert messages[0]["message"] == "Hello"
        
        messages = env.get_messages("agent2")
        assert len(messages) == 0


class TestSequentialMode:
    
    def test_basic_stepping(self):
        env = SimpleSequentialEnv()
        env.reset()
        
        assert env.current_agent == "agent1"
        
        obs, reward, term, trunc, info = env.step("good")
        assert env.current_agent == "agent2"
        assert reward == 1.0
        
        obs, reward, term, trunc, info = env.step("bad")
        assert env.current_agent == "agent3"
        assert reward == 0.0
        
        obs, reward, term, trunc, info = env.step("neutral")
        assert env.current_agent == "agent1"
        assert env.step_count == 1
    
    def test_agent_iteration(self):
        env = SimpleSequentialEnv()
        env.reset()
        
        agents_seen = []
        for i, agent in enumerate(env.agent_iter(max_iter=9)):
            agents_seen.append(agent)
            obs, reward, term, trunc, info = env.last()
            
            if term or trunc:
                action = None
            else:
                action = "action"
            
            obs, reward, term, trunc, info = env.step(action)
            
            if i >= 8:
                break
        
        assert agents_seen == ["agent1", "agent2", "agent3"] * 3
    
    def test_last_method(self):
        env = SimpleSequentialEnv()
        env.reset()
        
        obs, reward, term, trunc, info = env.last()
        assert "agent1" in obs
        assert reward == 0.0
        assert term is False
        assert trunc is False
        
        obs, reward, term, trunc, info = env.last(observe=False)
        assert obs is None
    
    def test_termination(self):
        env = SimpleSequentialEnv()
        env.reset()
        
        env.step("terminate")
        assert env.terminations["agent1"] is True
        assert env.terminations["agent2"] is False
        assert env.terminations["agent3"] is False
    
    def test_truncation(self):
        env = SimpleSequentialEnv()
        env.reset()
        env.max_steps = 2
        
        for _ in range(6):
            env.step("action")
        
        assert all(env.truncations.values())
    
    def test_cumulative_rewards(self):
        env = SimpleSequentialEnv()
        env.reset()
        
        env.step("good")
        assert env._cumulative_rewards["agent1"] == 1.0
        
        env.step("good")
        assert env._cumulative_rewards["agent2"] == 1.0
        
        env.step("good")
        env.step("good")
        assert env._cumulative_rewards["agent1"] == 2.0
    
    def test_agent_iter_with_simultaneous_raises(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        
        with pytest.raises(ValueError, match="agent_iter is only for sequential"):
            for agent in env.agent_iter():
                pass
    
    def test_last_with_simultaneous_raises(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        
        with pytest.raises(ValueError, match="last\\(\\) is only for sequential"):
            env.last()


class TestSimultaneousMode:
    
    def test_basic_stepping(self):
        env = SimpleSimultaneousEnv()
        obs, info = env.reset()
        
        actions = {
            "agent1": "good",
            "agent2": "bad",
            "agent3": "neutral"
        }
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        assert len(obs) == 3
        assert rewards["agent1"] == 1.0
        assert rewards["agent2"] == -1.0
        assert rewards["agent3"] == 0.0
        assert env.step_count == 1
    
    def test_missing_actions(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        
        actions = {
            "agent1": "good",
            "agent2": "bad"
        }
        
        with pytest.raises(ValueError, match="Missing actions"):
            env.step(actions)
    
    def test_extra_actions(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        
        actions = {
            "agent1": "good",
            "agent2": "bad",
            "agent3": "neutral",
            "agent4": "extra"
        }
        
        with pytest.raises(ValueError, match="non-active agents"):
            env.step(actions)
    
    def test_termination(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        
        actions = {
            "agent1": "terminate",
            "agent2": "good",
            "agent3": "good"
        }
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        assert terms["agent1"] is True
        assert terms["agent2"] is False
        assert terms["agent3"] is False
        
        assert "agent1" not in env.agents
        assert len(env.agents) == 2
    
    def test_truncation(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        env.max_steps = 2
        
        actions = {agent: "action" for agent in env.agents}
        
        env.step(actions)
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        assert all(truncs.values())
    
    def test_all_agents_terminate(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        
        actions = {
            "agent1": "terminate",
            "agent2": "terminate",
            "agent3": "terminate"
        }
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        assert all(terms.values())
        assert len(env.agents) == 0
    
    def test_no_current_agent(self):
        env = SimpleSimultaneousEnv()
        env.reset()
        
        assert env.current_agent is None


class TestAgentSelector:
    
    def test_initialization(self):
        selector = AgentSelector(["a", "b", "c"])
        assert selector.selected == "a"
        assert selector.agents == ["a", "b", "c"]
    
    def test_next(self):
        selector = AgentSelector(["a", "b", "c"])
        
        assert selector.next() == "b"
        assert selector.selected == "b"
        
        assert selector.next() == "c"
        assert selector.selected == "c"
        
        assert selector.next() == "a"
        assert selector.selected == "a"
    
    def test_is_first_last(self):
        selector = AgentSelector(["a", "b", "c"])
        
        assert selector.is_first() is True
        assert selector.is_last() is False
        
        selector.next()
        assert selector.is_first() is False
        assert selector.is_last() is False
        
        selector.next()
        assert selector.is_first() is False
        assert selector.is_last() is True
    
    def test_remove_agent(self):
        selector = AgentSelector(["a", "b", "c"])
        
        selector.next()
        selector.remove_agent("b")
        
        assert selector.agents == ["a", "c"]
        assert selector.selected == "c"
    
    def test_reinit(self):
        selector = AgentSelector(["a", "b", "c"])
        selector.next()
        selector.next()
        
        selector.reinit(["x", "y", "z"])
        assert selector.agents == ["x", "y", "z"]
        assert selector.selected == "x"
    
    def test_empty_agents(self):
        selector = AgentSelector([])
        assert selector.selected is None
        assert selector.next() is None