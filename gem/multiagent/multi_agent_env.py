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
from typing import Any, Dict, List, Optional, Tuple, Union

from gem.core import Env
from gem.multiagent.utils import AgentSelector


class MultiAgentEnv(Env):
    
    def __init__(self, simultaneous: bool = True):
        super().__init__()
        
        self.agents: List[str] = []
        self.possible_agents: List[str] = []
        
        self.simultaneous = simultaneous
        
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.rewards: Dict[str, float] = {}
        self.infos: Dict[str, dict] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        
        self._agent_selector: Optional[AgentSelector] = None
        self.agent_selection: Optional[str] = None
        
        self.shared_memory: List[str] = []
        self.global_context: str = ""
        
    @property
    def num_agents(self) -> int:
        return len(self.agents)
    
    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)
    
    @property
    def current_agent(self) -> Optional[str]:
        if not self.simultaneous and self._agent_selector:
            return self._agent_selector.selected
        return None
    
    def step(self, action: Union[str, Dict[str, str]]) -> Tuple:
        if self.simultaneous:
            if not isinstance(action, dict):
                raise ValueError(f"Simultaneous mode requires dict of actions, got {type(action)}")
            return self._step_simultaneous(action)
        else:
            if isinstance(action, dict):
                raise ValueError(f"Sequential mode requires single action, got dict")
            return self._step_sequential(action)
    
    @abc.abstractmethod
    def _step_simultaneous(self, actions: Dict[str, str]) -> Tuple[
        Dict[str, str],
        Dict[str, float], 
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict]
    ]:
        self._validate_actions(actions)
        raise NotImplementedError
    
    @abc.abstractmethod
    def _step_sequential(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        if self.current_agent is None:
            raise ValueError("No agent selected for sequential step")
        raise NotImplementedError
    
    def reset(self, seed: Optional[int] = None) -> Tuple:
        super().reset(seed)
        
        self.agents = self.possible_agents.copy()
        
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        self.shared_memory = []
        self.global_context = ""
        
        if not self.simultaneous:
            self._agent_selector = AgentSelector(self.agents)
            self.agent_selection = self._agent_selector.selected
        
        if self.simultaneous:
            observations = {agent: self.observe(agent) for agent in self.agents}
            infos = {agent: {} for agent in self.agents}
            return observations, infos
        else:
            return self.observe(self.current_agent), {}
    
    @abc.abstractmethod
    def observe(self, agent: str) -> str:
        raise NotImplementedError
    
    def agent_iter(self, max_iter: int = 2**63):
        if self.simultaneous:
            raise ValueError("agent_iter is only for sequential mode")
        
        return AECIterator(self, max_iter)
    
    def last(self, observe: bool = True) -> Tuple[str, float, bool, bool, dict]:
        if self.simultaneous:
            raise ValueError("last() is only for sequential mode")
        
        if self.current_agent is None:
            raise ValueError("No agent selected")
        
        agent = self.current_agent
        
        obs = self.observe(agent) if observe else None
        reward = self._cumulative_rewards.get(agent, 0.0)
        terminated = self.terminations.get(agent, False)
        truncated = self.truncations.get(agent, False)
        info = self.infos.get(agent, {})
        
        return obs, reward, terminated, truncated, info
    
    def add_agent(self, agent_id: str, role: str = "participant"):
        if agent_id not in self.possible_agents:
            self.possible_agents.append(agent_id)
        if agent_id not in self.agents:
            self.agents.append(agent_id)
            self.terminations[agent_id] = False
            self.truncations[agent_id] = False
            self.rewards[agent_id] = 0.0
            self._cumulative_rewards[agent_id] = 0.0
            self.infos[agent_id] = {"role": role}
            
            if not self.simultaneous and self._agent_selector:
                self._agent_selector.reinit(self.agents)
    
    def remove_agent(self, agent_id: str):
        if agent_id in self.agents:
            self.agents.remove(agent_id)
            self.terminations[agent_id] = True
            
            if not self.simultaneous and self._agent_selector:
                self._agent_selector.remove_agent(agent_id)
    
    def _validate_actions(self, actions: Dict[str, str]) -> None:
        action_agents = set(actions.keys())
        active_agents = set(self.agents)
        
        if action_agents != active_agents:
            missing = active_agents - action_agents
            extra = action_agents - active_agents
            
            error_parts = []
            if missing:
                error_parts.append(f"Missing actions for agents: {sorted(missing)}")
            if extra:
                error_parts.append(f"Actions provided for non-active agents: {sorted(extra)}")
            
            raise ValueError(". ".join(error_parts))
    
    def _accumulate_rewards(self):
        for agent in self.agents:
            if agent in self.rewards:
                self._cumulative_rewards[agent] += self.rewards[agent]
    
    def _clear_rewards(self):
        for agent in self.agents:
            self.rewards[agent] = 0.0
    
    def _was_dead_step(self, action: Optional[str]) -> bool:
        return action is None
    
    def _remove_dead_agents(self):
        self.agents = [
            agent for agent in self.agents
            if not (self.terminations.get(agent, False) or self.truncations.get(agent, False))
        ]
        
        if not self.simultaneous and self._agent_selector and self.agents:
            self._agent_selector.reinit(self.agents)
            self.agent_selection = self._agent_selector.selected
        elif not self.agents:
            self.agent_selection = None
    
    def observation_space(self, agent: str) -> Any:
        return getattr(self, "observation_spaces", {}).get(agent)
    
    def action_space(self, agent: str) -> Any:
        return getattr(self, "action_spaces", {}).get(agent)
    
    def state(self) -> Any:
        raise NotImplementedError
    
    def send_message(self, from_agent: str, to_agent: str, message: str):
        if not hasattr(self, "message_buffer"):
            self.message_buffer = {}
        if to_agent not in self.message_buffer:
            self.message_buffer[to_agent] = []
        self.message_buffer[to_agent].append({
            "from": from_agent,
            "message": message
        })
    
    def get_messages(self, agent: str) -> List[Dict]:
        if not hasattr(self, "message_buffer"):
            return []
        messages = self.message_buffer.get(agent, [])
        if agent in self.message_buffer:
            self.message_buffer[agent] = []
        return messages


class AECIterator:
    
    def __init__(self, env: MultiAgentEnv, max_iter: int):
        self.env = env
        self.max_iter = max_iter
        self._current_iter = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        if self._current_iter >= self.max_iter:
            raise StopIteration
        
        if not self.env.agents:
            raise StopIteration
        
        if all(self.env.terminations.get(a, False) or self.env.truncations.get(a, False) 
               for a in self.env.agents):
            raise StopIteration
        
        self._current_iter += 1
        return self.env.current_agent