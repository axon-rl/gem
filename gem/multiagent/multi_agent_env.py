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
import warnings
from typing import Any, Dict, List, Union, Optional, Tuple

from gem.core import Env


class MultiAgentEnv(Env):
    '''
    Base multi-agent environment
    '''

    def __init__(self):
        super().__init__()

        self.agents: List[str] = []
        self.active_mask: Dict[str, bool] = {}

        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.rewards: Dict[str, float] = {}
        self.infos: Dict[str, dict] = {}
        self._cumulative_rewards: Dict[str, float] = {}

        self._agent_iter = None

        self.shared_memory = []
        self.global_context = ""

    def _reset_agent_iter(self):
        self._agent_iter = AgentIterator(self.agents, self.active_mask)

    @property
    def agent_iter(self):
        if not self._agent_iter or self._agent_iter.is_end():
            self._reset_agent_iter()
        return self._agent_iter

    def step(self, action_or_actions: Union[str, Dict[str, str]]) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        '''
        Master function for environment stepping.

        By default, will attempt to call the Simultaneous step function.
        If not implemented, will fall back to sequential stepping.
        '''
        ret = None
        if isinstance(action_or_actions, dict):
            try:
                ret = self._step(action_or_actions)
            except NotImplementedError:
                warnings.warn(
                    "Simultaneous step not implemented, falling back to sequential stepping."
                )
                for agent in self.agent_iter:
                    # Don't silently fail, if the action is invalid
                    # let it raise an error.
                    self._handle_single_step(action_or_actions[agent])
                ret = None
            return self._step_global_dynamics(ret)
        else:
            ret = self._handle_single_step(action_or_actions)
            if self._agent_iter.is_end():
                ret = self._step_global_dynamics()
            return ret
    
    def _handle_single_step(self, action: str) -> None:
        '''
        Validation wrapper over a sequential step function.
        '''
        current_agent = self._agent_iter.current_agent
        if not current_agent:
            raise ValueError("No active agent selected")
        if current_agent not in self.agents:
            raise ValueError(f"Agent {current_agent} not in environment")
        
        self._step_single(current_agent, action)
    
    def _step(self, actions: Dict[str, str]) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        '''
        Parallel step function. As the lack of "parallel" suggests,
        this is ideally the default mode of operation for multi-agent environments.
        '''
        raise NotImplementedError

    def _step_single(self, current_agent: str, action: str) -> None:
        '''
        Sequential step function, the more general mode of 
        operation for multi-agent environments.

        Provides testing convenience and expressivity for games 
        whose players cannot make moves in parallel (i.e. Chess)

        This is because the environment steps can be broken up to be
        on a per-player basis, allowing for more complex interactions
        between individual players and the environment.
        '''
        raise NotImplementedError

    def _step_global_dynamics(self, ret = None) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        '''
        After all players have finished making their moves,
        handle all global environmental dynamics before 
        starting a new "turn".
        '''
        self._reset_agent_iter()
        if ret is None:
            ret = self.get_all_states()
        # Flush the cumulative rewards after each step.
        for agent in self.agents:
            self.rewards[agent] = 0.0
            self._cumulative_rewards[agent] = 0.0
            # Update the active mask
            if self.terminations[agent] or self.truncations[agent]:
                self.active_mask[agent] = False
        return ret
    
    def _validate_actions(self, actions: Dict[str, str]):
        for agent in self.agents:
            if agent not in self.terminations or self.terminations[agent]:
                continue
            if agent not in self.truncations or self.truncations[agent]:
                continue
            if agent not in actions:
                raise ValueError(f"Missing action for active agent {agent}")

        for agent in actions:
            if not self.active_mask[agent]:
                raise ValueError(f"Agent {agent} provided action but is not active")

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        if seed is not None:
            self._np_random = self._make_np_random(seed)

        self.active_mask = {agent: True for agent in self.agents}

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.shared_memory = []
        self.global_context = ""

        self._reset_agent_iter()

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos
    
    def observe(self, agent: str) -> str:
        raise NotImplementedError

    def get_state(self, agent: str) -> Tuple[str, float, bool, bool, dict]:
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} not in environment")

        return (
            self.observe(agent),
            self._cumulative_rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def get_all_states(self) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        # Get a tuple for all states
        all_states = [self.get_state(agent) for agent in self.agents]
        # Translate into "characteristic-first" instead of "agent-first"
        observations, rewards, terminations, truncations, infos = zip(*all_states)
        return (
            dict(zip(self.agents, observations)),
            dict(zip(self.agents, rewards)),
            dict(zip(self.agents, terminations)),
            dict(zip(self.agents, truncations)),
            dict(zip(self.agents, infos)),
        )

    def update_reward(self, agent: str, reward: float):
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} not in environment")
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

    def send_message(self, from_agent: str, to_agent: str, message: str):
        if from_agent not in self.agents:
            raise ValueError(f"Sender {from_agent} not in environment")
        if to_agent not in self.agents:
            raise ValueError(f"Receiver {to_agent} not in environment")

        self.shared_memory.append(
            {"from": from_agent, "to": to_agent, "message": message}
        )

    def broadcast_message(self, from_agent: str, message: str):
        if from_agent not in self.agents:
            raise ValueError(f"Sender {from_agent} not in environment")

        for agent in self.agents:
            if agent != from_agent:
                self.shared_memory.append(
                    {"from": from_agent, "to": agent, "message": message}
                )

class AgentIterator:
    '''
    Iterator to help Sequential environments iterate over agents.
    '''
    
    def __init__(self, agents: List[str], active_mask):
        self._agents = agents.copy()
        self.active_mask = active_mask
        self.current_agent = None
        self._current_idx = 0

    def __next__(self) -> Optional[str]:
        if not any(self.active_mask):
            return None
        
        if self.is_end():
            raise StopIteration

        self.current_agent = self._agents[self._current_idx]

        # Point to the next agent
        self._current_idx += 1
        while not self.is_end() and not self.active_mask[self._agents[self._current_idx]]:
            self._current_idx += 1

        return self.current_agent
    
    def is_alive(self, agent):
        try:
            return self.active_mask[agent]
        except KeyError:
            raise KeyError("Agent does not exist in this iterator.")

    def is_end(self):
        return self._current_idx >= len(self._agents)

    def __iter__(self):
        return self