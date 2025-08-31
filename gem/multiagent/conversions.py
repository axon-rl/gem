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

"""Conversion utilities between AEC and Parallel environments."""

from typing import Any, Dict, Optional, Tuple

from gem.core import EnvWrapper
from gem.multiagent.aec_env import AECEnv
from gem.multiagent.parallel_env import ParallelEnv


class AECToParallelWrapper(ParallelEnv, EnvWrapper):
    """Wrapper to convert AEC environment to Parallel interface.
    
    This wrapper collects actions from all agents and executes them
    sequentially in the underlying AEC environment, then returns
    results for all agents simultaneously.
    """
    
    def __init__(self, aec_env: AECEnv):
        """Initialize the wrapper.
        
        Args:
            aec_env: The AEC environment to wrap.
        """
        ParallelEnv.__init__(self)
        EnvWrapper.__init__(self, aec_env)
        
        self.aec_env = aec_env
        
        # Copy agent information
        self.possible_agents = aec_env.possible_agents.copy()
        self.agents = aec_env.agents.copy()
        
        # Copy spaces
        self.observation_spaces = aec_env.observation_spaces.copy()
        self.action_spaces = aec_env.action_spaces.copy()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """Reset the environment.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Initial observations and infos for all agents.
        """
        # Reset the AEC environment
        first_obs, first_info = self.aec_env.reset(seed)
        
        # Copy agent lists
        self.agents = self.aec_env.agents.copy()
        
        # Collect observations for all agents
        observations = {}
        infos = {}
        
        # Get observation for first agent
        if self.aec_env.agent_selection:
            observations[self.aec_env.agent_selection] = first_obs
            infos[self.aec_env.agent_selection] = first_info
        
        # Get observations for remaining agents
        for agent in self.agents:
            if agent not in observations:
                observations[agent] = self.aec_env.observe(agent)
                infos[agent] = self.aec_env.infos.get(agent, {})
        
        return observations, infos
    
    def step(
        self, actions: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """Execute actions for all agents.
        
        Args:
            actions: Actions for all active agents.
            
        Returns:
            Observations, rewards, terminations, truncations, and infos for all agents.
        """
        # Validate actions
        self._validate_actions(actions)
        
        # Store initial state
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        # Execute all agents' actions in sequence
        agents_to_step = self.agents.copy()
        
        for agent in agents_to_step:
            if agent in actions:
                # Set this agent as selected
                self.aec_env.agent_selection = agent
                
                # Execute the action
                self.aec_env.step(actions[agent])
                
                # Collect results
                obs, reward, terminated, truncated, info = self.aec_env.last()
                
                observations[agent] = obs
                rewards[agent] = reward
                terminations[agent] = terminated
                truncations[agent] = truncated
                infos[agent] = info
        
        # Update our agent list
        self.agents = [
            agent for agent in self.agents
            if not (terminations.get(agent, False) or truncations.get(agent, False))
        ]
        
        return observations, rewards, terminations, truncations, infos


class ParallelToAECWrapper(AECEnv, EnvWrapper):
    """Wrapper to convert Parallel environment to AEC interface.
    
    This wrapper buffers actions from agents and executes them all
    at once when a cycle is complete, providing sequential access
    to a parallel environment.
    """
    
    def __init__(self, parallel_env: ParallelEnv):
        """Initialize the wrapper.
        
        Args:
            parallel_env: The Parallel environment to wrap.
        """
        AECEnv.__init__(self)
        EnvWrapper.__init__(self, parallel_env)
        
        self.parallel_env = parallel_env
        
        # Copy agent information
        self.possible_agents = parallel_env.possible_agents.copy()
        self.agents = parallel_env.agents.copy()
        
        # Copy spaces
        self.observation_spaces = parallel_env.observation_spaces.copy()
        self.action_spaces = parallel_env.action_spaces.copy()
        
        # Action buffer for collecting actions
        self._action_buffer: Dict[str, str] = {}
        
        # Store last parallel step results
        self._observations: Dict[str, str] = {}
        self._rewards: Dict[str, float] = {}
        self._terminations: Dict[str, bool] = {}
        self._truncations: Dict[str, bool] = {}
        self._infos: Dict[str, Dict] = {}
        
        # Agent selection
        from gem.multiagent.agent_selector import AgentSelector
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.selected
    
    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Initial observation for first agent and info.
        """
        # Reset parallel environment
        observations, infos = self.parallel_env.reset(seed)
        
        # Store results
        self._observations = observations
        self._infos = infos
        
        # Reset agent management
        self.agents = self.parallel_env.agents.copy()
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.selected
        
        # Clear buffers
        self._action_buffer = {}
        self._rewards = {agent: 0.0 for agent in self.agents}
        self._terminations = {agent: False for agent in self.agents}
        self._truncations = {agent: False for agent in self.agents}
        
        # Copy to parent class attributes
        self.rewards = self._rewards.copy()
        self.terminations = self._terminations.copy()
        self.truncations = self._truncations.copy()
        self.infos = self._infos.copy()
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        # Return first observation
        if self.agent_selection:
            return self._observations[self.agent_selection], self._infos[self.agent_selection]
        return "", {}
    
    def observe(self, agent: str) -> str:
        """Get observation for specified agent.
        
        Args:
            agent: Agent ID to get observation for.
            
        Returns:
            Observation for the agent.
        """
        return self._observations.get(agent, "")
    
    def step(self, action: Optional[str]) -> None:
        """Execute action for current agent.
        
        Args:
            action: Action for the current agent (None if dead).
        """
        if self.agent_selection is None:
            return
        
        current_agent = self.agent_selection
        
        # Store action if not dead
        if not self._was_dead_step(action):
            self._action_buffer[current_agent] = action
        
        # Move to next agent
        self._agent_selector.next()
        self.agent_selection = self._agent_selector.selected
        
        # If we've completed a cycle, execute parallel step
        if self._agent_selector.is_first() or not self.agents:
            if self._action_buffer:
                # Execute parallel step with buffered actions
                (
                    self._observations,
                    self._rewards,
                    self._terminations,
                    self._truncations,
                    self._infos
                ) = self.parallel_env.step(self._action_buffer)
                
                # Update parent class attributes
                self.rewards = self._rewards.copy()
                self.terminations = self._terminations.copy()
                self.truncations = self._truncations.copy()
                self.infos = self._infos.copy()
                
                # Accumulate rewards
                for agent, reward in self._rewards.items():
                    if agent not in self._cumulative_rewards:
                        self._cumulative_rewards[agent] = 0.0
                    self._cumulative_rewards[agent] += reward
                
                # Update agent list
                self.agents = [
                    agent for agent in self.agents
                    if not (self._terminations.get(agent, False) or 
                           self._truncations.get(agent, False))
                ]
                
                # Update agent selector if agents changed
                if set(self._agent_selector.agents) != set(self.agents):
                    self._agent_selector.reinit(self.agents)
                    self.agent_selection = self._agent_selector.selected
                
                # Clear action buffer
                self._action_buffer = {}


def aec_to_parallel(aec_env: AECEnv) -> ParallelEnv:
    """Convert an AEC environment to Parallel interface.
    
    Args:
        aec_env: The AEC environment to convert.
        
    Returns:
        A Parallel environment wrapping the AEC environment.
    """
    return AECToParallelWrapper(aec_env)


def parallel_to_aec(parallel_env: ParallelEnv) -> AECEnv:
    """Convert a Parallel environment to AEC interface.
    
    Args:
        parallel_env: The Parallel environment to convert.
        
    Returns:
        An AEC environment wrapping the Parallel environment.
    """
    return ParallelToAECWrapper(parallel_env)