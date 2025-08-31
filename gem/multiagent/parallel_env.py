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

"""Parallel API for simultaneous multi-agent environments."""

import abc
from typing import Any, Dict, Optional, Tuple

from gem.multiagent.core import MultiAgentEnv


class ParallelEnv(MultiAgentEnv):
    """Parallel multi-agent environment where agents act simultaneously.
    
    In Parallel environments, all agents receive observations and take
    actions simultaneously. This is suitable for real-time scenarios
    and environments where agents act independently without turns.
    """
    
    def __init__(self):
        super().__init__()
        self.metadata = {"is_parallelizable": True}
    
    @abc.abstractmethod
    def step(
        self, actions: Dict[str, str]
    ) -> Tuple[
        Dict[str, str],      # observations
        Dict[str, float],    # rewards
        Dict[str, bool],     # terminated
        Dict[str, bool],     # truncated
        Dict[str, Dict]      # infos
    ]:
        """Execute actions for all agents simultaneously.
        
        This method processes actions from all active agents at once,
        updates the environment state, and returns results for all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions.
                    Should include entries for all active agents.
        
        Returns:
            Tuple containing:
                - observations: Dict mapping agent IDs to observations
                - rewards: Dict mapping agent IDs to rewards
                - terminated: Dict mapping agent IDs to termination status
                - truncated: Dict mapping agent IDs to truncation status
                - infos: Dict mapping agent IDs to info dictionaries
        """
        # Validate that actions are provided for all active agents
        if set(actions.keys()) != set(self.agents):
            missing = set(self.agents) - set(actions.keys())
            extra = set(actions.keys()) - set(self.agents)
            msg = []
            if missing:
                msg.append(f"Missing actions for agents: {missing}")
            if extra:
                msg.append(f"Extra actions for non-active agents: {extra}")
            raise ValueError(". ".join(msg))
        
        # Subclasses should implement the actual step logic
        raise NotImplementedError
    
    @abc.abstractmethod
    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple containing:
                - observations: Initial observations for all agents
                - infos: Initial info dictionaries for all agents
        """
        super().reset(seed)
        
        # Subclasses should:
        # 1. Initialize environment state
        # 2. Generate initial observations for all agents
        # 3. Return observations and infos
        raise NotImplementedError
    
    def render(self) -> Optional[Any]:
        """Render the environment.
        
        Returns:
            Rendered output depending on render mode.
        """
        return None
    
    def state(self) -> Any:
        """Returns the global state of the environment.
        
        This is useful for centralized training methods like QMIX
        that require a global view of the environment state.
        
        Returns:
            Global state of the environment.
        """
        raise NotImplementedError(
            "state() method not implemented. "
            "Override this method if global state is needed for centralized training."
        )
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        pass
    
    def _validate_actions(self, actions: Dict[str, str]) -> None:
        """Validate that actions are provided for all active agents.
        
        Args:
            actions: Dictionary of actions to validate.
            
        Raises:
            ValueError: If actions are missing for some agents or
                       provided for non-active agents.
        """
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
    
    def _remove_dead_agents(self) -> None:
        """Remove terminated or truncated agents from active agents list."""
        self.agents = [
            agent for agent in self.agents
            if not (self.terminations.get(agent, False) or 
                   self.truncations.get(agent, False))
        ]