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

"""Core multi-agent environment classes for GEM."""

import abc
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, TypeVar

from gem.core import Env
from gem.utils import seeding

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
AgentID = TypeVar("AgentID", bound=str)


class MultiAgentEnv(Env):
    """Base class for multi-agent environments in GEM.

    This class extends GEM's base Env class to support multiple agents.
    It provides the foundation for both sequential (AEC) and parallel
    multi-agent environments.
    """

    def __init__(self):
        super().__init__()
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
        """List of currently active agent IDs.

        Returns:
            List of agent IDs that are currently active in the environment.
        """
        return self._agents

    @agents.setter
    def agents(self, value: List[str]):
        """Set the list of active agents."""
        self._agents = value

    @property
    def possible_agents(self) -> List[str]:
        """List of all possible agents that could be in the environment.

        Returns:
            List of all agent IDs that could potentially exist in the environment.
        """
        return self._possible_agents

    @possible_agents.setter
    def possible_agents(self, value: List[str]):
        """Set the list of possible agents."""
        self._possible_agents = value

    @property
    def num_agents(self) -> int:
        """Number of currently active agents."""
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Maximum number of agents possible in the environment."""
        return len(self.possible_agents)

    def observation_space(self, agent: str) -> Any:
        """Returns observation space for a specific agent.

        Args:
            agent: The agent ID to get observation space for.

        Returns:
            The observation space for the specified agent.
        """
        return self.observation_spaces.get(agent)

    def action_space(self, agent: str) -> Any:
        """Returns action space for a specific agent.

        Args:
            agent: The agent ID to get action space for.

        Returns:
            The action space for the specified agent.
        """
        return self.action_spaces.get(agent)

    def _accumulate_rewards(self) -> None:
        """Accumulate rewards for all agents."""
        for agent in self.agents:
            if agent in self.rewards:
                if agent not in self._cumulative_rewards:
                    self._cumulative_rewards[agent] = 0.0
                self._cumulative_rewards[agent] += self.rewards[agent]

    def _clear_rewards(self) -> None:
        """Clear per-step rewards."""
        self.rewards = {agent: 0.0 for agent in self.agents}

    def _was_dead_step(self, action: Optional[ActType]) -> bool:
        """Check if this was a dead step (action on terminated agent).

        Args:
            action: The action taken (None for dead agents).

        Returns:
            True if this was a dead step, False otherwise.
        """
        return action is None

    @abc.abstractmethod
    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one timestep of the environment's dynamics.

        This method must be implemented by subclasses to define how
        the environment processes actions and updates state.

        Args:
            action: Action(s) to be executed.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the environment to initial state.

        This method resets the base multi-agent state. Subclasses should
        override this to add their specific reset logic and return the
        appropriate values.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            seeding.set_seed(seed)

        # Reset agent lists
        self.agents = self.possible_agents.copy()

        # Reset state dictionaries
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

    def close(self) -> None:
        """Close the environment and clean up resources."""

    def render(self) -> Optional[Any]:
        """Render the environment.

        Returns:
            Rendered output depending on render mode.
        """
        return None

    def state(self) -> Any:
        """Returns the global state of the environment.

        This is useful for centralized training methods that require
        a global view of the environment state.

        Returns:
            Global state of the environment.
        """
        raise NotImplementedError(
            "state() method not implemented. "
            "Override this method if global state is needed."
        )
