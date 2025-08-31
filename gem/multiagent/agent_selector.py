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

"""Agent selection utility for AEC environments."""

from typing import List


class AgentSelector:
    """Utility for managing agent turn order in AEC environments.

    This class handles the selection and cycling of agents in sequential
    environments, maintaining the current agent and providing methods to
    advance through the agent order.
    """

    def __init__(self, agents: List[str]):
        """Initialize the agent selector.

        Args:
            agents: List of agent IDs in turn order.
        """
        self.agents = agents.copy()
        self._current_idx = 0
        self.selected = self.agents[0] if agents else None

    def reinit(self, agents: List[str]) -> None:
        """Reinitialize with a new list of agents.

        Args:
            agents: New list of agent IDs in turn order.
        """
        self.agents = agents.copy()
        self._current_idx = 0
        self.selected = self.agents[0] if agents else None

    def reset(self) -> str:
        """Reset to the first agent.

        Returns:
            The first agent ID.
        """
        self._current_idx = 0
        self.selected = self.agents[0] if self.agents else None
        return self.selected

    def next(self) -> str:
        """Move to the next agent in order.

        Returns:
            The next agent ID.
        """
        if not self.agents:
            self.selected = None
            return None

        self._current_idx = (self._current_idx + 1) % len(self.agents)
        self.selected = self.agents[self._current_idx]
        return self.selected

    def is_last(self) -> bool:
        """Check if the current agent is the last in the cycle.

        Returns:
            True if current agent is last, False otherwise.
        """
        if not self.agents:
            return True
        return self._current_idx == len(self.agents) - 1

    def is_first(self) -> bool:
        """Check if the current agent is the first in the cycle.

        Returns:
            True if current agent is first, False otherwise.
        """
        return self._current_idx == 0

    def agent_order(self) -> List[str]:
        """Get the current agent order.

        Returns:
            List of agent IDs in turn order.
        """
        return self.agents.copy()

    def remove_agent(self, agent: str) -> None:
        """Remove an agent from the selection order.

        Args:
            agent: The agent ID to remove.
        """
        if agent not in self.agents:
            return

        # Get index of agent to remove
        agent_idx = self.agents.index(agent)

        # If we're removing the currently selected agent, need to handle carefully
        if agent_idx == self._current_idx:
            # If it's the last agent, wrap to start
            if self._current_idx >= len(self.agents) - 1:
                self._current_idx = 0
            # Otherwise current_idx stays the same (next agent moves into this position)
        elif agent_idx < self._current_idx:
            # If we remove an agent before current, adjust index
            self._current_idx -= 1

        # Remove the agent
        self.agents.remove(agent)

        # Update selected
        if self.agents:
            self._current_idx = min(self._current_idx, len(self.agents) - 1)
            self.selected = self.agents[self._current_idx]
        else:
            self.selected = None
            self._current_idx = 0

    def __len__(self) -> int:
        """Get the number of agents.

        Returns:
            Number of agents in the selector.
        """
        return len(self.agents)
