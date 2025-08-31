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

"""Agent Environment Cycle (AEC) API for sequential multi-agent environments."""

import abc
from typing import Any, Dict, Iterator, Optional, Tuple

from gem.multiagent.core import MultiAgentEnv


class AECEnv(MultiAgentEnv):
    """Sequential multi-agent environment following PettingZoo's AEC pattern.

    In AEC environments, agents act sequentially - one agent acts at a time
    in a specified order. This is suitable for turn-based games and scenarios
    where agents must act in sequence.
    """

    def __init__(self):
        super().__init__()
        self.agent_selection: Optional[str] = None
        self._agent_selector = None
        self._cumulative_rewards = {}
        self._last_observation = None
        self._last_info = {}

    @abc.abstractmethod
    def observe(self, agent: str) -> str:
        """Get observation for a specific agent.

        Args:
            agent: The agent ID to get observation for.

        Returns:
            The observation for the specified agent.
        """
        raise NotImplementedError

    def last(
        self, observe: bool = True
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Returns observation, reward, terminated, truncated, info for current agent.

        This method provides the last step's results for the currently selected agent.
        It's the primary method for getting agent-specific information in AEC environments.

        Args:
            observe: Whether to return observation (True) or None (False).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) for current agent.
        """
        agent = self.agent_selection

        if agent is None:
            raise ValueError("No agent selected. Call reset() first.")

        observation = self.observe(agent) if observe else None

        # Get agent-specific values, with defaults for safety
        reward = self._cumulative_rewards.get(agent, 0.0)
        terminated = self.terminations.get(agent, False)
        truncated = self.truncations.get(agent, False)
        info = self.infos.get(agent, {})

        # Reset cumulative reward after returning it
        self._cumulative_rewards[agent] = 0.0

        return observation, reward, terminated, truncated, info

    def agent_iter(self, max_iter: int = 2**63) -> Iterator[str]:
        """Create an iterator over active agents.

        This iterator cycles through agents, yielding each active agent in turn.
        It automatically handles terminated/truncated agents.

        Args:
            max_iter: Maximum number of iterations (default is effectively infinite).

        Yields:
            The next active agent ID.
        """
        return AECIterable(self, max_iter)

    def _was_dead_step(self, action: Optional[Any]) -> bool:
        """Check if this was a dead step (action on terminated agent).

        Args:
            action: The action taken (None for dead agents).

        Returns:
            True if this was a dead step, False otherwise.
        """
        if action is None:
            return True

        agent = self.agent_selection
        if agent is None:
            return False

        return (
            self.terminations.get(agent, False)
            or self.truncations.get(agent, False)
            or agent not in self.agents
        )

    @abc.abstractmethod
    def step(self, action: Optional[str]) -> None:
        """Process action for the current agent and update environment state.

        This method executes the action for the currently selected agent,
        updates the environment state, and advances to the next agent.

        Args:
            action: The action to execute for the current agent.
                   Should be None for terminated/truncated agents.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (initial observation for first agent, info dictionary).
        """
        # Note: Call parent reset but don't raise NotImplementedError here
        # since parent's reset returns None, not the expected tuple
        # Subclasses should:
        # 1. Call MultiAgentEnv.reset(seed) to initialize base state
        # 2. Initialize agent_selection
        # 3. Set up agent_selector if using one
        # 4. Return first observation
        raise NotImplementedError


class AECIterable:
    """Iterator for cycling through agents in AEC environments."""

    def __init__(self, env: AECEnv, max_iter: int):
        self.env = env
        self.max_iter = max_iter
        self.iter_count = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.iter_count >= self.max_iter:
            raise StopIteration

        # Check if all agents are done
        if not self.env.agents:
            raise StopIteration

        # Get current agent
        agent = self.env.agent_selection

        if agent is None:
            raise StopIteration

        # Check if current agent is terminated/truncated
        if self.env.terminations.get(agent, False) or self.env.truncations.get(
            agent, False
        ):
            # If all agents are terminated/truncated, stop
            if all(
                self.env.terminations.get(a, False)
                or self.env.truncations.get(a, False)
                for a in self.env.agents
            ):
                raise StopIteration

        self.iter_count += 1
        return agent
