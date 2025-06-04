"""Core APIs."""

import abc
from typing import Any, Optional, SupportsFloat, Tuple, TypeVar

from gem.utils import seeding

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Env(abc.ABC):

    @abc.abstractmethod
    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs one time step of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): A typical observation is a text string describing the current state.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task).
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied, e.g., timelimit.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """

    def reset(self, seed: Optional[int] = None) -> Tuple[ObsType, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalized policy about the environment.

        Returns:
            observation (ObsType): A typical observation is a text string describing the current state.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
