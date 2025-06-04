"""Base environment for TextArena."""

import abc

from gem import Env


class TextArenaEnv(Env):
    @abc.abstractmethod
    def sample_random_action(self) -> str:
        """Sample a random action given the current state.

        Returns:
            str: Action string.
        """
