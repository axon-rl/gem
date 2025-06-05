"""Base environment for TextArena."""

import abc

from gem import Env


class MultiTurnEnv(Env):

    def sample_random_action(self) -> str:
        """Samples a random action given the current state.

        This is typically used to construct a random agent.

        Returns:
            str: Action string.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_task_prefix(self) -> str:
        """Returns the description about the multi-turn task as prefix.

        Returns:
            str: Task description in text.
        """

    @abc.abstractmethod
    def get_task_suffix(self) -> str:
        """Returns the instruction for the agent as suffix.

        Returns:
            str: Agent instruction in text.
        """
