"""Single-turn environment based on reasoning_gym.

https://github.com/open-thought/reasoning-gym.
"""

from gem import Env


class ReasoningGymEnv(Env):

    def __init__(self):
        super().__init__()

    def step(self, action):
        return super().step(action)

    def reset(self):
        return super().reset()
