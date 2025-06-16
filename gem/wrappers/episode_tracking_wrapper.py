"""An environment wrapper for tracking episode statistics.
Note: Must be used as the outermost wrapper.
"""

from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env, EnvWrapper, ActType, WrapperObsType

class EpisodeTrackingWrapper(EnvWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.step_counter = 0
        self.cumulative_rewards = 0.0

    def step(
        self, action: ActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1
        self.cumulative_rewards += reward
        info["step_counter"] = self.step_counter
        info["cumulative_rewards"] = self.cumulative_rewards
        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        prev_ep_step_counter = self.step_counter
        prev_ep_step_cumulative_rewards = self.cumulative_rewards
        obs, info = self.env.reset(seed)
        self.step_counter = 0
        self.cumulative_rewards = 0.0
        info["step_counter"] = self.step_counter
        info["prev_ep_step_counter"] = prev_ep_step_counter
        info["cumulative_rewards"] = self.cumulative_rewards
        info["prev_ep_cumulative_rewards"] = prev_ep_step_cumulative_rewards
        return obs, info