"""An environment wrapper that verifies the complete trajectory.
E.g. it verifies the format and correctness at trajectory level.
"""

from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import ActType, Env, EnvWrapper
from gem.utils.qa_em import is_valid_sequence
from gem.wrappers.observation_wrapper import ObservationWrapper, WrapperObsType


class TrajVerifyWrapper(EnvWrapper):
    def __init__(
        self,
        env: Env,
        tags: list[str] = ["think", "search", "information", "answer"],
        formatted_reward: float = 0.2,
        verbose: bool = False,
    ):
        super().__init__(env)
        self.tags = tags
        self.formatted_reward = formatted_reward
        self.verbose = verbose

        # TODO: make it work even it has other wrappers
        assert isinstance(
            env, ObservationWrapper
        ), "env must be wrapped by ObservationWrapper"

    def step(
        self, action: ActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            return obs, reward, terminated, truncated, info
        else:
            _obs_processed = self._maybe_discard_content_after_answer(obs)
            is_valid_format, message = is_valid_sequence(_obs_processed, self.tags)
            is_correct = info["is_correct"]
            if is_valid_format and is_correct:
                reward_traj = reward
            elif not is_valid_format and is_correct:
                reward_traj = reward - self.formatted_reward
            elif is_valid_format and not is_correct:
                reward_traj = self.formatted_reward
            else:
                reward_traj = reward
            
            if self.verbose:
                print(f"Error message: {message}")
            return obs, reward_traj, terminated, truncated, info

    def _maybe_discard_content_after_answer(self, obs: str) -> str:
        last_idx = obs.rfind("<|im_start|>assistant")

        if obs[last_idx:].count("</answer>") == 0:
            return obs
        else:
            last_idx = obs.rfind("</answer>")
            last_idx += len("</answer>")
            return obs[:last_idx]

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        return self.env.reset(seed)
