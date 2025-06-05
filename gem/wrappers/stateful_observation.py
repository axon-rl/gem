"""A collection of stateful observation wrappers."""

from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import ObservationWrapper
from gem.envs.multi_turn import MultiTurnEnv


def maybe_add_new_line(text: str):
    if text and not text.endswith("\n"):
        return text + "\n"
    return text


class ConcatenatedObservation(ObservationWrapper):
    def __init__(self, env: MultiTurnEnv, max_history_length: Optional[int] = None):
        super().__init__(env)
        assert isinstance(
            env, MultiTurnEnv
        ), "ConcatenatedObservation wrapper only supports MultiTurnEnv"
        self.env = env
        self.max_history_length = max_history_length
        self.obs_queue = deque(maxlen=max_history_length)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        self.obs_queue.clear()
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.append(obs)
        return self._get_wrapped_obs(), reward, terminated, truncated, info

    def _get_wrapped_obs(self) -> str:
        wrapped_obs = self.env.get_task_prefix()
        for obs in self.obs_queue:
            wrapped_obs += maybe_add_new_line(obs)
        wrapped_obs += self.env.get_task_suffix()
        return wrapped_obs


class ChatTemplatedObservation(ConcatenatedObservation):
    def __init__(
        self, env: MultiTurnEnv, tokenizer, max_history_length: Optional[int] = None
    ):
        super().__init__(env, max_history_length)
        self.tokenizer = tokenizer
        self.act_queue = deque(maxlen=max_history_length)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        obs, info = super().reset(seed=seed)
        wrapped_obs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": obs}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return wrapped_obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.append(obs)
        self.act_queue.append(action)

        obs_list = [self.env.get_task_prefix()] + list(self.obs_queue)[:-1]
        act_list = list(self.act_queue)

        chat_messages = []
        for o, a in zip(obs_list, act_list):
            chat_messages.append({"role": "user", "content": o})
            chat_messages.append({"role": "assistant", "content": a})
        last_o = maybe_add_new_line(self.obs_queue[-1])
        last_o += self.env.get_task_suffix()
        chat_messages.append({"role": "user", "content": last_o})

        wrapped_obs = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return wrapped_obs, reward, terminated, truncated, info
