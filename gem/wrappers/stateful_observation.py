"""A collection of stateful observation wrappers."""

from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env, ObservationWrapper


def maybe_add_new_line(text: str):
    if text and not text.endswith("\n"):
        return text + "\n"
    return text


class ConcatenatedObservation(ObservationWrapper):
    def __init__(self, env: Env, max_history_length: Optional[int] = None):
        super().__init__(env)
        self.env = env
        self.max_history_length = max_history_length
        self.obs_queue = deque(
            maxlen=max_history_length + 1 if max_history_length else None
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        self.obs_queue.clear()
        obs, info = self.env.reset(seed=seed)
        self.obs_queue.append(obs)
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.append(next_obs)
        return self.observation(), reward, terminated, truncated, info

    def observation(self) -> str:
        wrapped_obs = ""
        for obs in self.obs_queue:
            wrapped_obs += maybe_add_new_line(obs)
        return wrapped_obs


class ChatTemplatedObservation(ObservationWrapper):
    def __init__(self, env: Env, tokenizer, max_history_length: Optional[int] = None):
        super().__init__(env)
        self.tokenizer = tokenizer
        self.obs_queue = deque(
            maxlen=max_history_length + 1 if max_history_length else None
        )
        self.act_queue = deque(maxlen=max_history_length)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        self.act_queue.clear()
        self.obs_queue.clear()
        obs, info = self.env.reset(seed=seed)
        self.obs_queue.append(obs)
        return self.observation(), info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.act_queue.append(action)
        self.obs_queue.append(next_obs)
        return self.observation(), reward, terminated, truncated, info

    def observation(self):
        assert len(self.act_queue) == len(self.obs_queue) - 1, (
            f"Action queue should be one shorter than observation queue, but got: "
            f"{len(self.obs_queue)=}, {len(self.act_queue)=}."
            f"{self.obs_queue=}, {self.act_queue=}"
        )

        obs_list = list(self.obs_queue)[:-1]
        act_list = list(self.act_queue)

        chat_messages = []
        for o, a in zip(obs_list, act_list):
            chat_messages.append({"role": "user", "content": o})
            chat_messages.append({"role": "assistant", "content": a})
        next_obs = self.obs_queue[-1]
        chat_messages.append({"role": "user", "content": next_obs})

        wrapped_obs = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return wrapped_obs
