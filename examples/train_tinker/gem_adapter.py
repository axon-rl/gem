from __future__ import annotations

import json
import gem
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Sequence

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.completers import StopCondition


class GemTinkerEnv(Env):
    def __init__(
        self,
        env_gem: gem.Env,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        self.env_gem = env_gem
        self.renderer = renderer
        self.convo: list[renderers.Message] = list(convo_prefix or [])

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        obs, _ = self.env_gem.reset()
        self.convo.append({"role": "user", "content": obs})
        return self.renderer.build_generation_prompt(self.convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        text = message.get("content", "") if parse_success else ""
        next_obs, reward, terminated, truncated, info = self.env_gem.step(text)
        reward = float(reward)

        metrics: Metrics = {}
        for k, v in (info or {}).items():
            if isinstance(v, (int, float)):
                metrics[k] = v

        done = terminated or truncated
        if done:
            next_ob = tinker.ModelInput.empty()
            next_stop = self.stop_condition
        else:
            self.convo.append({"role": "assistant", "content": text})
            self.convo.append({"role": "user", "content": next_obs})
            next_ob = self.renderer.build_generation_prompt(self.convo)
            next_stop = self.stop_condition

        return StepResult(
            reward=reward,
            episode_done=done,
            next_observation=next_ob,
            next_stop_condition=next_stop,
            metrics=metrics,
        )


@dataclass(frozen=True)
class GemEnvGroupBuilder(EnvGroupBuilder):
    pool: list[gem.Env]
    renderer: renderers.Renderer
    group_size: int
    groups_per_batch: int
    env_id: str
    convo_prefix: list[renderers.Message] | None = None
    group_index: int = -1  # which env in the pool to use for this

    async def make_envs(self) -> Sequence[Env]:
        # duplicate the env for the group size
        assert (
            0 <= self.group_index < len(self.pool) // self.group_size
        ), "group_index must be within the range of the pool size"
        assert hasattr(
            self.pool[0], "get_state"
        ), "env must support get_state() to run in GemEnvGroupBuilder"

        env_0 = self.pool[self.group_index]
        env_0.reset()
        envs = [GemTinkerEnv(env_0, self.renderer, self.convo_prefix)]
        for i in range(1, self.group_size):
            env_i = self.pool[self.groups_per_batch * i + self.group_index]
            env_state = deepcopy(env_0.get_state())
            env_i.set_state(env_state)
            envs.append(GemTinkerEnv(env_i, self.renderer, self.convo_prefix))
        return envs

    def logging_tags(self) -> list[str]:
        return self.env_id.split(":")


class GemDataset(RLDataset):
    def __init__(
        self, builder_config: dict[str, Any], groups_per_batch: int, n_batches: int
    ):
        self.builder_config = builder_config
        self.groups_per_batch = groups_per_batch
        self.n_batches = n_batches

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        return [
            GemEnvGroupBuilder(group_index=i, **self.builder_config)
            for i in range(self.groups_per_batch)
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class GemDatasetBuilder(RLDatasetBuilder):
    env_id: str
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    groups_per_batch: int
    n_batches: int = 100
    env_kwargs_json: str | None = None
    convo_prefix: list[renderers.Message] | None = None
    gem_path: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        env_kwargs = json.loads(self.env_kwargs_json) if self.env_kwargs_json else {}
        pool = [
            gem.make(self.env_id, **env_kwargs)
            for _ in range(self.groups_per_batch * self.group_size)
        ]
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        builder_config = {
            "pool": pool,
            "renderer": renderer,
            "group_size": self.group_size,
            "groups_per_batch": self.groups_per_batch,
            "env_id": self.env_id,
            "convo_prefix": self.convo_prefix,
        }
        return GemDataset(builder_config, self.groups_per_batch, self.n_batches), None
