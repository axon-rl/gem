import abc
from typing import Any, Dict, Iterator, Optional, Tuple

from gem.multiagent.multi_agent_env import MultiAgentEnv


class AECEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__()
        self.agent_selection: Optional[str] = None
        self._agent_selector = None
        self._cumulative_rewards = {}
        self._last_observation = None
        self._last_info = {}

    @abc.abstractmethod
    def observe(self, agent: str) -> str:
        raise NotImplementedError

    def last(
        self, observe: bool = True
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        agent = self.agent_selection

        if agent is None:
            raise ValueError("No agent selected. Call reset() first.")

        observation = self.observe(agent) if observe else None

        reward = self._cumulative_rewards.get(agent, 0.0)
        terminated = self.terminations.get(agent, False)
        truncated = self.truncations.get(agent, False)
        info = self.infos.get(agent, {})

        self._cumulative_rewards[agent] = 0.0

        return observation, reward, terminated, truncated, info

    def agent_iter(self, max_iter: int = 2**63) -> Iterator[str]:
        return AECIterable(self, max_iter)

    def _was_dead_step(self, action: Optional[Any]) -> bool:
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
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class AECIterable:
    def __init__(self, env: AECEnv, max_iter: int):
        self.env = env
        self.max_iter = max_iter
        self.iter_count = 0
        self._agent_index = 0
        self._agents_snapshot = env.agents.copy() if env.agents else []

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.iter_count >= self.max_iter:
            raise StopIteration

        if not self._agents_snapshot:
            raise StopIteration

        if all(
            self.env.terminations.get(a, False)
            or self.env.truncations.get(a, False)
            for a in self._agents_snapshot
        ):
            raise StopIteration

        agent = self._agents_snapshot[self._agent_index % len(self._agents_snapshot)]
        
        self._agent_index = (self._agent_index + 1) % len(self._agents_snapshot)
        self.iter_count += 1
        
        return agent