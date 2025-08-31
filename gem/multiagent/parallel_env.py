import abc
from typing import Any, Dict, Optional, Tuple

from gem.multiagent.multi_agent_env import MultiAgentEnv


class ParallelEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__()
        self.metadata = {"is_parallelizable": True}

    @abc.abstractmethod
    def step(self, actions: Dict[str, str]) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        if set(actions.keys()) != set(self.agents):
            missing = set(self.agents) - set(actions.keys())
            extra = set(actions.keys()) - set(self.agents)
            msg = []
            if missing:
                msg.append(f"Missing actions for agents: {missing}")
            if extra:
                msg.append(f"Extra actions for non-active agents: {extra}")
            raise ValueError(". ".join(msg))

        raise NotImplementedError

    @abc.abstractmethod
    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        raise NotImplementedError

    def render(self) -> Optional[Any]:
        return None

    def state(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def _validate_actions(self, actions: Dict[str, str]) -> None:
        action_agents = set(actions.keys())
        active_agents = set(self.agents)

        if action_agents != active_agents:
            missing = active_agents - action_agents
            extra = action_agents - active_agents

            error_parts = []
            if missing:
                error_parts.append(f"Missing actions for agents: {sorted(missing)}")
            if extra:
                error_parts.append(
                    f"Actions provided for non-active agents: {sorted(extra)}"
                )

            raise ValueError(". ".join(error_parts))

    def _remove_dead_agents(self) -> None:
        self.agents = [
            agent
            for agent in self.agents
            if not (
                self.terminations.get(agent, False)
                or self.truncations.get(agent, False)
            )
        ]
