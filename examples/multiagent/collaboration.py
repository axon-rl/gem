#!/usr/bin/env python3
# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple

from gem import make, register
from gem.multiagent.multi_agent_env import MultiAgentEnv
from gem.multiagent.parallel_env import ParallelEnv


class CollaborationEnv(ParallelEnv):
    def __init__(self, max_rounds: int = 3):
        super().__init__()

        self.possible_agents = ["researcher", "analyst", "reviewer"]
        self.agents = self.possible_agents.copy()

        self.max_rounds = max_rounds
        self.round_count = 0

        self.shared_memory = {}
        self.task = "Solve a complex problem collaboratively"

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

    def step(self, actions: Dict[str, str]) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        self._validate_actions(actions)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            action = actions[agent]
            self.shared_memory[agent] = action

            if "complete" in action.lower():
                rewards[agent] = 1.0
            else:
                rewards[agent] = 0.1

            terminations[agent] = False
            truncations[agent] = False
            infos[agent] = {"round": self.round_count}

            observations[agent] = self._get_observation(agent)

        self.round_count += 1

        if self.round_count >= self.max_rounds or any(
            "complete" in actions[a].lower() for a in self.agents
        ):
            for agent in self.agents:
                terminations[agent] = True

        self.terminations = terminations
        self.truncations = truncations
        self.rewards = rewards
        self.infos = infos

        self._accumulate_rewards()

        return observations, rewards, terminations, truncations, infos

    def _get_observation(self, agent: str) -> str:
        other_agents_info = []
        for other_agent, info in self.shared_memory.items():
            if other_agent != agent:
                other_agents_info.append(f"{other_agent}: {info}")

        if other_agents_info:
            return f"Task: {self.task}\nOther agents' actions:\n" + "\n".join(
                other_agents_info
            )
        else:
            return f"Task: {self.task}\nYou are the first to act this round."

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        MultiAgentEnv.reset(self, seed)

        self.agents = self.possible_agents.copy()
        self.round_count = 0
        self.shared_memory = {}

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        observations = {
            agent: f"Task: {self.task}\nYou are {agent}. Begin collaboration."
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}

        return observations, infos


def main():
    register("Collaboration-v0", entry_point=CollaborationEnv)

    env = make("Collaboration-v0")

    print("=== Multi-Agent Collaboration Example ===")
    print("Demonstrating agents working together on a task\n")

    observations, _ = env.reset()

    print("Initial observations:")
    for agent, obs in observations.items():
        print(f"[{agent}]:\n{obs}\n")

    round_num = 1
    while env.agents:
        print(f"--- Round {round_num} ---")

        actions = {}
        for agent in env.agents:
            if round_num == 1:
                actions[agent] = f"Starting analysis of the problem"
            elif round_num == 2:
                actions[agent] = f"Building on others' work"
            else:
                actions[agent] = f"Task complete - final solution ready"

        for agent, action in actions.items():
            print(f"[{agent}]: {action}")

        observations, rewards, terminations, truncations, _ = env.step(actions)

        print(f"\nRewards: {rewards}")
        print(f"Shared memory: {env.shared_memory}\n")

        if all(terminations.values()) or all(truncations.values()):
            break

        round_num += 1

    print("=== Collaboration Complete ===")
    print(f"Total rounds: {env.round_count}")
    print(f"Final cumulative rewards: {env._cumulative_rewards}")


if __name__ == "__main__":
    main()
