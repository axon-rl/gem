#!/usr/bin/env python3

import os
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from gem.multiagent import AgentSelector, MultiAgentEnv


class SimpleTestEnv(MultiAgentEnv):

    def __init__(self, mode: str = "sequential"):
        super().__init__()

        self.possible_agents = ["agent_0", "agent_1", "agent_2"]
        self.agent_selector = AgentSelector(self.possible_agents, mode=mode)
        self.step_count = 0
        self.max_steps = 10

    def observe(self, agent: str) -> str:
        return f"Step {self.step_count}: Observation for {agent}"

    def _process_actions(self, actions: Dict[str, str]) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        observations = {}
        rewards = {}

        for agent, action in actions.items():
            observations[agent] = self.observe(agent)
            rewards[agent] = 1.0 if "good" in action else 0.0

            # Terminate if action is "exit"
            if action == "exit":
                self.terminations[agent] = True

        self.step_count += 1

        # Truncate all agents if max steps reached
        if self.step_count >= self.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True

        return observations, rewards, self.terminations, self.truncations, self.infos


def test_sequential_mode():
    print("=" * 50)
    print("Testing Sequential Mode")
    print("=" * 50)

    env = SimpleTestEnv(mode="sequential")
    obs, info = env.reset()

    print(f"Initial observations: {obs}")
    print(f"Active agents: {env.agent_selector.get_active_agents()}")

    # Test sequential stepping
    for i in range(5):
        active = env.agent_selector.get_active_agents()
        print(f"\nStep {i}: Active agent(s) = {active}")

        # Only provide action for active agent
        actions = {agent: f"action_{i}" for agent in active}

        obs, rewards, term, trunc, info = env.step(actions)
        print(f"  Observations: {obs}")
        print(f"  Rewards: {rewards}")

    # Test get_state and get_active_states
    print("\n--- Testing get_state methods ---")
    state = env.get_state("agent_0")
    print(f"State for agent_0: {state}")

    active_states = env.get_active_states()
    print(f"Active states: {active_states}")


def test_parallel_mode():
    print("\n" + "=" * 50)
    print("Testing Parallel Mode")
    print("=" * 50)

    env = SimpleTestEnv(mode="parallel")
    obs, info = env.reset()

    print(f"Initial observations: {obs}")
    print(f"Active agents: {env.agent_selector.get_active_agents()}")

    # Test parallel stepping
    for i in range(3):
        active = env.agent_selector.get_active_agents()
        print(f"\nStep {i}: Active agent(s) = {active}")

        # All agents act simultaneously
        actions = {agent: f"good_action_{i}" for agent in active}

        obs, rewards, term, trunc, info = env.step(actions)
        print(f"  Observations: {obs}")
        print(f"  Rewards: {rewards}")

    # Test termination
    print("\n--- Testing termination ---")
    actions = {"agent_1": "exit", "agent_0": "normal", "agent_2": "normal"}
    obs, rewards, term, trunc, info = env.step(actions)
    print(f"Terminations: {term}")
    print(f"Remaining agents: {env.agents}")


def test_agent_iter():
    print("\n" + "=" * 50)
    print("Testing agent_iter")
    print("=" * 50)

    env = SimpleTestEnv(mode="sequential")
    env.reset()

    print("Iterating through agents:")
    count = 0
    for agent in env.agent_iter(max_iter=10):
        print(f"  Iteration {count}: {agent}")

        # Get state for current agent
        state = env.get_state(agent)
        print(f"    State: obs={state[0][:20]}..., reward={state[1]}")

        # Take action
        env.step({agent: "test_action"})

        count += 1
        if count >= 5:
            break


if __name__ == "__main__":
    test_sequential_mode()
    test_parallel_mode()
    test_agent_iter()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
