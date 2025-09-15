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

import logging
import random
from typing import Dict, Tuple

import fire

from gem.multiagent import AgentSelector, MultiAgentEnv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleTestEnv(MultiAgentEnv):
    """Simple test environment for multi-agent testing."""

    def __init__(self, mode: str = "sequential", num_agents: int = 3):
        super().__init__()

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
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

            if "good" in action:
                rewards[agent] = 1.0
            elif "bad" in action:
                rewards[agent] = -1.0
            else:
                rewards[agent] = 0.0

            if action == "exit":
                self.terminations[agent] = True

        self.step_count += 1

        if self.step_count >= self.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True

        return observations, rewards, self.terminations, self.truncations, self.infos


def test_sequential_mode():
    """Test sequential mode where agents take turns."""
    logger.info("Testing Sequential Mode")

    env = SimpleTestEnv(mode="sequential")
    obs, info = env.reset()

    logger.info(f"Initial agents: {env.agents}")
    logger.info(f"Active agents: {env.agent_selector.get_active_agents()}")

    for i in range(5):
        active = env.agent_selector.get_active_agents()
        actions = {agent: f"action_{i}" for agent in active}

        obs, rewards, term, trunc, info = env.step(actions)

        logger.info(f"Step {i}: Active={active[0]}, Reward={rewards.get(active[0], 0)}")

    state = env.get_state("agent_0")
    logger.info(f"Agent_0 state: cumulative_reward={state[1]}")

    return env


def test_parallel_mode():
    """Test parallel mode where all agents act simultaneously."""
    logger.info("Testing Parallel Mode")

    env = SimpleTestEnv(mode="parallel")
    obs, info = env.reset()

    logger.info(f"Initial agents: {env.agents}")
    logger.info(f"Active agents: {env.agent_selector.get_active_agents()}")

    for i in range(3):
        active = env.agent_selector.get_active_agents()
        actions = {agent: f"good_action_{i}" for agent in active}

        obs, rewards, term, trunc, info = env.step(actions)

        total_reward = sum(rewards.values())
        logger.info(f"Step {i}: Total reward={total_reward}")

    actions = {"agent_1": "exit", "agent_0": "normal", "agent_2": "normal"}
    obs, rewards, term, trunc, info = env.step(actions)
    logger.info(f"After termination: Remaining agents={env.agents}")

    return env


def test_agent_management():
    """Test dynamic agent management."""
    logger.info("Testing Agent Management")

    env = SimpleTestEnv(mode="sequential")
    env.reset()

    env.add_agent("agent_3")
    logger.info(f"After adding agent_3: {env.agents}")

    env.remove_agent("agent_3")
    logger.info(f"After removing agent_3: {env.agents}")

    env.send_message("agent_0", "agent_1", "Hello!")
    env.broadcast_message("agent_0", "Hello everyone!")
    logger.info(f"Messages sent: {len(env.shared_memory)} total")

    return env


def test_agent_selector():
    """Test AgentSelector functionality."""
    logger.info("Testing AgentSelector")

    agents = ["a1", "a2", "a3"]

    selector = AgentSelector(agents, mode="sequential")
    logger.info(f"Sequential - Initial: {selector.selected}")

    for _ in range(3):
        selector.next()
    logger.info(f"Sequential - After 3 next(): {selector.selected}")

    selector_p = AgentSelector(agents, mode="parallel")
    active = selector_p.get_active_agents()
    logger.info(f"Parallel - Active agents: {active}")

    selector.reinit(["b1", "b2"])
    logger.info(f"After reinit: {selector.selected}")

    return selector


def test_agent_selector_advanced():
    """Test advanced AgentSelector functionality including is_first, is_last, and edge cases."""
    logger.info("Testing Advanced AgentSelector Features")

    agents = ["a1", "a2", "a3", "a4"]
    selector = AgentSelector(agents, mode="sequential")

    logger.info(f"Initial: selected={selector.selected}, is_first={selector.is_first()}, is_last={selector.is_last()}")

    for i in range(len(agents)):
        logger.info(f"Position {i}: selected={selector.selected}, is_first={selector.is_first()}, is_last={selector.is_last()}")
        selector.next()

    assert selector.is_first() == True, "Should be at first agent after full cycle"
    logger.info("Full cycle completed, back to first agent")

    selector.reinit(["x1", "x2"])
    logger.info(f"After reinit with 2 agents: selected={selector.selected}, is_first={selector.is_first()}, is_last={selector.is_last()}")

    selector.next()
    assert selector.is_last() == True, "Should be at last agent with 2-agent list"

    selector.add_agent("x3")
    logger.info(f"After adding x3: agents={selector._agents}, selected={selector.selected}")

    selector.remove_agent("x1")
    logger.info(f"After removing x1: agents={selector._agents}, selected={selector.selected}")

    selector.remove_agent("x2")
    selector.remove_agent("x3")
    logger.info(f"After removing all: agents={selector._agents}, selected={selector.selected}")

    assert selector.selected is None, "Selected should be None when no agents"
    assert selector._agents == [], "Agents list should be empty"

    return selector


def test_get_active_states():
    """Test get_active_states method."""
    logger.info("Testing get_active_states")

    env_seq = SimpleTestEnv(mode="sequential")
    env_seq.reset()

    active_states = env_seq.get_active_states()
    assert len(active_states) == 1, "Sequential mode should have 1 active state"
    assert "agent_0" in active_states, "agent_0 should be active initially"

    obs, reward, term, trunc, info = active_states["agent_0"]
    logger.info(f"Sequential active state: agent=agent_0, obs={obs[:20]}..., reward={reward}")

    env_par = SimpleTestEnv(mode="parallel")
    env_par.reset()

    active_states = env_par.get_active_states()
    assert len(active_states) == 3, "Parallel mode should have 3 active states"

    for agent in ["agent_0", "agent_1", "agent_2"]:
        assert agent in active_states, f"{agent} should be in active states"

    logger.info(f"Parallel active states: {list(active_states.keys())}")

    return env_seq, env_par


def test_observe_method():
    """Test observe method directly."""
    logger.info("Testing observe method")

    env = SimpleTestEnv(mode="sequential")
    env.reset()

    for agent in env.agents:
        obs = env.observe(agent)
        assert isinstance(obs, str), "Observation should be string"
        assert agent in obs, f"Observation should mention {agent}"
        logger.info(f"Observation for {agent}: {obs}")

    env.step({"agent_0": "test_action"})
    obs_after = env.observe("agent_0")
    assert "Step 1" in obs_after, "Observation should show step count"
    logger.info(f"Observation after step: {obs_after}")

    return env


def test_validation():
    """Test action validation in step method."""
    logger.info("Testing Action Validation")

    env = SimpleTestEnv(mode="sequential")
    env.reset()

    try:
        env.step({})
        logger.error("Should have raised ValueError for missing action")
    except ValueError as e:
        logger.info(f"Correctly rejected missing action: {e}")

    try:
        env.step({"agent_1": "action", "agent_2": "action"})
        logger.error("Should have raised ValueError for non-active agent")
    except ValueError as e:
        logger.info(f"Correctly rejected non-active agent action: {e}")

    try:
        env.step({"agent_0": "valid_action"})
        logger.info("Valid action accepted")
    except ValueError:
        logger.error("Should not raise error for valid action")

    return env


def test_dead_agent_removal():
    """Test automatic removal of dead agents."""
    logger.info("Testing Dead Agent Removal")

    env = SimpleTestEnv(mode="parallel")
    env.reset()

    initial_agents = env.agents.copy()
    logger.info(f"Initial agents: {initial_agents}")

    env.step({"agent_0": "normal", "agent_1": "exit", "agent_2": "normal"})
    assert "agent_1" not in env.agents, "agent_1 should be removed after termination"
    logger.info(f"After agent_1 exits: {env.agents}")

    for _ in range(10):
        if env.agents:
            actions = {agent: "action" for agent in env.agent_selector.get_active_agents()}
            env.step(actions)

    assert len(env.agents) == 0, "All agents should be removed after truncation"
    logger.info(f"After truncation: {env.agents}")

    return env


def test_message_errors():
    """Test message sending error conditions."""
    logger.info("Testing Message Error Handling")

    env = SimpleTestEnv(mode="sequential")
    env.reset()

    try:
        env.send_message("agent_0", "agent_999", "Hello")
        logger.error("Should have raised ValueError for non-existent receiver")
    except ValueError as e:
        logger.info(f"Correctly rejected non-existent receiver: {e}")

    try:
        env.send_message("agent_999", "agent_0", "Hello")
        logger.error("Should have raised ValueError for non-existent sender")
    except ValueError as e:
        logger.info(f"Correctly rejected non-existent sender: {e}")

    try:
        env.broadcast_message("agent_999", "Hello all")
        logger.error("Should have raised ValueError for non-existent broadcaster")
    except ValueError as e:
        logger.info(f"Correctly rejected non-existent broadcaster: {e}")

    env.send_message("agent_0", "agent_1", "Valid message")
    env.broadcast_message("agent_0", "Valid broadcast")
    logger.info(f"Valid messages sent: {len(env.shared_memory)} total")

    return env


def test_error_handling():
    """Test error conditions and edge cases."""
    logger.info("Testing Error Handling")

    env = SimpleTestEnv(mode="sequential")
    env.reset()

    try:
        env.step("not_a_dict")
        logger.error("Should have raised ValueError for non-dict action")
    except ValueError as e:
        logger.info(f"Correctly rejected non-dict action: {e}")

    try:
        env.get_state("agent_999")
        logger.error("Should have raised ValueError for non-existent agent")
    except ValueError as e:
        logger.info(f"Correctly rejected non-existent agent: {e}")

    try:
        bad_selector = AgentSelector(["a1"], mode="unknown_mode")
        bad_selector.get_active_agents()
        logger.error("Should have raised ValueError for unknown mode")
    except ValueError as e:
        logger.info(f"Correctly rejected unknown mode: {e}")

    return True


def test_cumulative_rewards():
    """Test cumulative reward tracking."""
    logger.info("Testing Cumulative Rewards")

    env = SimpleTestEnv(mode="sequential")
    env.reset()

    env.step({"agent_0": "good_action"})
    env.step({"agent_1": "bad_action"})
    env.step({"agent_2": "neutral"})
    env.step({"agent_0": "good_action"})

    state_0 = env.get_state("agent_0")
    state_1 = env.get_state("agent_1")
    state_2 = env.get_state("agent_2")

    logger.info(f"Agent_0 cumulative: {state_0[1]} (expected: 2.0)")
    logger.info(f"Agent_1 cumulative: {state_1[1]} (expected: -1.0)")
    logger.info(f"Agent_2 cumulative: {state_2[1]} (expected: 0.0)")

    assert state_0[1] == 2.0, "Agent_0 should have cumulative reward of 2.0"
    assert state_1[1] == -1.0, "Agent_1 should have cumulative reward of -1.0"
    assert state_2[1] == 0.0, "Agent_2 should have cumulative reward of 0.0"

    return env


def test_random_policy(mode: str = "sequential", num_steps: int = 10):
    """Test with random policy."""
    logger.info(f"Testing Random Policy ({mode} mode)")

    env = SimpleTestEnv(mode=mode)
    obs, info = env.reset()

    actions_pool = ["good_action", "bad_action", "neutral", "action_1", "action_2"]

    total_rewards = {agent: 0.0 for agent in env.agents}

    for step in range(num_steps):
        active = env.agent_selector.get_active_agents()

        actions = {agent: random.choice(actions_pool) for agent in active}

        obs, rewards, term, trunc, info = env.step(actions)

        for agent, reward in rewards.items():
            total_rewards[agent] = total_rewards.get(agent, 0) + reward

        if all(term.values()) or all(trunc.values()):
            logger.info(f"Episode ended at step {step}")
            break

    logger.info(f"Final rewards: {total_rewards}")

    return env


def test_all(verbose: bool = False):
    """Run all tests."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("Running Multi-Agent Environment Tests")
    print("=" * 60)

    test_sequential_mode()
    print()

    test_parallel_mode()
    print()

    test_agent_management()
    print()

    test_agent_selector()
    print()

    test_agent_selector_advanced()
    print()

    test_get_active_states()
    print()

    test_observe_method()
    print()

    test_validation()
    print()

    test_dead_agent_removal()
    print()

    test_message_errors()
    print()

    test_error_handling()
    print()

    test_cumulative_rewards()
    print()

    test_random_policy("sequential", num_steps=5)
    print()

    test_random_policy("parallel", num_steps=5)

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    fire.Fire(test_all)

    """Run with:
        python -m tests.test_multiagent.test_multiagent
        python -m tests.test_multiagent.test_multiagent --verbose=True
    """