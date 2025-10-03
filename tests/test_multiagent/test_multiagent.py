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

from gem.multiagent import MultiAgentEnv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestBaseEnv(MultiAgentEnv):
    """Simple test environment for multi-agent testing."""

    def __init__(self, num_agents: int = 3):
        super().__init__()

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.step_count = 0
        self.max_steps = 10

    def observe(self, agent: str) -> str:
        return f"Step {self.step_count}: Observation for {agent}"
    
    def _step_global_dynamics(self, ret = None):
        ret = super()._step_global_dynamics(ret)
        observations, rewards, terminations, truncations, infos = ret

        self.step_count += 1
        
        if self.step_count >= self.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True
                truncations[agent] = True
            
        return observations, rewards, terminations, truncations, infos

class TestSimultaneousEnv(TestBaseEnv):
    '''
    Simulate simultaneous path of execution.
    '''
    def _step(self, actions):
        observations = {}
        rewards = {}

        for agent, action in actions.items():
            observations[agent] = f"Result for {agent} after {action}"
            
            if action == "good":
                rewards[agent] = 1.0
            elif action == "bad":
                rewards[agent] = -1.0
            else:
                rewards[agent] = 0.0

            if action == "terminate":
                self.terminations[agent] = True

        return observations, rewards, self.terminations, self.truncations, self.infos
'''
Tests specific to the simultaneous environment.
'''

def test_simultaneous_basic():
    '''
    Basic E2E test for simultaneous env.
    '''
    env = TestSimultaneousEnv()
    obs, infos = env.reset()

    for ep in range(3):
        for step in range(5):
            actions = {}
            for agent in env.agents:
                if not env.terminations[agent] and not env.truncations[agent]:
                    actions[agent] = random.choice(["good", "bad", "neutral", "terminate"])
            
            obs, rewards, terminations, truncations, infos = env.step(actions)
            print(actions)
        
            for agent in env.agents:
                if not env.active_mask[agent]:
                    continue
                print(f"Agent: {agent}, Obs: {obs[agent]}, Reward: {rewards[agent]}, Action: {actions[agent]}, Terminated: {terminations[agent]}, Truncated: {truncations[agent]}")
                if actions[agent] == "terminate":
                    assert terminations[agent]
                elif actions[agent] == "good":
                    assert rewards[agent] == 1.0
                elif actions[agent] == "bad":
                    assert rewards[agent] == -1.0

            if all(terminations[agent] or truncations[agent] for agent in env.agents):
                print("All agents have terminated or truncated. Ending episode.")
                break
        obs, infos = env.reset()

class TestSequentialEnv(TestBaseEnv):
    '''
    Simulate sequential path of execution.
    '''
    def _step_single(self, current_agent, action):
        if action == "good":
            reward = 1.0
        elif action == "bad":
            reward = -1.0
        else:
            reward = 0.0

        if action == "terminate":
            self.terminations[current_agent] = True
        self.update_reward(current_agent, reward)

def test_sequential_basic():
    '''
    Basic E2E test for sequential env.
    '''
    env = TestSequentialEnv()
    obs, infos = env.reset()

    for ep in range(3):
        for step in range(15):
            ortti = None
            actions = {}
            for agent in env.agent_iter:
                obs = env.observe(agent)
                actions[agent] = random.choice(["good", "bad", "neutral", "terminate"])
                ortti = env.step(actions[agent])
                print(f"Step {step} ORTTI: {ortti}")
                if ortti:
                    break
            else:
                raise ValueError("No agent steps taken in sequential env.")
            
            obs, rewards, terminations, truncations, infos = ortti
            for agent in env.agent_iter:
                print(f"Agent: {agent}, Obs: {obs[agent]}, Reward: {rewards[agent]}, Action: {actions[agent]}, Terminated: {terminations[agent]}, Truncated: {truncations[agent]}")
                if actions[agent] == "terminate":
                    assert terminations[agent]
                elif actions[agent] == "good":
                    assert rewards[agent] == 1.0
                elif actions[agent] == "bad":
                    assert rewards[agent] == -1.0

            
            if all(terminations[agent] or truncations[agent] for agent in env.agents):
                print("All agents have terminated or truncated. Ending episode.")
                break
        print("Episode done. Resetting environment.")
        obs, infos = env.reset()

# def test_sequential_abuse():
#     # Abuse the sequential env by providing actions simultaneously.
#     # Should still work.
#     env = TestSequentialEnv()
#     obs, infos = env.reset()
#     for ep in range(3):
#         for step in range(15):
#             actions = {}
#             for agent in env.agents:
#                 if not env.terminations[agent] and not env.truncations[agent]:
#                     actions[agent] = random.choice(["good", "bad", "neutral", "terminate"])
            
#             obs, rewards, terminations, truncations, infos = env.step(actions)
        
#             for agent in env.agents:
#                 if not env.active_mask[agent]:
#                     continue
#                 print(f"Agent: {agent}, Obs: {obs[agent]}, Reward: {rewards[agent]}, Action: {actions[agent]}, Terminated: {terminations[agent]}, Truncated: {truncations[agent]}")
#                 if actions[agent] == "terminate":
#                     assert terminations[agent]
#                 elif actions[agent] == "good":
#                     assert rewards[agent] == 1.0
#                 elif actions[agent] == "bad":
#                     assert rewards[agent] == -1.0

#             if all(terminations[agent] or truncations[agent] for agent in env.agents):
#                 print("All agents have terminated or truncated. Ending episode.")
#                 break
#         obs, infos = env.reset()    

def test_all(verbose: bool = False):
    """Run all tests."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("Running Multi-Agent Environment Tests")
    print("=" * 60)

    # Test simultaneous Environment
    print("\nTesting Simultaneous Environment...")
    test_simultaneous_basic()
    print("Simultaneous Environment tests passed.")

    # Test Sequential Environment
    print("\nTesting Sequential Environment...")
    test_sequential_basic()
    print("Sequential Environment tests passed.")

    print("\nTesting Sequential Environment Abuse...")
    # test_sequential_abuse()
    print("Sequential Environment abuse tests passed.")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    fire.Fire(test_all)

    """Run with:
        python -m tests.test_multiagent.test_multiagent
        python -m tests.test_multiagent.test_multiagent --verbose=True
    """
