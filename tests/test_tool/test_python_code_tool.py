#!/usr/bin/env python

# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
import json
import requests
import fire
import logging
import sys
import os
import random

from transformers import AutoTokenizer

import gem
from gem.envs.multi_turn import MultiTurnEnv
from gem.utils.debug import run_and_print_episode
from gem.wrappers.stateful_observation import ChatTemplatedObservation, ConcatenatedObservation

# Add parent directory to path to import PistonTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gem.envs.textarena.guess_the_number import GuessTheNumberEnv
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.tools.python_code_tool import PythonCodeTool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_ACTIONS = [
    """<python>print('Hello from Python!')</python> ...""",
    """Dummy action""",
    """<python>import sys\n\nprint('Hello from Python!')\nprint(f'Arguments: {sys.argv[1:]}')\nfor i in range(5):\n    print(f'Number {i}')</python> ...""",
    """```<python>\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ...""",
    """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ...""",
    """```<python>prnit('Hello from Python!')</python> ...""",
]

def test_single_action():
    env = GuessTheNumberEnv()
    tool = PythonCodeTool()
    env = ToolEnvWrapper(env, tools=[tool])
    obs, info = env.reset()
    for i, test_action in enumerate(TEST_ACTIONS):
        print(f"------ Test {i} ------")
        print(f"Action: {test_action!r}")
        obs, reward, terminated, truncated, info = env.step(test_action)
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}\n")

def test_episode():
    env: MultiTurnEnv = gem.make("ta:GuessTheNumber-v0", max_turns=3)
    policy = lambda _: random.choice(TEST_ACTIONS)
    tool = PythonCodeTool()

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 2: CONCATENATED OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = ConcatenatedObservation(wrapped_env)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 3: CHAT TEMPLATE OBSERVATION")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = ChatTemplatedObservation(wrapped_env, tokenizer)
    run_and_print_episode(wrapped_env, policy)


def main():
    """Main entry point for the test script
    Run with:
        python -m tests.test_tool.test_python_code_tool single_action
        python -m tests.test_tool.test_python_code_tool episode
    """
    fire.Fire({
        "single_action": test_single_action,
        "episode": test_episode,
    })

if __name__ == "__main__":
    main()
