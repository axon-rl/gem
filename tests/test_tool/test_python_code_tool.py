#!/usr/bin/env python

# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
import logging
import os
import random
import sys
from functools import partial

import fire
from transformers import AutoTokenizer

import gem
from gem.envs.multi_turn import MultiTurnEnv
from gem.utils.debug import run_and_print_episode
from gem.wrappers.stateful_observation import (ChatTemplatedObservation,
                                               ConcatenatedObservation)

# Add parent directory to path to import PistonTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gem.envs.textarena.guess_the_number import GuessTheNumberEnv
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_ACTIONS = [
    """<python>print('Hello from Python!')</python> ...""",
    """Dummy action""",
    """<python>import sys\n\nprint('Hello from Python!')\nprint(f'Arguments: {sys.argv[1:]}')\nfor i in range(5):\n    print(f'Number {i}')</python> ...""",
    """```<python>\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ...""",
    """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ...""",
    """```<python>prnit('Hello from Python!')</python> ...""",
]


def test_single_action(env_name: str = "ta:GuessTheNumber-v0"):
    env: MultiTurnEnv = gem.make(env_name, max_turns=3)
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


def test_episode(env_name: str = "ta:GuessTheNumber-v0"):
    env: MultiTurnEnv = gem.make(env_name, max_turns=3)
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

    print("\n" * 5, "BATCH EPISODE: VECTORIZED ENV")
    num_envs = 3
    tool_env_wrapper = partial(
        ToolEnvWrapper, tools=[tool], max_tool_uses=3
    )
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, ConcatenatedObservation],
        max_turns=3,
    )

    run_and_print_episode(
        ta_vec_env,
        lambda _: [env.sample_random_action() for _ in range(num_envs)],
        ignore_done=True,
        max_steps=5,
    )


def main():
    """Main entry point for the test script
    Run with:
        python -m tests.test_tool.test_python_code_tool single_action --env_name ta:GuessTheNumber-v0
        python -m tests.test_tool.test_python_code_tool episode --env_name ta:GuessTheNumber-v0
    """
    fire.Fire(
        {
            "single_action": test_single_action,
            "episode": test_episode,
        }
    )


if __name__ == "__main__":
    main()
