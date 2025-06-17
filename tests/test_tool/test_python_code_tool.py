#!/usr/bin/env python

# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
import logging
import random
from functools import partial
from typing import List
import random

import fire
from transformers import AutoTokenizer

import gem
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_ACTIONS = [
    """<python>print('Hello from Python!')</python> ...""",
    """Dummy action""",
    """<python>import sys\n\nprint('Hello from Python!')\nprint(f'Arguments: {sys.argv[1:]}')\nfor i in range(5):\n    print(f'Number {i}')</python> ...""",
    """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ...""",
    """```<python>prnit('Hello from Python!')</python> ...""",
    "\\boxed{30}"
]

SLEEP_ACTION = """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ..."""


def test_single_action(env_name: str = "ta:GuessTheNumber-v0"):
    env = gem.make(env_name, max_turns=3)
    tool = PythonCodeTool(timeout=2)
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
    env = gem.make(env_name, max_turns=3)
    policy = lambda _: random.choice(TEST_ACTIONS)
    tool = PythonCodeTool(timeout=2)

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 2: CHAT TEMPLATE OBSERVATION")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "BATCH EPISODE 1: SYNC VECTORIZED ENV")
    num_envs = 4
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=False,
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        # policy=lambda _: [random.choice(TEST_ACTIONS) for _ in range(num_envs)],
        policy=lambda _: [SLEEP_ACTION for _ in range(num_envs)],
        ignore_done=True,
        max_steps=5,
    )

    print("\n" * 5, "BATCH EPISODE 2: ASYNC VECTORIZED ENV")
    num_envs = 4
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=True,
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        # policy=lambda _: [random.choice(TEST_ACTIONS) for _ in range(num_envs)],
        policy=lambda _: [SLEEP_ACTION for _ in range(num_envs)],
        ignore_done=True,
        max_steps=2,
    )


def test_llm_episode(
    env_name: str = "ta:GuessTheNumber-v0", model_name: str = "Qwen/Qwen3-0.6B-Base"
):
    """Test episode with LLM observation and Python code tool."""
    from vllm import LLM, SamplingParams

    env = gem.make(env_name, max_turns=3)
    llm = LLM(
        model=model_name,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        max_tokens=100,
        top_p=0.95,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def policy(obs):
        assert isinstance(
            obs, str
        ), f"Observation should be a string but is {type(obs)}."
        response = llm.generate(
            [obs],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        # print(f"LLM OBSERVATION: {obs!r}")
        # print(f"LLM RESPONSE: {response}")
        action = response[0].outputs[0].text
        # print(f"LLM ACTION: {action!r}")
        return action

    def batch_policy(obss):
        assert isinstance(
            obss, List
        ), f"Observation should be a string but is {type(obss)}."
        response = llm.generate(
            obss,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        print(f"LLM OBSERVATION: {obss!r}")
        print(f"LLM RESPONSE: {response}")
        actions = [r.outputs[0].text for r in response]
        print(f"LLM ACTION: {actions!r}")
        return actions

    tool = PythonCodeTool(timeout=2)

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 2: CHAT TEMPLATE OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "BATCH EPISODE 1: SYNC VECTORIZED ENV")
    num_envs = 4
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=False,
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        policy=batch_policy,
        ignore_done=True,
        max_steps=2,
    )

    print("\n" * 5, "BATCH EPISODE 2: ASYNC VECTORIZED ENV")
    num_envs = 4
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=True,
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        policy=batch_policy,
        ignore_done=True,
        max_steps=2,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "single_action": test_single_action,
            "episode": test_episode,
            "llm_episode": test_llm_episode,
        }
    )
    print(f"\n\nAll tests run.\n\n")

    """Run with:
    python -m tests.test_tool.test_python_code_tool single_action --env_name ta:GuessTheNumber-v0
    python -m tests.test_tool.test_python_code_tool episode --env_name ta:GuessTheNumber-v0
    python -m tests.test_tool.test_python_code_tool llm_episode --env_name ta:GuessTheNumber-v0 --model_name Qwen/Qwen3-0.6B-Base
    python -m tests.test_tool.test_python_code_tool episode --env_name eval:MATH500
    python -m tests.test_tool.test_python_code_tool llm_episode --env_name eval:MATH500 --model_name Qwen/Qwen3-0.6B-Base
    """
