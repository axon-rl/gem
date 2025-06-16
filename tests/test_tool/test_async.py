#!/usr/bin/env python

# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
import asyncio
import logging
from typing import List
import random

import fire
from transformers import AutoTokenizer

import gem
from gem.utils.debug import run_and_print_episode_async
from gem.wrappers.wrapper_factory import get_wrapper_fns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_ACTIONS = [
    # """<python>print('Hello from Python!')</python> ...""",
    """Dummy action""",
    # """<python>import sys\n\nprint('Hello from Python!')\nprint(f'Arguments: {sys.argv[1:]}')\nfor i in range(5):\n    print(f'Number {i}')</python> ...""",
    # """```<python>\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ...""",
    """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ...""",
    # """```<python>prnit('Hello from Python!')</python> ...""",
    "\\boxed{30}",
]

SLEEP_ACTION = """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ..."""


async def test_episode(env_name: str = "ta:GuessTheNumber-v0"):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapper_fns = get_wrapper_fns("python_tool,concat_chat", tokenizer=tokenizer)

    print("\n" * 5, "BATCH EPISODE: ASYNC VECTORIZED ENV")
    num_envs = 3
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=wrapper_fns,
        max_turns=3,
        async_mode=True,
    )
    await run_and_print_episode_async(
        ta_vec_env,
        # policy=lambda _: [SLEEP_ACTION for _ in range(num_envs)],
        policy=lambda _: [random.choice(TEST_ACTIONS) for _ in range(num_envs)],
        ignore_done=True,
        max_steps=5,
    )


async def test_llm_episode(
    env_name: str = "ta:GuessTheNumber-v0", model_name: str = "Qwen/Qwen3-0.6B-Base"
):
    """Test episode with LLM observation and Python code tool."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        max_tokens=100,
        top_p=0.95,
    )

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

    wrapper_fns = get_wrapper_fns("concat_chat,python_tool", tokenizer=llm.get_tokenizer())
    print("\n" * 5, "BATCH EPISODE: ASYNC VECTORIZED ENV")
    num_envs = 3
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=wrapper_fns,
        max_turns=3,
        async_mode=True,
    )
    await run_and_print_episode_async(
        ta_vec_env,
        policy=batch_policy,
        ignore_done=True,
        max_steps=5,
    )


if __name__ == "__main__":

    def run_test_episode(*args, **kwargs):
        asyncio.run(test_episode(*args, **kwargs))

    def run_test_llm_episode(*args, **kwargs):
        asyncio.run(test_llm_episode(*args, **kwargs))

    fire.Fire(
        {
            "episode": run_test_episode,
            "llm_episode": run_test_llm_episode,
        }
    )
    print(f"\n\nAll tests run.\n\n")

    """Run with:
    python -m tests.test_tool.test_async episode --env_name ta:GuessTheNumber-v0
    python -m tests.test_tool.test_async llm_episode --env_name ta:GuessTheNumber-v0 --model_name Qwen/Qwen3-0.6B-Base
    python -m tests.test_tool.test_async episode --env_name --env_name eval:MATH500
    python -m tests.test_tool.test_async llm_episode --env_name eval:MATH500 --model_name Qwen/Qwen3-0.6B-Base
    """
