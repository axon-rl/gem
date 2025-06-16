#!/usr/bin/env python

# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
import logging
import time
from functools import partial
from pprint import pprint
from typing import List, Optional
import random

import fire
import numpy as np
from vllm import LLM, SamplingParams

import gem
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.wrappers.stateful_observation import ChatTemplatedObservation
from gem.wrappers.episode_tracking_wrapper import EpisodeTrackingWrapper

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
    "\\boxed{30}",
]
from transformers import AutoTokenizer


def collect_episodes(
    env,
    policy,
    num_episodes: int = 10,
    print_episodes: bool = False,
):
    num_envs = env.num_envs
    episode_count = 0
    episode_rewards = []
    episode_accuracies = []
    episode_lengths = []
    episode_tool_uses = []
    obs, _ = env.reset()
    done = False
    step_count = 0
    while True:
        step_count += 1
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        for i in range(num_envs):
            if done[i]:
                episode_count += 1
                episode_rewards.append(info[i]["prev_ep_cumulative_rewards"])
                episode_accuracies.append(reward[i] == 1.0)
                episode_lengths.append(info[i]["prev_ep_step_counter"])
                if "prev_ep_tool_use_counter" in info[i]:
                    episode_tool_uses.append(info[i]["prev_ep_tool_use_counter"])

        if print_episodes:
            print("=" * 30)
            print(
                f"Step {info[0]['step_counter']} (Episodes collected so far: {episode_count}/{num_episodes})"
            )
            print(
                "-" * 10,
                "observation",
                "-" * 10,
            )
            pprint(obs[0])
            print(
                "-" * 10,
                "action",
                "-" * 10,
            )
            pprint(action[0])
            print(
                "-" * 10,
                "reward",
                "-" * 10,
            )
            pprint(reward[0])
            print(f"terminated: {terminated[0]}, truncated: {truncated[0]}, correct: {reward[0] == 1.0}")
            print(
                "-" * 10,
                "next observation",
                "-" * 10,
            )
            pprint(next_obs[0])
            print("=" * 30)

        obs = next_obs

        if episode_count >= num_episodes:
            break

    return episode_rewards, episode_lengths, episode_accuracies, episode_tool_uses


def eval(
    env_name: str = "ta:GuessTheNumber-v0",
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    num_episodes: int = 100,
    batch_size: int = 10,
    max_turns: int = 3,
    max_tool_uses: int = 10,
    temperature: float = 0.7,
    print_episodes: bool = True,
    enable_python_tool: bool = True,
    max_tokens: int = 3000,
):
    """Test episode with LLM observation and Python code tool."""
    llm = LLM(
        model=model_name,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def batch_policy(obss):
        assert isinstance(
            obss, List
        ), f"Observation should be a string but is {type(obss)}."
        # return [random.choice(TEST_ACTIONS) for _ in range(len(obss))]
        response = llm.generate(
            obss,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        # print(f"LLM OBSERVATION: {obss!r}")
        # print(f"LLM RESPONSE: {response}")
        actions = [r.outputs[0].text for r in response]
        # print(f"LLM ACTION: {actions!r}")
        return actions

    tool = PythonCodeTool(timeout=5)
    wrappers = []
    if enable_python_tool:
        tool_env_wrapper = partial(
            ToolEnvWrapper, tools=[tool], max_tool_uses=max_tool_uses
        )
        wrappers.append(tool_env_wrapper)
    chat_wrapper = partial(ChatTemplatedObservation, tokenizer=llm.get_tokenizer())
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # chat_wrapper = partial(ChatTemplatedObservation, tokenizer=tokenizer)
    wrappers.append(chat_wrapper)
    wrappers.append(EpisodeTrackingWrapper)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=batch_size,
        wrappers=wrappers,
        max_turns=max_turns,
        async_mode=False,
    )
    start_time = time.time()
    episode_rewards, episode_lengths, episode_accuracies, tool_uses = collect_episodes(
        ta_vec_env,
        policy=batch_policy,
        num_episodes=num_episodes,
        print_episodes=print_episodes,
    )

    # Print eval results
    width = 50
    header = " EVALUATION RESULTS "
    print("\n" + "=" * width)
    print(f"{header:=^{width}}")
    print("=" * width)
    # Configuration
    print(f"  {'Environment:':<20} {env_name}")
    print(f"  {'Model:':<20} {model_name}")
    print(f"  {'Num Episodes:':<20} {len(episode_rewards)}")
    print("-" * width)
    # Metrics
    print(f"  {'Cumulative reward:':<20} {np.mean(episode_rewards):.2f} ({episode_rewards})")
    print(f"  {'Accuracy:':<20} {np.mean(episode_accuracies):.2f} ({episode_accuracies})")
    print(f"  {'Episode Length:':<20} {np.mean(episode_lengths):.2f} ({episode_lengths})")
    if tool_uses:
        print(f"  {'Tool Uses:':<20} {np.mean(tool_uses):.2f} ({tool_uses})")
    print("-" * width)
    print(f"  {'Time Taken:':<20} {time.time() - start_time:.2f} seconds")
    print("=" * width + "\n")


if __name__ == "__main__":
    fire.Fire(eval)

    """Run with:
    python -m eval.eval --env_name ta:GuessTheNumber-v0 --model_name Qwen/Qwen3-0.6B --num_episodes 30 --batch_size 5
    python -m eval.eval --env_name ta:GuessTheNumber-v0 --model_name Qwen/Qwen3-0.6B
    python -m eval.eval --env_name ta:GuessTheNumber-v0 --model_name GAIR/ToRL-1.5B
    python -m eval.eval --env_name eval:MATH500 --model_name GAIR/ToRL-1.5B --temperature 0.0 --num_episodes 100 --batch_size 3 --enable_python_tool False --max_tokens 100
    python -m eval.eval --env_name eval:MATH500 --model_name GAIR/ToRL-1.5B --temperature 0.0 --num_episodes 20 --batch_size 3 --max_tokens 100
    python -m eval.eval --env_name eval:MATH500 --model_name VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step --temperature 0.0 --num_episodes 20 --batch_size 3 --max_tokens 100
    """
