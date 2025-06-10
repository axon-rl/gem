#!/usr/bin/env python

# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
import asyncio
import logging
import os
import sys
import time
from functools import partial
from pprint import pprint
from typing import List

import fire
import numpy as np
from vllm import LLM, SamplingParams

import gem
from gem.wrappers.stateful_observation import ConcatenatedObservation

# Add parent directory to path to import PistonTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def collect_episodes(
    env, policy, num_episodes: int = 10, print_episodes: bool = False
):
    num_envs = env.num_envs
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    env_rewards = [[] for _ in range(num_envs)]
    env_steps = [0 for _ in range(num_envs)]
    tool_uses = []
    obs, _ = await env.reset()
    done = False
    step_count = 0
    while True:
        step_count += 1
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = await env.step(action)
        done = terminated | truncated

        for i in range(num_envs):
            env_rewards[i].append(reward[i])
            env_steps[i] += 1
            if done[i]:
                episode_count += 1
                episode_rewards.append(np.sum(env_rewards[i]))
                env_rewards[i] = []
                episode_lengths.append(env_steps[i])
                env_steps[i] = 0
                if hasattr(env.envs[i], "tool_use_counter"):
                    tool_uses.append(env.envs[i].tool_use_counter)

        if print_episodes:
            print("=" * 30)
            print(
                f"Step {env_steps[0]} (Episodes collected so far: {episode_count}/{num_episodes})"
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
            print(f"terminated: {terminated[0]}, truncated: {truncated[0]}")
            print(
                "-" * 10,
                "next observation",
                "-" * 10,
            )
            pprint(next_obs[0])
            print("=" * 30)

        if episode_count >= num_episodes:
            break

    return episode_rewards, episode_lengths, tool_uses


async def eval(
    env_name: str = "ta:GuessTheNumber-v0",
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    num_episodes: int = 100,
    batch_size: int = 10,
    max_turns: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.95,
    print_episodes: bool = False,
):
    """Test episode with LLM observation and Python code tool."""
    llm = LLM(
        model=model_name,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=256,
        top_p=top_p,
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
        # print(f"LLM OBSERVATION: {obss!r}")
        # print(f"LLM RESPONSE: {response}")
        actions = [r.outputs[0].text for r in response]
        # print(f"LLM ACTION: {actions!r}")
        return actions

    tool = PythonCodeTool()
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=None)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=batch_size,
        wrappers=[tool_env_wrapper, ConcatenatedObservation],
        max_turns=max_turns,
        async_mode=True,
    )
    start_time = time.time()
    episode_rewards, episode_lengths, tool_uses = await collect_episodes(
        ta_vec_env,
        policy=batch_policy,
        num_episodes=num_episodes,
        print_episodes=print_episodes,
    )
    print("\n" * 5, "EVALUATION RESULTS")
    print(f"----TIME: {time.time() - start_time:.2f} seconds")
    print(f"----ENV: {env_name}")
    print(f"----MODEL: {model_name}")
    print(f"----NUM EPISODES: {len(episode_rewards)}")
    print(f"----EPISODE REWARD: {np.mean(episode_rewards)} ({episode_rewards})")
    print(f"----EPISODE LENGTH: {np.mean(episode_lengths)} ({episode_lengths})")
    if tool_uses:
        print(f"----TOOL USES: {np.mean(tool_uses)} ({tool_uses})")


if __name__ == "__main__":
    """Run with:
    python -m gem.eval.eval --env_name ta:GuessTheNumber-v0 --model_name Qwen/Qwen3-0.6B --num_episodes 30 --batch_size 5 --print_episodes True
    python -m gem.eval.eval --env_name ta:GuessTheNumber-v0 --model_name Qwen/Qwen3-0.6B --print_episodes True
    python -m gem.eval.eval --env_name ta:GuessTheNumber-v0 --model_name GAIR/ToRL-1.5B --print_episodes True
    """

    def run_eval(*args, **kwargs):
        asyncio.run(eval(*args, **kwargs))

    fire.Fire(run_eval)
