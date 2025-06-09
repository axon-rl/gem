"""Debugging utils."""

from pprint import pprint

import numpy as np


def run_and_print_episode(env, policy, ignore_done: bool = False, max_steps: int = 1e9):
    obs, _ = env.reset()
    done = False
    step_count = 0
    while True:
        step_count += 1
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        print("=" * 30)
        print(
            "-" * 10,
            "observation",
            "-" * 10,
        )
        pprint(obs)
        print(
            "-" * 10,
            "action",
            "-" * 10,
        )
        pprint(action)
        print(
            "-" * 10,
            "reward",
            "-" * 10,
        )
        pprint(reward)

        done = terminated | truncated

        print("terminated: ", terminated, "truncated: ", truncated)
        if isinstance(done, np.ndarray):
            done = done.all()

        print("=" * 30)
        obs = next_obs

        if not ignore_done and done:
            break
        if step_count >= max_steps:
            break
