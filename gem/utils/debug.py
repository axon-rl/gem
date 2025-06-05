"""Debugging utils."""


def run_and_print_episode(env, policy):
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        print(
            "-" * 10,
            "observation",
            "-" * 10,
        )
        print(obs)
        print(
            "-" * 10,
            "action",
            "-" * 10,
        )
        print(action)
        print(
            "-" * 10,
            "reward",
            "-" * 10,
        )
        print(reward)
        print("=" * 30)
        obs = next_obs
