import gem


def test():
    env = gem.make("ta:GuessTheNumber-v0")
    obs, _ = env.reset()

    done = False
    while not done:
        action = env.sample_random_action()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        print(
            "=" * 20,
            "observation",
            "=" * 20,
        )
        print(obs)
        print(
            "=" * 20,
            "action",
            "=" * 20,
        )
        print(action)
        print(
            "=" * 20,
            "reward",
            "=" * 20,
        )
        print(reward)
        print(
            "=" * 20,
            "next observation",
            "=" * 20,
        )
        print(next_obs)
        obs = next_obs


if __name__ == "__main__":
    test()
