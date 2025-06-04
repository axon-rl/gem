import gem


def test():
    env = gem.make("GuessTheNumber-v0")
    obs, _ = env.reset()

    done = False
    while not done:
        action = env.sample_random_action()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        print("=" * 20)
        print(obs)
        print(action)
        print(reward)
        obs = next_obs


if __name__ == "__main__":
    test()
