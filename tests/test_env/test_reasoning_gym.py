import reasoning_gym

import gem
from gem.utils.debug import run_and_print_episode


def test():
    data = reasoning_gym.create_dataset("leg_counting", size=5, seed=42)

    for i, x in enumerate(data):
        q = x["question"]
        a = x["answer"]
        print(f'{i}: q="{q}", a="{a}"')
        print("metadata:", x["metadata"])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x["answer"], entry=x) == 1.0

    print("\n" * 5)
    rg_env = gem.make("rg:leg_counting", seed=10, size=int(1e9))
    for _ in range(10):
        run_and_print_episode(rg_env, lambda _: "\\boxed{30}")


if __name__ == "__main__":
    test()
