from time import time

import fire
from datasets import Dataset

import gem
from gem.envs.code_env import CodeEnv

# Unit tests


def test_reward_code_contests():
    """
    Test the reward function on the code contests dataset.
    """
    model_response = """
```python
import sys
from itertools import permutations
def main():
    N, M, R = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    A, B, C = [], [], []
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        C.append(c)
    INF = float('inf')
    dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        dist[i][i] = 0
    for i in range(M):
        a, b, c = A[i], B[i], C[i]
        dist[a][b] = c
        dist[b][a] = c
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    min_dist = INF
    for perm in permutations(r):
        total = 0
        for i in range(R-1):
            total += dist[perm[i]][perm[i+1]]
        if total < min_dist:
            min_dist = total
    print(min_dist)
if __name__ == "__main__":
    main()
    ```
    """
    dataset = Dataset.from_dict(
        {
            "problem": [""],
            "tests": [
                {
                    "inputs": [
                        # Test case 1: Simple path with 3 cities
                        "4 3 3\n1 2 3\n1 2 3\n2 3 2\n3 4 4\n",
                        # Test case 2: Complete graph with 5 cities
                        "5 10 4\n1 2 3 4\n1 2 5\n1 3 5\n1 4 5\n1 5 5\n2 3 5\n2 4 5\n2 5 5\n3 4 5\n3 5 5\n4 5 5\n",
                        # Test case 3: Larger graph with 7 cities
                        "7 21 4\n1 3 5 7\n1 2 4\n1 3 8\n1 4 1\n1 5 7\n1 6 3\n1 7 9\n2 3 5\n2 4 2\n2 5 6\n2 6 8\n2 7 4\n3 4 7\n3 5 9\n3 6 1\n3 7 6\n4 5 3\n4 6 5\n4 7 8\n5 6 2\n5 7 4\n6 7 7\n",
                    ],
                    "outputs": ["5\n", "15\n", "11\n"],
                }
            ],
        }
    )
    env = CodeEnv(
        dataset=dataset,
        dataset_name="single_codecontests_example",
        max_tests=3,
        verbose=True,
    )
    env.reset()
    print(env.tests)
    st_time = time()
    _, reward, _, _, _ = env.step(model_response)
    print("Time cost", time() - st_time)
    assert reward == 1


# Integrated tests


def test_llm_episode(model_name: str = "agentica-org/DeepCoder-1.5B-Preview"):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.6,
        max_tokens=32768,
        top_p=0.95,
    )

    tokenizer = llm.get_tokenizer()

    env = gem.make("eval:CodeContest", verbose=True)
    obs, _ = env.reset()

    formatted_obs = tokenizer.apply_chat_template(
        [{"content": obs, "role": "user"}], add_generation_prompt=True, tokenize=False
    )
    output = llm.generate(
        [formatted_obs],
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    action = output[0].outputs[0].text
    env.step(action)


def evaluate(
    model_name: str = "agentica-org/DeepCoder-1.5B-Preview", max_tokens: int = 32752
):
    from vllm import LLM, SamplingParams

    NUM_TEST = 200

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.6,
        max_tokens=max_tokens,
        top_p=0.95,
    )

    tokenizer = llm.get_tokenizer()

    env = gem.make("eval:CodeContest", verbose=True)
    dataset = env.dataset[:NUM_TEST]
    obss = dataset["problem"]

    formatted_obss = [
        tokenizer.apply_chat_template(
            [{"content": obs, "role": "user"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for obs in obss
    ]
    outputs = llm.generate(
        formatted_obss,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    all_pass = 0
    for i, output in enumerate(outputs):
        action = output.outputs[0].text
        env.tests = dataset["tests"][i]
        _, r, _, _, _ = env.step(action)
        all_pass += float(r == 1)

    print(f"Tested {len(outputs)} questions; ", "Accuracy: ", all_pass / len(outputs))


if __name__ == "__main__":

    fire.Fire(
        {
            "action": test_reward_code_contests,
            "llm_episode": test_llm_episode,
            "evaluate": evaluate,
        }
    )
    print(f"\n\nAll tests run.")

    """Run with:
    python -m tests.test_env.test_code action
    python -m tests.test_env.test_code llm_episode
    python -m tests.test_env.test_code evaluate
    8k -> 5.45%
    """
