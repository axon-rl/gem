# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fire
import reasoning_gym

import gem
from gem.utils.debug import run_and_print_episode


def test(env_name: str = "rg:leg_counting"):
    data = reasoning_gym.create_dataset(env_name.removeprefix("rg:"), size=5, seed=42)

    for i, x in enumerate(data):
        q = x["question"]
        a = x["answer"]
        print(f'{i}: q="{q}", a="{a}"')
        print("metadata:", x["metadata"])
        # use the dataset's `score_answer` method for algorithmic verification
        assert data.score_answer(answer=x["answer"], entry=x) == 1.0

    # Test env & reset
    print("\n" * 5)
    rg_env = gem.make(env_name, seed=10, size=int(1e9))
    for _ in range(10):
        run_and_print_episode(rg_env, lambda _: "\\boxed{30}")

    # Test VecEnv & autoreset
    print("\n" * 5)
    num_envs = 8
    base_seed = 42
    rg_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        vec_kwargs=[{"seed": base_seed + i} for i in range(num_envs)],
        size=int(1e9),
    )
    run_and_print_episode(
        rg_vec_env, lambda _: ["\\boxed{30}"] * num_envs, ignore_done=True, max_steps=10
    )


if __name__ == "__main__":
    fire.Fire(test)
    print(f"\n\nAll tests run.\n\n")

    """Run with:
        python -m tests.test_env.test_reasoning_gym --env_name rg:leg_counting
    """
