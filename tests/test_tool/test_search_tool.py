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

import json
import os
import random
from functools import partial
from pathlib import Path
from typing import Optional

import fire
import pandas as pd
from transformers import AutoTokenizer

import gem
from gem.tools.search_tool import SearchTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

TEST_ACTIONS = [
    """<search>What is the capital of France?</search> ...""",
    """Dummy action""",
    """<think>I need to search for Python list comprehension examples</think><search>Python list comprehension examples</search> ...""",
    """```<search>First query</search> ... <search>Second query</search>``` ...""",
    """```<search>Test the max number of tools</search> ...``` ...""",
]


def test_single_action(search_url: str, env_name: str = "game:GuessTheNumber-v0"):
    env = gem.make(env_name, max_turns=4)
    tool = SearchTool(search_url=search_url, topk=2)
    env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    obs, info = env.reset()

    print(f"Using real requests with URL: {search_url}")

    for i, test_action in enumerate(TEST_ACTIONS):
        print(f"------ Test {i} ------")
        print(f"Action: {test_action!r}")
        try:
            obs, reward, terminated, truncated, info = env.step(test_action)
            print(f"Observation: {obs}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info}\n")
        except Exception as e:
            print(f"Error during real request: {e}")
            print("Observation: [Error occurred]")
            print("Continuing with next test...\n")


def test_episode(
    search_url: str,
    env_name: str = "qa:NaturalQuestions",
    tokenizer_name: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
):
    env = gem.make(env_name, max_turns=3, load_from_cache_file=False)
    policy = lambda _: random.choice(TEST_ACTIONS)
    tool = SearchTool(search_url=search_url, topk=2)

    print(f"Using real requests with URL: {search_url}")

    def run_episode_test(episode_name, wrapped_env, policy_func=None):
        print(f"\n{episode_name}")
        try:
            run_and_print_episode(wrapped_env, policy_func or policy)
        except Exception as e:
            print(f"Error during real request episode: {e}")

    # Episode 1: Default observation
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_episode_test("EPISODE 1: DEFAULT OBSERVATION", wrapped_env)

    # Episode 2: Chat template observation
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_episode_test("EPISODE 2: CHAT TEMPLATE OBSERVATION", wrapped_env)

    # Episode 3: Chat template observation on reset
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat_on_reset"](
        wrapped_env, tokenizer=tokenizer
    )
    run_episode_test("EPISODE 3: CHAT TEMPLATE OBSERVATION ON RESET", wrapped_env)

    # Batch episode: Sync vectorized env
    num_envs = 3
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        [env_name] * num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        max_turns=3,
    )
    batch_policy = lambda _: [random.choice(TEST_ACTIONS) for _ in range(num_envs)]
    run_episode_test(
        "EPISODE 4: BATCH EPISODE SYNC VECTORIZED ENV", ta_vec_env, batch_policy
    )


def test_llm_episode(
    search_url: str,
    env_name: str = "eval:QaOpen",
    model_name: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
):
    """Test episode with LLM observation and Search tool."""
    from datasets import Dataset
    from vllm import LLM, SamplingParams

    env = gem.make(env_name, max_turns=3)
    # hack: fix the question and answer of the dataset
    question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"
    prompt = f"Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
    answer = "Sergei Fedorov"
    dataset = Dataset.from_dict({"question": [prompt], "answer": [answer]})
    env.dataset = dataset

    llm = LLM(
        model=model_name,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        max_tokens=100,
        top_p=0.95,
    )
    tokenizer = llm.get_tokenizer()

    def policy(obs):
        assert isinstance(obs, str), (
            f"Observation should be a string but is {type(obs)}."
        )
        response = llm.generate(
            [obs],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        action = response[0].outputs[0].text
        return action

    tool = SearchTool(search_url=search_url, topk=2)

    print(f"Using real requests with URL: {search_url}")

    def run_episode_test(episode_name, wrapped_env, policy_func, **kwargs):
        print(f"\n{episode_name}")
        try:
            run_and_print_episode(wrapped_env, policy_func, **kwargs)
        except Exception as e:
            print(f"Error during real request episode: {e}")

    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_episode_test("EPISODE 1: CHAT TEMPLATE OBSERVATION", wrapped_env, policy)


def evaluate(
    search_url: Optional[str] = None,
    model_name: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
    env_name: str = "eval:QaOpen",
    max_tokens: int = 3000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n_examples: int = -1,
    max_tool_uses: int = 5,
    obs_wrapper: str = "concat_chat",
    verbose: bool = False,
) -> tuple[float, list[list[dict]]]:
    """Evaluate the model on the QaOpen dataset with the Search tool."""
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    stop_tokens = ["</answer>"]

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop_tokens,
        include_stop_str_in_output=True,
    )
    tokenizer = llm.get_tokenizer()

    base_env = gem.make(env_name, seed=42)
    dataset = base_env.dataset
    if n_examples > 0:
        dataset = dataset.select(range(n_examples))
        base_env.dataset = dataset

    if verbose:
        print(
            "First question:\n",
            "-" * 20,
            "\n",
            dataset[0]["question"],
            "\n",
            "-" * 20,
            "\n",
        )

    tool = SearchTool(search_url=search_url, topk=3, timeout=5)
    wrapped_env = ToolEnvWrapper(
        base_env,
        tools=[tool],
        max_tool_uses=max_tool_uses,
        tool_reward=0.0,
        tool_success_reward=0.0,
    )
    wrapped_env = WRAPPER_FACTORY[obs_wrapper](wrapped_env, tokenizer=tokenizer)

    progress_bar = tqdm(total=len(dataset))
    num_done = 0
    all_pass = 0
    episodes = []

    while True:
        obs, info = wrapped_env.reset()
        terminated = False
        truncated = False
        ground_truth = wrapped_env.unwrapped.answer
        episode = []

        if wrapped_env.unwrapped.epoch > 0:  # force to end if traversed full dataset
            break

        while not (terminated or truncated):
            response = llm.generate(
                [obs], sampling_params=sampling_params, use_tqdm=False
            )
            action = response[0].outputs[0].text
            next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated

            episode.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "ground_truth": ground_truth,
                }
            )

            obs = next_obs

        if reward == 1:
            all_pass += 1
        num_done += 1
        episodes.append(episode)
        progress_bar.update(1)
        progress_bar.set_description(
            f"{env_name} | Accuracy: {all_pass / num_done:.2%}"
        )

        if verbose:
            print(f"Action: {action!r}")
            print(f"Answer: {base_env.answer!r}")
            print(f"Observation: {obs!r}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info!r}")

    acc = all_pass / len(dataset)
    print(f"Tested {len(dataset)} questions; Accuracy: {acc:.2%}")

    return acc, episodes


def benchmark(
    env_names: str = "eval:2Wiki,eval:PopQA,eval:TriviaQA,eval:HotpotQA,eval:Bamboogle,eval:NaturalQuestions,eval:Musique",
    model_name: str = "Qwen/Qwen3-1.7B",
    output_dir: str = None,
    **kwargs,
):
    env_names = env_names.split(",")

    # Determine output directory
    save_results = False
    if output_dir:
        save_results = True
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Check if model_name is a local directory
        if os.path.isdir(model_name):
            output_dir = Path(model_name)
            output_dir = os.path.join(output_dir.parent, f"eval_{output_dir.stem}")
            save_results = True
            os.makedirs(output_dir, exist_ok=True)

    # Store results
    results = []
    all_episodes = {}

    print(f"Running evaluation on {len(env_names)} environments...")
    print(f"Model: {model_name}")
    if save_results:
        print(f"Output directory: {output_dir}")
    else:
        print(
            "Results will not be saved (output_dir not specified and model_name is not a local directory)"
        )

    # Run evaluation for each environment
    for env_name in env_names:
        print(f"\nEvaluating on {env_name}...")

        try:
            acc, episodes = evaluate(model_name=model_name, env_name=env_name, **kwargs)

            results.append(
                {
                    "env_name": env_name,
                    "model_name": model_name,
                    "accuracy": acc,
                    "num_episodes": len(episodes),
                }
            )

            all_episodes[env_name] = episodes

            print(f"✓ {env_name}: {acc:.2%} accuracy")

        except Exception as e:
            print(f"✗ {env_name}: Error - {str(e)}")
            results.append(
                {
                    "env_name": env_name,
                    "model_name": model_name,
                    "accuracy": None,
                    "num_episodes": 0,
                    "error": str(e),
                }
            )

        # Save results if output directory is determined
        if save_results:
            # Save accuracy results to CSV
            df = pd.DataFrame(results)
            csv_path = os.path.join(output_dir, "evaluation_results.csv")
            df.to_csv(csv_path, index=False)

            # Save episodes to JSON
            json_path = os.path.join(output_dir, "evaluation_episodes.json")
            with open(json_path, "w") as f:
                json.dump(all_episodes, f, indent=2)

    # Print summary
    if save_results:
        print(f"\nAccuracy results saved to: {csv_path}")
        print(f"Episodes saved to: {json_path}")
    print(f"\nSummary:")
    print(f"Total environments: {len(env_names)}")
    successful_results = [r for r in results if r["accuracy"] is not None]
    if successful_results:
        avg_acc = sum(r["accuracy"] for r in successful_results) / len(
            successful_results
        )
        print(f"Average accuracy: {avg_acc:.2%}")

    return


def main():
    """Run with:
    # To test with real search server:
    python -m tests.test_tool.test_search_tool single_action --search_url http://localhost:8000/retrieve
    python -m tests.test_tool.test_search_tool episode --search_url http://localhost:8000/retrieve
    python -m tests.test_tool.test_search_tool llm_episode --search_url http://localhost:8000/retrieve
    python -m tests.test_tool.test_search_tool evaluate --search_url http://localhost:8000/retrieve --n_examples 1 --verbose
    python -m tests.test_tool.test_search_tool benchmark --search_url http://localhost:8000/retrieve --n_examples 5 --verbose --env_names eval:2Wiki,eval:PopQA
    """
    fire.Fire(
        {
            "single_action": test_single_action,
            "episode": test_episode,
            "llm_episode": test_llm_episode,
            "evaluate": evaluate,
            "benchmark": benchmark,
        }
    )


if __name__ == "__main__":
    main()
