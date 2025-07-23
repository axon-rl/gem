import logging
import random
from functools import partial
from typing import List

import fire

import gem
from gem.envs.math_env import MathEnv
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_ACTIONS = [
    """<python>from gem.tools.python_code_tool import PythonCodeTool\ntool = PythonCodeTool()\ntool.execute_action(1)</python> ...""",
    """<python>print('Hello from Python!')</python> ...""",
    """Dummy action""",
    """<python>import sys\n\nprint('Hello from Python!')\nprint(f'Arguments: {sys.argv[1:]}')\nfor i in range(5):\n    print(f'Number {i}')</python> ...""",
    """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ...""",
    """```<python>prnit('Hello from Python!')</python> ...""",
    "\\boxed{30}",
]

SLEEP_ACTION = """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ..."""


def test_single_action(env_name: str = "game:GuessTheNumber-v0"):
    env = gem.make(env_name, max_turns=3)
    tool = PythonCodeTool(timeout=2, keep_error_last_line=True)
    env = ToolEnvWrapper(env, tools=[tool])
    obs, info = env.reset()
    for i, test_action in enumerate(TEST_ACTIONS):
        print(f"------ Test {i} ------")
        print(f"Action: {test_action!r}")
        obs, reward, terminated, truncated, info = env.step(test_action)
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}\n")


def test_episode(env_name: str = "game:GuessTheNumber-v0"):
    from transformers import AutoTokenizer

    env = gem.make(env_name, max_turns=3)
    policy = lambda _: random.choice(TEST_ACTIONS)
    tool = PythonCodeTool(timeout=2)

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 2: CHAT TEMPLATE OBSERVATION")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 3: CHAT ON RESET TEMPLATE OBSERVATION")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat_on_reset"](
        wrapped_env, tokenizer=tokenizer
    )
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "BATCH EPISODE 1: SYNC VECTORIZED ENV")
    num_envs = 4
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=False,
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        # policy=lambda _: [random.choice(TEST_ACTIONS) for _ in range(num_envs)],
        policy=lambda _: [SLEEP_ACTION for _ in range(num_envs)],
        ignore_done=True,
        max_steps=5,
    )

    print("\n" * 5, "BATCH EPISODE 2: ASYNC VECTORIZED ENV")
    num_envs = 4
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=True,
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        # policy=lambda _: [random.choice(TEST_ACTIONS) for _ in range(num_envs)],
        policy=lambda _: [SLEEP_ACTION for _ in range(num_envs)],
        ignore_done=True,
        max_steps=2,
    )


def test_llm_episode(
    env_name: str = "game:GuessTheNumber-v0", model_name: str = "Qwen/Qwen3-0.6B-Base"
):
    """Test episode with LLM observation and Python code tool."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    env = gem.make(env_name, max_turns=3)
    llm = LLM(
        model=model_name,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        max_tokens=100,
        top_p=0.95,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def policy(obs):
        assert isinstance(
            obs, str
        ), f"Observation should be a string but is {type(obs)}."
        response = llm.generate(
            [obs],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        # print(f"LLM OBSERVATION: {obs!r}")
        # print(f"LLM RESPONSE: {response}")
        action = response[0].outputs[0].text
        # print(f"LLM ACTION: {action!r}")
        return action

    def batch_policy(obss):
        assert isinstance(
            obss, List
        ), f"Observation should be a string but is {type(obss)}."
        response = llm.generate(
            obss,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        print(f"LLM OBSERVATION: {obss!r}")
        print(f"LLM RESPONSE: {response}")
        actions = [r.outputs[0].text for r in response]
        print(f"LLM ACTION: {actions!r}")
        return actions

    tool = PythonCodeTool(timeout=2)

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 2: CHAT TEMPLATE OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "BATCH EPISODE 1: SYNC VECTORIZED ENV")
    num_envs = 4
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=False,
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        policy=batch_policy,
        ignore_done=True,
        max_steps=2,
    )

    print("\n" * 5, "BATCH EPISODE 2: ASYNC VECTORIZED ENV")
    num_envs = 4
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=True,
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        policy=batch_policy,
        ignore_done=True,
        max_steps=2,
    )


def evaluate(
    model_name: str = "Qwen/Qwen3-4B-Base",
    test_set_name: str = "amc",
    max_tokens: int = 4096,
    max_model_len: int = 30000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    n_examples: int = -1,
    max_tool_uses: int = 5,
    keep_error_last_line: bool = False,
    obs_wrapper: str = "concat_chat",
    dump_dir: str = "",
):
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

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
    )
    tokenizer = llm.get_tokenizer()

    try:
        base_env = gem.make(test_set_name)
    except Exception:
        base_env = MathEnv(f"axon-rl/math-eval", test_set_name)

    dataset = base_env.dataset
    if n_examples > 0:
        dataset = dataset.shuffle()
        dataset = dataset.select(range(n_examples))
        base_env.dataset = dataset

    tool = PythonCodeTool(timeout=2, keep_error_last_line=keep_error_last_line)
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
        done = False
        ground_truth = wrapped_env.env.env.answer
        episode = []

        if wrapped_env.env.env.epoch > 0:  # force to end if traversed full dataset
            break
        print("new episode")
        while not done:
            response = llm.generate(
                [obs], sampling_params=sampling_params, use_tqdm=False
            )
            action = response[0].outputs[0].text
            next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated

            _obs_len = len(tokenizer.encode(next_obs))
            print(_obs_len, len(response[0].outputs[0].token_ids))
            if _obs_len > max_model_len:
                print(f"[Warning] Too long obs: {_obs_len}")
                done = True

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
            f"{test_set_name} | Accuracy: {all_pass / num_done:.2%}"
        )

    if dump_dir:
        import json
        import os

        # Save episodes to JSON
        json_path = os.path.join(dump_dir, "evaluation_episodes.json")
        with open(json_path, "w") as f:
            json.dump(episodes, f, indent=2)
    acc = all_pass / len(dataset)
    print(f"Tested {len(dataset)} questions; Accuracy: {acc:.2%}")
    if not dump_dir:
        return acc, episodes


def benchmark(
    env_names: List[str] = ["aime24", "amc", "math", "minerva", "olympiad_bench"],
    model_name: str = "Qwen/Qwen3-1.7B",
    output_dir: str = None,
    keep_error_last_line: bool = False,
    **kwargs,
):
    import json
    import os
    from pathlib import Path

    import pandas as pd

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
            acc, episodes = evaluate(
                model_name=model_name,
                test_set_name=env_name,
                keep_error_last_line=keep_error_last_line,
                **kwargs,
            )

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
            if os.path.exists(csv_path):
                df.to_csv(csv_path, index=False, header=False, mode="a")
            else:
                df.to_csv(csv_path, index=False)

            # Save episodes to JSON
            json_path = os.path.join(output_dir, "evaluation_episodes.json")
            with open(json_path, "w") as f:
                json.dump(all_episodes, f, indent=2)

    # Print summary
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
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(
        {
            "single_action": test_single_action,
            "episode": test_episode,
            "llm_episode": test_llm_episode,
            "evaluate": evaluate,
            "benchmark": benchmark,
        }
    )

    """Run with:
    python -m tests.test_tool.test_python_code_tool single_action --env_name game:GuessTheNumber-v0
    python -m tests.test_tool.test_python_code_tool episode --env_name game:GuessTheNumber-v0
    python -m tests.test_tool.test_python_code_tool llm_episode --env_name game:GuessTheNumber-v0 --model_name Qwen/Qwen3-0.6B-Base
    python -m tests.test_tool.test_python_code_tool episode --env_name eval:MATH500
    python -m tests.test_tool.test_python_code_tool llm_episode --env_name eval:MATH500 --model_name Qwen/Qwen3-0.6B-Base
    python -m tests.test_tool.test_python_code_tool evaluate 
    python -m tests.test_tool.test_python_code_tool benchmark --model_name
    """
