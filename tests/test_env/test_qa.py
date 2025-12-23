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

import gem


def test_e2e_llm_episode(model_name: str = "Qwen/Qwen3-4B"):
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

    env = gem.make("eval:QaOpen", verbose=True)
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

    print(f"Action: {action!r}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")


def test_action_sequence():
    """Tests the environment with a sequence of predefined actions."""
    env = gem.make("eval:QaOpen", verbose=True)

    actions = [
        "<answer>The first president of the United States was George Washington.",
        "<answer>The Earth revolves around the Sun.</answer>",
        "<answer>Water is composed of two hydrogen atoms and one oxygen atom.</answer>",
        "<answer>The powerhouse of the cell is the mitochondria.</answer>",
    ]

    for i, action in enumerate(actions):
        obs, _ = env.reset()

        print(f"------ Test {i} ------")
        if i == 3:
            action = f"<answer>     {env.answer[0]}</answer>"
        print(f"Action: {action!r}")
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Next observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

def test_transitions():
    env_set = [
        "eval:QaOpen",
        "eval:2Wiki",
        "eval:PopQA",
        "eval:TriviaQA",
        "qa:HotpotQA", # Train split
        "eval:HotpotQA", # Test split
        "qa:NatualQuestions", # Train split
        "eval:NaturalQuestions", # Test split
        "eval:Bamboogle",
    ]

    for env_name in env_set:
        print(f"Testing environment: {env_name}")
        # All of these are episodic environments anyways so no need to track steps.

        # Test 2 things: Boxed answers, and Tagged answers.

        # Test 1: Boxed answers
        env = gem.make(env_name, verbose=False, extract_boxed=True)
        # Test 1A: Give valid boxed answer
        obs, _ = env.reset()
        answer = env.answer
        ortti = env.step(f"\\boxed{{{answer}}}")
        assert ortti[1] == 1.0, f"Failed boxed answer test in {env_name} with valid boxed answer."
        print(f"Passed boxed answer test with valid boxed answer in {env_name}.")

        # Test 1B: Give invalid boxed answer
        obs, _ = env.reset()
        wrong_answer = "This is definitely not the correct answer."
        ortti = env.step(f"\\boxed{wrong_answer}")
        assert ortti[1] == 0.0, f"Failed boxed answer test in {env_name} with invalid boxed answer."
        print(f"Passed boxed answer test with invalid boxed answer in {env_name}.")

        # Test 1C: Give malformed answer
        obs, _ = env.reset()
        malformed_answer = "This answer is missing the boxed tags."
        ortti = env.step(malformed_answer)
        assert ortti[1] == 0.0, f"Failed boxed answer test in {env_name} with malformed boxed answer."
        print(f"Passed boxed answer test with malformed boxed answer in {env_name}.")

        # Test 1D: Give tagged answer (supposed to be invalid)
        obs, _ = env.reset()
        tagged_answer = f"<answer>{env.answer}</answer>"
        ortti = env.step(tagged_answer)
        assert ortti[1] == 0.0, f"Failed boxed answer test in {env_name} with tagged answer."
        print(f"Passed boxed answer test with tagged answer in {env_name}.")

        # Test 2: Tagged answers
        # Ditto the boxed answers.
        env = gem.make(env_name, verbose=False, extract_boxed=False)
        obs, _ = env.reset()
        answer = env.answer
        # Test 2A: Give valid tagged answer
        ortti = env.step(f"<answer>{answer}</answer>")
        assert ortti[1] == 1.0, f"Failed tagged answer test in {env_name} with valid tagged answer."
        print(f"Passed tagged answer test with valid tagged answer in {env_name}.")

        # Test 2B: Give invalid tagged answer
        obs, _ = env.reset()
        wrong_answer = "This is definitely not the correct answer." 
        ortti = env.step(f"<answer>{wrong_answer}</answer>")
        assert ortti[1] == 0.0, f"Failed tagged answer test in {env_name} with invalid tagged answer."
        print(f"Passed tagged answer test with invalid tagged answer in {env_name}.")   

        # Test 2C: Give malformed tagged answer
        obs, _ = env.reset()
        malformed_answer = "This answer is missing the answer tags."
        ortti = env.step(malformed_answer)
        assert ortti[1] == 0.0, f"Failed tagged answer test in {env_name} with malformed tagged answer."
        print(f"Passed tagged answer test with malformed tagged answer in {env_name}.") 

        # Test 2D: Give boxed answer (supposed to be invalid)
        obs, _ = env.reset()
        boxed_answer = f"\\boxed{{{env.answer}}}"
        ortti = env.step(boxed_answer)
        assert ortti[1] == 0.0, f"Failed tagged answer test in {env_name} with boxed answer."
        print(f"Passed tagged answer test with boxed answer in {env_name}.")

        print(f"Completed transition tests for environment: {env_name}\n")

def evaluate_llm(
    model_name: str = "Qwen/Qwen3-4B",
    env_name: str = "eval:QaOpen",
    max_tokens: int = 16384,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n_examples: int = -1,
    question_key: str = "question",
    answer_key: str = "answer",
):
    import numpy as np
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    tokenizer = llm.get_tokenizer()
    env = gem.make(env_name, verbose=True, load_from_cache_file=False)
    dataset = env.dataset

    if n_examples > 0:
        dataset = dataset.select(range(min(n_examples, len(dataset))))

    obss = dataset[question_key]

    formatted_obss = [
        tokenizer.apply_chat_template(
            [{"content": obs, "role": "user"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for obs in obss
    ]
    print("example question", formatted_obss[0])

    # formatted_obss = formatted_obss * n

    outputs = llm.generate(
        formatted_obss,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    all_pass = 0
    num_done = 0
    all_len = []
    progress_bar = tqdm(total=len(dataset))
    episodes = []

    for i, output in enumerate(outputs):
        action = output.outputs[0].text
        all_len.append(len(output.outputs[0].token_ids))
        env.answer = dataset[answer_key][i]
        _, r, _, _, _ = env.step(action)
        all_pass += float(r == 1)
        num_done += 1
        progress_bar.update(1)
        progress_bar.set_description(
            f"{env_name} | Accuracy: {all_pass / num_done:.2%}"
        )
        episodes.append(
            [
                {
                    "obs": formatted_obss[i],
                    "action": action,
                    "reward": r,
                    "ground_truth": env.answer,
                }
            ]
        )
    acc = all_pass / len(outputs)
    print(
        f"[QA Evaluation] Tested {len(outputs)} questions; ",
        "Accuracy: ",
        acc,
        "Response Length: ",
        np.mean(all_len),
    )
    return acc, episodes


def benchmark_llm(
    env_names: str = "eval:2Wiki,eval:PopQA,eval:TriviaQA,eval:HotpotQA,eval:Bamboogle,eval:NaturalQuestions,eval:Musique",
    model_name: str = "Qwen/Qwen3-1.7B",
    output_dir: str = None,
    **kwargs,
):
    import json
    import os
    from pathlib import Path

    import pandas as pd

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

    print(f"Running QA evaluation on {len(env_names)} environments...")
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
            acc, episodes = evaluate_llm(model_name=model_name, env_name=env_name, **kwargs)

            result = {
                "env_name": env_name,
                "model_name": model_name,
                "accuracy": acc,
                "num_episodes": len(episodes),
            }

            all_episodes[env_name] = episodes

            print(f"✓ {env_name}: {acc:.2%} accuracy")

        except Exception as e:
            print(f"✗ {env_name}: Error - {str(e)}")
            result = {
                "env_name": env_name,
                "model_name": model_name,
                "accuracy": None,
                "num_episodes": 0,
                "error": str(e),
            }

        results.append(result)

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

    return results, all_episodes


if __name__ == "__main__":

    fire.Fire(
        {
            "e2e_llm_episode": test_e2e_llm_episode,
            "action_sequence": test_action_sequence,
            "test_transitions": test_transitions,
            "evaluate_llm": evaluate_llm,
            "benchmark_llm": benchmark_llm,
        }
    )

    """Run with:
    python -m tests.test_env.test_qa e2e_llm_episode
    python -m tests.test_env.test_qa action_sequence
    python -m tests.test_env.test_qa test_transitions
    python -m tests.test_env.test_qa evaluate
    python -m tests.test_env.test_qa benchmark_llm
    """
