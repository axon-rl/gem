from typing import List

import fire

import gem


def test_llm_episode(model_name: str = "agentica-org/DeepScaleR-1.5B-Preview"):
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

    env = gem.make("eval:MATH500", verbose=True)
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
    print(env.step(action))


def evaluate(
    model_name: str = "agentica-org/DeepScaleR-1.5B-Preview",
    test_set_name: str = "amc",
    prompt_template="",
    apply_chat_template: bool = False,
    max_tokens: int = 32752,
    temperature: float = 0.6,
    top_p: float = 0.95,
    n: int = 1,
):
    import numpy as np
    from datasets import load_dataset
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    def apply_qwen3_general_template(question: str) -> str:
        return (
            f"<|im_start|>user\nQuestion: {question}"
            "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    def apply_qwen3_think_template(question: str) -> str:
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>"

    TEMPLATE = {"": lambda x: x, "qwen3": apply_qwen3_general_template, "qwen3_think": apply_qwen3_think_template}

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    tokenizer = llm.get_tokenizer()

    env = gem.make("eval:MATH500", verbose=True)  # Dummy env for the reward function.

    # Single-turn evaluation
    dataset = load_dataset("axon-rl/math-eval")[test_set_name]
    obss = dataset["problem"]

    formatted_obss = [
        (
            tokenizer.apply_chat_template(
                [{"content": obs, "role": "user"}],
                add_generation_prompt=True,
                tokenize=False,
            )
            if apply_chat_template
            else TEMPLATE[prompt_template](obs)
        )
        for obs in obss
    ]
    print("example question", formatted_obss[0])

    formatted_obss = formatted_obss * n

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
        env.answer = dataset["answer"][i]
        _, r, _, _, _ = env.step(action)
        all_pass += float(r == 1)
        num_done += 1
        progress_bar.update(1)
        progress_bar.set_description(
            f"{test_set_name} | Accuracy: {all_pass / num_done:.2%}"
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
        f"[Without tool call] Tested {len(outputs)} questions; ",
        "Accuracy: ",
        acc,
        "Response Length: ",
        np.mean(all_len),
    )
    return acc, episodes


def benchmark(
    env_names: List[str] = ["aime24", "amc", "math", "minerva", "olympiad_bench"],
    model_name: str = "Qwen/Qwen3-1.7B",
    output_dir: str = None,
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
                model_name=model_name, test_set_name=env_name, **kwargs
            )

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
            df = pd.DataFrame({k: [v] for k, v in result.items()})
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
            "llm_episode": test_llm_episode,
            "evaluate": evaluate,
            "benchmark": benchmark,
        }
    )

    """Run with:
    python -m tests.test_env.test_math llm_episode
    python -m tests.test_env.test_math evaluate --max_tokens 8192
    python -m tests.test_env.test_math evaluate --max_tokens 8192 --model_name Qwen/Qwen3-4B-Base --prompt_template qwen3 --test_set amc,aime24,math
    python -m tests.test_env.test_math benchmark --max_tokens 4096 --model_name Qwen/Qwen3-4B-Base --prompt_template qwen3
    """
