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
    test_set: str = "amc",
    prompt_template="",
    apply_chat_template: bool = False,
    max_tokens: int = 32752,
    temperature: float = 0.6,
    top_p: float = 0.95,
    n: int = 1,
):
    import numpy as np
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    def apply_qwen3_general_template(question: str) -> str:
        return (
            f"<|im_start|>user\nQuestion: {question}"
            "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    TEMPLATE = {"": lambda x: x, "qwen3": apply_qwen3_general_template}

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
    for test_ds in test_set:
        print("testing", test_ds)

        # Single-turn evaluation
        dataset = load_dataset("axon-rl/math-eval")[test_ds]
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
        all_len = []
        for i, output in enumerate(outputs):
            action = output.outputs[0].text
            all_len.append(len(output.outputs[0].token_ids))
            env.answer = dataset["answer"][i]
            _, r, _, _, _ = env.step(action)
            all_pass += float(r == 1)
            # print("Pred", action)
            # print("GT", env.answer)
            # print("-"*50)
        print(
            f"[Without tool call] Tested {len(outputs)} questions; ",
            "Accuracy: ",
            all_pass / len(outputs),
            "Response Length: ",
            np.mean(all_len),
        )


if __name__ == "__main__":

    fire.Fire(
        {
            "llm_episode": test_llm_episode,
            "evaluate": evaluate,
        }
    )
    print(f"\n\nAll tests run.")

    """Run with:
    python -m tests.test_env.test_math llm_episode
    python -m tests.test_env.test_math evaluate --max_tokens 8192
    python -m tests.test_env.test_math evaluate --max_tokens 8192 --model_name Qwen/Qwen3-4B-Base --prompt_template qwen3 --test_set amc,aime24,math
    """
