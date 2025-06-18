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
    model_name: str = "agentica-org/DeepScaleR-1.5B-Preview", max_tokens: int = 32752
):
    from vllm import LLM, SamplingParams

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

    env = gem.make("eval:MATH500", verbose=True)
    dataset = env.dataset
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
        env.answer = dataset["answer"][i]
        _, r, _, _, _ = env.step(action)
        all_pass += float(r == 1)

    print(f"Tested {len(outputs)} questions; ", "Accuracy: ", all_pass / len(outputs))


def evaluate_tool():
    pass


if __name__ == "__main__":

    fire.Fire(
        {
            "llm_episode": test_llm_episode,
            "evaluate": evaluate,
            "evaluate_tool": evaluate_tool,
        }
    )
    print(f"\n\nAll tests run.")

    """Run with:
    python -m tests.test_env.test_math llm_episode
    python -m tests.test_env.test_math evaluate --max_tokens 8192
    """
