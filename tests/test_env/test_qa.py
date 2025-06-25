import fire

import gem


def test_llm_episode(model_name: str = "Qwen/Qwen3-4B"):
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


if __name__ == "__main__":

    fire.Fire(
        {
            "llm_episode": test_llm_episode,
            "action_sequence": test_action_sequence,
        }
    )

    """Run with:
    python -m tests.test_env.test_qa llm_episode
    python -m tests.test_env.test_qa action_sequence
    """
