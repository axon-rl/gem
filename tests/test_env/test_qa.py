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

def evaluate(model_name: str = "Qwen/Qwen3-4B", max_tokens: int = 32768):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.6,
        max_tokens=max_tokens,
        top_p=0.95,
    )

    tokenizer = llm.get_tokenizer()

    env = gem.make("eval:QaOpen", verbose=True)
    dataset = env.dataset
    dataset = dataset.select(range(500))
    obss = dataset["question"]
    prompt_template = """Answer the given question. 
You must conduct reasoning inside <think> and </think> first every time you get new information. You should directly provide the answer inside <answer> and </answer>, without detailed illustrations.

Here are some examples:
Question: What is the capital of France?
<think>France is a country in Europe. Its capital is a well-known city.</think>
<answer>Paris</answer>

Question: Who wrote 'Pride and Prejudice'?
<think>'Pride and Prejudice' is a famous novel from the 19th century. The author is a well-known English novelist.</think>
<answer>Jane Austen</answer>

Question: What is the largest planet in our solar system?
<think>The solar system contains several planets. The largest one is a gas giant.</think>
<answer>Jupiter</answer>

Question: {question}
"""

    formatted_obss = [
        tokenizer.apply_chat_template(
            [{"content": prompt_template.format(question=obs), "role": "user"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for obs in obss
    ]
    outputs = llm.generate(
        formatted_obss,
        sampling_params=sampling_params,
        # use_tqdm=True,
    )
    all_pass = 0
    for i, output in enumerate(outputs):
        action = output.outputs[0].text
        env.answer = dataset["answer"][i]
        _, r, _, _, _ = env.step(action)
        all_pass += float(r == 1)

    print(f"Tested {len(outputs)} questions; ", "Accuracy: ", all_pass / len(outputs))


if __name__ == "__main__":

    fire.Fire(
        {
            "llm_episode": test_llm_episode,
            "action_sequence": test_action_sequence,
            "evaluate": evaluate,
        }
    )

    """Run with:
    python -m tests.test_env.test_qa llm_episode
    python -m tests.test_env.test_qa action_sequence
    python -m tests.test_env.test_qa evaluate
    """
