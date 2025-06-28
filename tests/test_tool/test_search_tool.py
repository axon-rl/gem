import random
from functools import partial
from typing import List

import fire
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


def test_single_action(search_url: str, env_name: str = "ta:GuessTheNumber-v0"):
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


def test_episode(search_url: str, env_name: str = "ta:GuessTheNumber-v0"):
    env = gem.make(env_name, max_turns=3)
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
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_episode_test("EPISODE 2: CHAT TEMPLATE OBSERVATION", wrapped_env)

    # Episode 3: Chat template observation on reset
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat_on_reset"](wrapped_env, tokenizer=tokenizer)
    run_episode_test("EPISODE 3: CHAT TEMPLATE OBSERVATION ON RESET", wrapped_env)

    # Batch episode: Sync vectorized env
    print("\nBATCH EPISODE: SYNC VECTORIZED ENV")
    num_envs = 3
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        max_turns=3,
    )
    batch_policy = lambda _: [random.choice([TEST_ACTIONS[2]]) for _ in range(num_envs)]
    run_episode_test("", ta_vec_env, batch_policy)


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
    prompt = f'Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n'
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
        assert isinstance(
            obs, str
        ), f"Observation should be a string but is {type(obs)}."
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
    search_url: str,
    model_name: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
    max_tokens: int = 1024,
    n_examples: int = 500,
    max_tool_uses: int = 4,
    verbose: bool = False,
):
    """Evaluate the model on the QaOpen dataset with the Search tool."""
    from datasets import Dataset
    from tqdm import tqdm
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

    tool = SearchTool(search_url=search_url, topk=3)
    base_env = gem.make("eval:QaOpen", seed=42)
    dataset = base_env.dataset
    dataset = dataset.select(range(n_examples))
    base_env.dataset = dataset

    print("First question:\n", '-'*20, '\n', dataset[0]["question"], '\n', '-'*20, '\n')

    wrapped_env = ToolEnvWrapper(base_env, tools=[tool], max_tool_uses=max_tool_uses)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)

    all_pass = 0
    for _ in tqdm(range(n_examples)):
        obs, info = wrapped_env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            response = llm.generate(
                [obs], sampling_params=sampling_params, use_tqdm=False
            )
            action = response[0].outputs[0].text
            next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
            obs = next_obs
        if reward == 1:
            all_pass += 1
    
        if verbose: 
            print(f"Action: {action!r}")
            print(f"Answer: {base_env.answer!r}")
            print(f"Observation: {obs!r}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info!r}")

    print(f"Tested {len(dataset)} questions; Accuracy: {all_pass / len(dataset)}")


def main():
    """Run with:
    # To test with real search server:
    python -m tests.test_tool.test_search_tool single_action --search_url http://localhost:8000/retrieve
    python -m tests.test_tool.test_search_tool episode --search_url http://localhost:8000/retrieve
    python -m tests.test_tool.test_search_tool llm_episode --search_url http://localhost:8000/retrieve
    python -m tests.test_tool.test_search_tool evaluate --search_url http://localhost:8000/retrieve --n_examples 20 --verbose
    """
    fire.Fire(
        {
            "single_action": test_single_action,
            "episode": test_episode,
            "llm_episode": test_llm_episode,
            "evaluate": evaluate,
        }
    )


if __name__ == "__main__":
    main()
