import random
from functools import partial
from typing import List
from unittest.mock import Mock, patch

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
    """<search>Python list comprehension examples</search> ...""",
    """```<search>First query</search> ... <search>Second query</search>``` ...""",
]

MOCK_SEARCH_RESPONSE = {
    "result": [
        [
            {"document": {"contents": "Paris\nThe capital of France is Paris."}},
            {"document": {"contents": "France\nFrance is a country in Europe."}},
        ]
    ]
}


def mock_post(*args, **kwargs):
    mock_resp = Mock()
    mock_resp.json.return_value = MOCK_SEARCH_RESPONSE
    mock_resp.raise_for_status = lambda: None
    return mock_resp


def _should_use_real_requests(search_url: str) -> bool:
    """Determine if we should use real requests based on the search_url."""
    return (
        search_url
        and search_url != "http://dummy-search-url"
        and not search_url.startswith("http://dummy")
    )


def test_single_action(
    env_name: str = "ta:GuessTheNumber-v0", search_url: str = "http://dummy-search-url"
):
    env = gem.make(env_name, max_turns=3)
    tool = SearchTool(search_url=search_url, topk=2)
    env = ToolEnvWrapper(env, tools=[tool])
    obs, info = env.reset()

    use_real_requests = _should_use_real_requests(search_url)
    print(
        f"Using {'real' if use_real_requests else 'mocked'} requests with URL: {search_url}"
    )

    if use_real_requests:
        # Send real requests
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
    else:
        # Use mocked requests
        with patch("requests.post", side_effect=mock_post):
            for i, test_action in enumerate(TEST_ACTIONS):
                print(f"------ Test {i} ------")
                print(f"Action: {test_action!r}")
                obs, reward, terminated, truncated, info = env.step(test_action)
                print(f"Observation: {obs}")
                print(f"Reward: {reward}")
                print(f"Terminated: {terminated}")
                print(f"Truncated: {truncated}")
                print(f"Info: {info}\n")


def test_episode(
    env_name: str = "ta:GuessTheNumber-v0", search_url: str = "http://dummy-search-url"
):
    env = gem.make(env_name, max_turns=3)
    policy = lambda _: random.choice(TEST_ACTIONS)
    tool = SearchTool(search_url=search_url, topk=2)

    use_real_requests = _should_use_real_requests(search_url)
    print(
        f"Using {'real' if use_real_requests else 'mocked'} requests with URL: {search_url}"
    )

    def run_episode_test(episode_name, wrapped_env, policy_func=None):
        print(f"\n{episode_name}")
        if use_real_requests:
            try:
                run_and_print_episode(wrapped_env, policy_func or policy)
            except Exception as e:
                print(f"Error during real request episode: {e}")
        else:
            with patch("requests.post", side_effect=mock_post):
                run_and_print_episode(wrapped_env, policy_func or policy)

    # Episode 1: Default observation
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_episode_test("EPISODE 1: DEFAULT OBSERVATION", wrapped_env)

    # Episode 3: Chat template observation
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_episode_test("EPISODE 3: CHAT TEMPLATE OBSERVATION", wrapped_env)

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
    env_name: str = "ta:GuessTheNumber-v0",
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    search_url: str = "http://dummy-search-url",
):
    """Test episode with LLM observation and Search tool."""
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
        action = response[0].outputs[0].text
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
        actions = [r.outputs[0].text for r in response]
        return actions

    tool = SearchTool(search_url=search_url, topk=2)

    use_real_requests = _should_use_real_requests(search_url)
    print(
        f"Using {'real' if use_real_requests else 'mocked'} requests with URL: {search_url}"
    )

    def run_episode_test(episode_name, wrapped_env, policy_func, **kwargs):
        print(f"\n{episode_name}")
        if use_real_requests:
            try:
                run_and_print_episode(wrapped_env, policy_func, **kwargs)
            except Exception as e:
                print(f"Error during real request episode: {e}")
        else:
            with patch("requests.post", side_effect=mock_post):
                run_and_print_episode(wrapped_env, policy_func, **kwargs)

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_episode_test("EPISODE 1: DEFAULT OBSERVATION", wrapped_env, policy)

    print("\n" * 5, "EPISODE 2: CHAT TEMPLATE OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_episode_test("EPISODE 2: CHAT TEMPLATE OBSERVATION", wrapped_env, policy)

    print("\n" * 5, "BATCH EPISODE 1: SYNC VECTORIZED ENV")
    num_envs = 2
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=False,
        max_turns=3,
    )
    run_episode_test(
        "BATCH EPISODE 1: SYNC VECTORIZED ENV",
        ta_vec_env,
        policy_func=batch_policy,
        ignore_done=True,
        max_steps=2,
    )

    print("\n" * 5, "BATCH EPISODE 2: ASYNC VECTORIZED ENV")
    num_envs = 2
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        async_mode=True,
        max_turns=3,
    )
    run_episode_test(
        "BATCH EPISODE 2: ASYNC VECTORIZED ENV",
        ta_vec_env,
        policy_func=batch_policy,
        ignore_done=True,
        max_steps=2,
    )


def main():
    """Run with:
    python -m tests.test_tool.test_search_tool single_action

    # To test with real search server:
    python -m tests.test_tool.test_search_tool single_action --search_url http://localhost:8000/retrieve
    python -m tests.test_tool.test_search_tool episode --search_url http://localhost:8000/retrieve
    python -m tests.test_tool.test_search_tool llm_episode --search_url http://localhost:8000/retrieve
    """
    fire.Fire(
        {
            "single_action": test_single_action,
            "episode": test_episode,
            "llm_episode": test_llm_episode,
        }
    )


if __name__ == "__main__":
    main()
