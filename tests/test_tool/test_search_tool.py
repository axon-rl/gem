import random
from functools import partial
from unittest.mock import Mock, patch

import fire
from transformers import AutoTokenizer

import gem
from gem.envs.multi_turn import MultiTurnEnv
from gem.tools.search_tool import SearchTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.utils.debug import run_and_print_episode
from gem.wrappers.stateful_observation import (ChatTemplatedObservation,
                                               ConcatenatedObservation)

TEST_ACTIONS = [
    """<search>What is the capital of France?</search> ...""",
    """Dummy action""",
    """<search>Python list comprehension examples</search> ...""",
    """```<search>First query</search> ... <search>Second query</search>``` ...""",
]

MOCK_SEARCH_RESPONSE = {
    'result': [[
        {'document': {'contents': 'Paris\nThe capital of France is Paris.'}},
        {'document': {'contents': 'France\nFrance is a country in Europe.'}},
    ]]
}

def mock_post(*args, **kwargs):
    mock_resp = Mock()
    mock_resp.json.return_value = MOCK_SEARCH_RESPONSE
    mock_resp.raise_for_status = lambda: None
    return mock_resp


def test_single_action(env_name: str = "ta:GuessTheNumber-v0"):
    env: MultiTurnEnv = gem.make(env_name, max_turns=3)
    tool = SearchTool(search_url="http://dummy-search-url")
    env = ToolEnvWrapper(env, tools=[tool])
    obs, info = env.reset()
    with patch('requests.post', side_effect=mock_post):
        for i, test_action in enumerate(TEST_ACTIONS):
            print(f"------ Test {i} ------")
            print(f"Action: {test_action!r}")
            obs, reward, terminated, truncated, info = env.step(test_action)
            print(f"Observation: {obs}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info}\n")

def test_episode(env_name: str = "ta:GuessTheNumber-v0"):
    env: MultiTurnEnv = gem.make(env_name, max_turns=3)
    policy = lambda _: random.choice(TEST_ACTIONS)
    tool = SearchTool(search_url="http://dummy-search-url")

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    with patch('requests.post', side_effect=mock_post):
        run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 2: CONCATENATED OBSERVATION")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = ConcatenatedObservation(wrapped_env)
    with patch('requests.post', side_effect=mock_post):
        run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 3: CHAT TEMPLATE OBSERVATION")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = ChatTemplatedObservation(wrapped_env, tokenizer)
    with patch('requests.post', side_effect=mock_post):
        run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "BATCH EPISODE: SYNC VECTORIZED ENV")
    num_envs = 3
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[tool_env_wrapper, ConcatenatedObservation],
        max_turns=3,
    )
    with patch('requests.post', side_effect=mock_post):
        run_and_print_episode(
            ta_vec_env,
            policy=lambda _: [random.choice([TEST_ACTIONS[2]]) for _ in range(num_envs)],
            ignore_done=True,
            max_steps=5,
        )

def main():
    """Run with:
    python -m tests.test_tool.test_search_tool single_action --env_name ta:GuessTheNumber-v0
    python -m tests.test_tool.test_search_tool episode --env_name ta:GuessTheNumber-v0
    """
    fire.Fire(
        {
            "single_action": test_single_action,
            "episode": test_episode,
        }
    )


if __name__ == "__main__":
    main()
