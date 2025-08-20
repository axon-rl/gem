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

import random
from functools import partial
from typing import Optional

import fire

import gem
from gem.tools.mcp_tool import MCPTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

# Example actions using the sample Time MCP server tools in gem.tools.mcp_server.time_mcp
TEST_ACTIONS = [
    # Valid MCP tool calls
    '<tool_call><tool_name>current_time</tool_name><arguments>{"format": "YYYY-MM-DD", "timezone": "UTC"}</arguments></tool_call> ...',
    '<tool_call><tool_name>days_in_month</tool_name><arguments>{"date": "2025-03-01"}</arguments></tool_call> ...',
    '<think>I need to compute the relative time</think><tool_call><tool_name>relative_time</tool_name><arguments>{"time": "2025-12-31 23:50:00"}</arguments></tool_call> ...',
    '```<tool_call><tool_name>get_week_year</tool_name><arguments>{"date": "2025-03-23"}</arguments></tool_call> ... <tool_call><tool_name>convert_time</tool_name><arguments>{"sourceTimezone": "Asia/Shanghai", "targetTimezone": "Europe/London", "time": "2025-03-23 12:30:00"}</arguments></tool_call>``` ...',
    # Invalid/edge cases
    'Dummy action',
    '<tool_call><tool_name>non_existing_tool</tool_name><arguments>{"foo": "bar"}</arguments></tool_call> ...',
]


def test_single_action(mcp_url: str, env_name: str = "game:GuessTheNumber-v0"):
    """Run a few single-step tool calls against a running MCP server.

    Start the sample server (in another terminal) with for example:
      python -m gem.tools.mcp_server.time_mcp --transport streamable-http --host 127.0.0.1 --port 8081 --path /time-mcp
    Then run:
      python -m tests.test_tool.test_mcp_tool single_action --mcp_url http://127.0.0.1:8081/time-mcp
    """
    env = gem.make(env_name, max_turns=4)
    tool = MCPTool.from_url(mcp_url, validate_on_init=False)
    env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    obs, info = env.reset()

    print(f"Using MCP server: {mcp_url}")

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
            print(f"Error during MCP request: {e}")
            print("Observation: [Error occurred]")
            print("Continuing with next test...\n")


def test_episode(
    mcp_url: str,
    env_name: str = "qa:NaturalQuestions",
    tokenizer_name: str = "Qwen/Qwen3-0.6B-Base",
):
    from transformers import AutoTokenizer

    env = gem.make(env_name, max_turns=3, load_from_cache_file=False)
    policy = lambda _: random.choice(TEST_ACTIONS)
    tool = MCPTool.from_url(mcp_url, validate_on_init=False)

    print(f"Using MCP server: {mcp_url}")

    def run_episode_test(episode_name, wrapped_env, policy_func=None):
        print(f"\n{episode_name}")
        try:
            run_and_print_episode(wrapped_env, policy_func or policy)
        except Exception as e:
            print(f"Error during MCP episode: {e}")

    # Episode 1: Default observation
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    run_episode_test("EPISODE 1: DEFAULT OBSERVATION", wrapped_env)

    # Episode 2: Chat template observation
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_episode_test("EPISODE 2: CHAT TEMPLATE OBSERVATION", wrapped_env)

    # Episode 3: Chat template observation on reset
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat_on_reset"](
        wrapped_env, tokenizer=tokenizer
    )
    run_episode_test("EPISODE 3: CHAT TEMPLATE OBSERVATION ON RESET", wrapped_env)

    # Batch episode: Sync vectorized env
    num_envs = 3
    tool_env_wrapper = partial(ToolEnvWrapper, tools=[tool], max_tool_uses=3)
    chat_wrapper = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        [env_name] * num_envs,
        wrappers=[tool_env_wrapper, chat_wrapper],
        max_turns=3,
    )
    batch_policy = lambda _: [random.choice(TEST_ACTIONS) for _ in range(num_envs)]
    run_episode_test(
        "EPISODE 4: BATCH EPISODE SYNC VECTORIZED ENV", ta_vec_env, batch_policy
    )


def test_llm_episode(
    mcp_url: str,
    env_name: str = "eval:QaOpen",
    model_name: str = "Qwen/Qwen3-0.6B-Base",
):
    """Test episode with LLM observation and MCP tool."""
    from datasets import Dataset
    from vllm import LLM, SamplingParams

    env = gem.make(env_name, max_turns=3)
    # Create a simple single-question dataset encouraging the use of MCP tool
    question = (
        "Get the current date in UTC. Use the MCP tool with: "
        "<tool_call><tool_name>current_time</tool_name><arguments>{\"format\": \"YYYY-MM-DD\", \"timezone\": \"UTC\"}</arguments></tool_call>"
    )
    prompt = (
        "You must reason inside <think> and </think>. If you need external info, call the MCP tool as shown. "
        f"Question: {question}\n"
    )
    # No strict answer check here; this is just to exercise the tool flow
    answer = ""
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

    tool = MCPTool.from_url(mcp_url, validate_on_init=False)

    print(f"Using MCP server: {mcp_url}")

    def run_episode_test(episode_name, wrapped_env, policy_func, **kwargs):
        print(f"\n{episode_name}")
        try:
            run_and_print_episode(wrapped_env, policy_func, **kwargs)
        except Exception as e:
            print(f"Error during MCP request episode: {e}")

    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=3)
    wrapped_env = WRAPPER_FACTORY["concat_chat"](wrapped_env, tokenizer=tokenizer)
    run_episode_test("EPISODE 1: CHAT TEMPLATE OBSERVATION", wrapped_env, policy)


def main():
    """Run with:
    # Start the sample Time MCP server (in another terminal):
    #   python -m gem.tools.mcp_server.time_mcp
    #
    # Then run these:
    #   python -m tests.test_tool.test_mcp_tool single_action --mcp_url http://127.0.0.1:8081/time-mcp
    #   python -m tests.test_tool.test_mcp_tool episode --mcp_url http://127.0.0.1:8081/time-mcp
    #   python -m tests.test_tool.test_mcp_tool llm_episode --mcp_url http://127.0.0.1:8081/time-mcp
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
