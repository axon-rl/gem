#!/usr/bin/env python3
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


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Any, Dict, Optional, Tuple

from gem import make, register
from gem.multiagent.aec_env import AECEnv
from gem.multiagent.multi_agent_env import MultiAgentEnv
from gem.multiagent.utils import AgentSelector
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.search_tool import SearchTool


class ConversationEnv(AECEnv):
    def __init__(self):
        super().__init__()

        self.possible_agents = ["user", "assistant"]
        self.agents = self.possible_agents.copy()

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.selected

        self.python_tool = PythonCodeTool()
        self.search_tool = SearchTool(search_url=None)

        self.step_count = 0
        self.max_steps = 10
        self.conversation_history = []

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        self.last_action = None
        self.last_agent = None

    def observe(self, agent: str) -> str:
        if not self.conversation_history:
            return "Welcome! I can help you with Python code execution and search."

        if self.last_agent and self.last_agent != agent and self.last_action:
            return self.last_action

        return "Waiting for response..."

    def step(self, action: Optional[str]) -> None:
        if self.agent_selection is None or action is None:
            return

        current_agent = self.agent_selection

        if current_agent == "user":
            self.last_action = action
            self.last_agent = "user"

            if "goodbye" in action.lower() or "exit" in action.lower():
                self.terminations["user"] = True
                self.terminations["assistant"] = True
                self.rewards["user"] = 1.0
                self.rewards["assistant"] = 1.0

        elif current_agent == "assistant":
            response = self._process_assistant_action(action)
            self.last_action = response
            self.last_agent = "assistant"

            if "Result:" in response:
                self.rewards["assistant"] = 0.5

        self.conversation_history.append((current_agent, action))
        self.step_count += 1

        if self.step_count >= self.max_steps:
            self.truncations["user"] = True
            self.truncations["assistant"] = True

        self._accumulate_rewards()

        self._agent_selector.next()
        self.agent_selection = self._agent_selector.selected

    def _process_assistant_action(self, action: str) -> str:
        response = action

        if "<python>" in action or "```python" in action:
            try:
                is_valid, _, observation, _ = self.python_tool.execute_action(action)
                if is_valid:
                    response = f"Python execution result: {observation}"
                else:
                    response = "No valid Python code found."
            except Exception as e:
                response = f"Error executing Python: {str(e)}"

        elif "<search>" in action:
            try:
                is_valid, _, observation, _ = self.search_tool.execute_action(action)
                if is_valid:
                    response = observation
                else:
                    response = "No valid search query found."
            except Exception as e:
                response = f"Error with search: {str(e)}"

        return response

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        MultiAgentEnv.reset(self, seed)

        self.agents = self.possible_agents.copy()
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.selected

        self.step_count = 0
        self.conversation_history = []
        self.last_action = None
        self.last_agent = None

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        initial_obs = self.observe(self.agent_selection)
        return initial_obs, {}


def main():
    register(
        "Conversation-v0",
        entry_point=ConversationEnv,
    )

    env = make("Conversation-v0")

    print("=== User-Assistant Conversation Example ===")
    print("Demonstrating turn-based dialogue with tool use\n")

    obs, _ = env.reset()
    print(f"[{env.agent_selection}]: {obs}\n")

    scripted_interaction = [
        ("user", "Can you calculate the factorial of 5?"),
        (
            "assistant",
            "<python>import math\nprint(f'Factorial of 5 is: {math.factorial(5)}')</python>",
        ),
        ("user", "Great! Now search for Python tutorials"),
        ("assistant", "<search>Python programming tutorials</search>"),
        ("user", "Thank you, goodbye!"),
        ("assistant", "You're welcome! Have a great day!"),
    ]

    for expected_agent, action in scripted_interaction:
        if env.agent_selection != expected_agent:
            env.step(None)
            continue

        print(f"[{expected_agent}]: {action}")
        env.step(action)

        if not all(env.terminations.values()):
            obs, _, _, _, _ = env.last()
            if obs and "Result:" in obs:
                print(f"[System]: {obs}")
        print()

        if all(env.terminations.values()) or all(env.truncations.values()):
            break

    print("=== Conversation Complete ===")
    print(f"Total steps: {env.step_count}")
    print(f"Final rewards: {env._cumulative_rewards}")


if __name__ == "__main__":
    main()
