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

from typing import Any, Dict, Optional, Tuple

from gem import make, register
from gem.multiagent import MultiAgentEnv, AgentSelector
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.search_tool import SearchTool


class ConversationEnv(MultiAgentEnv):
    
    def __init__(self):
        super().__init__(simultaneous=False)
        
        self.possible_agents = ["user", "assistant"]
        
        self.python_tool = PythonCodeTool()
        self.search_tool = SearchTool(search_url=None)
        
        self.step_count = 0
        self.max_steps = 10
        self.conversation_history = []
        self.last_action = None
    
    def observe(self, agent: str) -> str:
        if not self.conversation_history:
            if agent == "user":
                return "Welcome! You can ask questions or request code execution. Type 'exit' to end."
            else:
                return "User hasn't said anything yet. Waiting for user input."
        
        history_str = "\n".join(self.conversation_history[-5:])
        
        if agent == "user":
            return f"Conversation:\n{history_str}\n\nYour turn (type 'exit' to end):"
        else:
            return f"Conversation:\n{history_str}\n\nProvide a helpful response:"
    
    def _step_sequential(self, action: str) -> Tuple[str, float, bool, bool, dict]:
        current = self.current_agent
        
        self.conversation_history.append(f"{current}: {action}")
        self.last_action = action
        
        reward = 0.0
        info = {}
        
        if current == "assistant" and action.startswith("execute:"):
            code = action[8:].strip()
            try:
                is_valid, has_error, result, _ = self.python_tool.execute_action(code)
                if is_valid and not has_error:
                    self.conversation_history.append(f"[Code Result]: {result}")
                    reward = 1.0
                    info["tool_result"] = result
                else:
                    self.conversation_history.append(f"[Code Error]: {result}")
                    reward = -0.5
                    info["tool_error"] = result
            except Exception as e:
                self.conversation_history.append(f"[Error]: {str(e)}")
                reward = -1.0
                info["error"] = str(e)
        
        elif current == "assistant" and action.startswith("search:"):
            query = action[7:].strip()
            try:
                result = self.search_tool.execute_action(query)
                self.conversation_history.append(f"[Search Result]: {result}")
                reward = 0.5
                info["search_result"] = result
            except Exception as e:
                self.conversation_history.append(f"[Search Error]: {str(e)}")
                reward = -0.5
                info["search_error"] = str(e)
        
        elif len(action.split()) > 3:
            reward = 0.1
        
        terminated = action.lower() in ["exit", "quit", "goodbye"]
        
        if self._agent_selector:
            self._agent_selector.next()
            self.agent_selection = self._agent_selector.selected
            
            if self._agent_selector.is_first():
                self.step_count += 1
                if self.step_count >= self.max_steps:
                    for agent in self.agents:
                        self.truncations[agent] = True
        
        if terminated:
            for agent in self.agents:
                self.terminations[agent] = True
        
        self.rewards[current] = reward
        self._cumulative_rewards[current] = self._cumulative_rewards.get(current, 0.0) + reward
        
        next_obs = self.observe(self.current_agent) if self.current_agent else ""
        
        return next_obs, reward, self.terminations[current], self.truncations[current], info
    
    def _step_simultaneous(self, actions: Dict[str, str]) -> Tuple:
        raise NotImplementedError("ConversationEnv is a sequential environment")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        
        self.conversation_history = []
        self.step_count = 0
        self.last_action = None
        
        return self.observe(self.current_agent), {}


def simulate_user_agent(observation: str) -> str:
    prompts = [
        "Can you calculate the sum of 1 to 100?",
        "execute: sum(range(1, 101))",
        "Great! Now search for information about Python generators",
        "search: Python generators yield",
        "exit",
    ]
    
    import random
    
    if "Welcome!" in observation:
        return prompts[0]
    elif "sum of 1 to 100" in str(observation):
        return "That's interesting! Can you show me the actual calculation?"
    elif "actual calculation" in str(observation):
        return prompts[1]
    elif "[Code Result]" in str(observation) and "5050" in str(observation):
        return prompts[2]
    elif "Python generators" in str(observation):
        return prompts[3]
    elif "[Search" in str(observation):
        return prompts[4]
    
    return random.choice(["Tell me more", "Interesting!", "What else can you do?", "exit"])


def simulate_assistant_agent(observation: str) -> str:
    if "hasn't said anything" in observation:
        return "Hello! I'm ready to help you with questions or code execution."
    elif "sum of 1 to 100" in observation:
        return "The sum of numbers from 1 to 100 is 5050. This can be calculated using the formula n*(n+1)/2 where n=100."
    elif "actual calculation" in observation:
        return "execute: print(f'Sum of 1 to 100: {sum(range(1, 101))}')"
    elif "Python generators" in observation:
        return "search: Python generators yield iteration memory efficient"
    elif "exit" in observation or "goodbye" in observation:
        return "Goodbye! Have a great day!"
    
    return "I can help you with calculations and searches. What would you like to know?"


def main():
    register("ConversationEnv-v0", entry_point="conversation:ConversationEnv")
    
    env = make("ConversationEnv-v0")
    obs, info = env.reset()
    
    print("=" * 50)
    print("Starting Conversation Environment")
    print("=" * 50)
    
    for i, agent in enumerate(env.agent_iter(max_iter=20)):
        obs, reward, terminated, truncated, info = env.last()
        
        print(f"\n[Step {i}] Agent: {agent}")
        print(f"Observation: {obs}")
        
        if terminated or truncated:
            action = None
            print("Conversation ended")
        else:
            if agent == "user":
                action = simulate_user_agent(obs)
            else:
                action = simulate_assistant_agent(obs)
            
            print(f"Action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.2f}")
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 50)
    print("Conversation Complete")
    print(f"Final cumulative rewards: {env._cumulative_rewards}")
    print("=" * 50)


if __name__ == "__main__":
    main()