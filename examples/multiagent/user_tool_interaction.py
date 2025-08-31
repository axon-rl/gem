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

"""Example of user-tool agent interaction in a multi-agent environment.

This example demonstrates how to create a multi-agent environment where
a user agent interacts with a tool agent to accomplish tasks.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Any, Dict, List, Optional, Tuple

from gem.multiagent.aec_env import AECEnv
from gem.multiagent.agent_selector import AgentSelector
from gem.multiagent.agents import UserAgent, ToolAgent, UserStrategy
from gem import register, make


class UserToolInteractionEnv(AECEnv):
    """Environment for user-tool agent interactions.
    
    This environment simulates a scenario where a user agent
    requests help from a tool agent to complete tasks.
    """
    
    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        user_strategy: UserStrategy = UserStrategy.SCRIPTED,
        max_turns: int = 10,
        **kwargs
    ):
        """Initialize the environment.
        
        Args:
            task: Task definition for the agents.
            user_strategy: Strategy for user agent behavior.
            max_turns: Maximum number of interaction turns.
            **kwargs: Additional configuration.
        """
        super().__init__()
        
        # Define agents
        self.possible_agents = ["user", "tool_agent"]
        self.agents = self.possible_agents.copy()
        
        # Initialize agent selector
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.selected
        
        # Create agent instances
        self.user_agent = UserAgent(
            agent_id="user",
            strategy=user_strategy,
            task=task,
            max_turns=max_turns,
            script=[
                "Hello, I need help with a search task.",
                "Can you search for information about Python programming?",
                "Thank you for your help!",
            ] if user_strategy == UserStrategy.SCRIPTED else None
        )
        
        # Create mock tools for demonstration
        mock_tools = {
            "search": self._mock_search_tool,
            "python": self._mock_python_tool,
            "respond": self._mock_respond_tool,
        }
        
        self.tool_agent = ToolAgent(
            agent_id="tool_agent",
            tools=mock_tools,
            strategy="react"
        )
        
        # Environment state
        self.task = task or {
            "instructions": "Help the user find information",
            "outputs": ["search results", "Python"]
        }
        self.max_turns = max_turns
        self.turn_count = 0
        self.conversation_history: List[Tuple[str, str]] = []
        
        # Define observation and action spaces (simplified for demo)
        self.observation_spaces = {
            "user": "Text observation space",
            "tool_agent": "Text observation space"
        }
        self.action_spaces = {
            "user": "Text action space",
            "tool_agent": "Text action space with tool calls"
        }
        
        # Initialize state tracking
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        # Store last action for generating observations
        self.last_action = None
        self.last_agent = None
    
    def _mock_search_tool(self, query: str = "") -> str:
        """Mock search tool for demonstration."""
        return f"Search results for '{query}': Found 10 relevant articles about Python programming."
    
    def _mock_python_tool(self, code: str = "") -> str:
        """Mock Python execution tool for demonstration."""
        return f"Executed Python code: {code[:50]}... Result: Success"
    
    def _mock_respond_tool(self, content: str = "") -> str:
        """Mock response tool for demonstration."""
        return content
    
    def observe(self, agent: str) -> str:
        """Get observation for specified agent.
        
        Args:
            agent: The agent to get observation for.
            
        Returns:
            The observation string.
        """
        if not self.conversation_history:
            return "Welcome! How can I help you today?"
        
        # Get the last message from the other agent
        if self.last_agent and self.last_agent != agent and self.last_action:
            return self.last_action
        
        # Return last message in conversation
        if self.conversation_history:
            last_agent, last_message = self.conversation_history[-1]
            if last_agent != agent:
                return last_message
        
        return "Waiting for response..."
    
    def step(self, action: Optional[str]) -> None:
        """Process action for current agent.
        
        Args:
            action: The action taken by the current agent.
        """
        if self.agent_selection is None:
            return
        
        current_agent = self.agent_selection
        
        # Handle dead step
        if self._was_dead_step(action):
            self._agent_selector.next()
            self.agent_selection = self._agent_selector.selected
            return
        
        # Process action based on agent type
        if current_agent == "user":
            # User provides input
            response = action or self.user_agent.act(self.observe("user"))
            self.last_action = response
            self.last_agent = "user"
            
            # Check for termination
            if "###STOP###" in response or "thank you" in response.lower():
                self.terminations["user"] = True
                self.terminations["tool_agent"] = True
                self.rewards["user"] = 1.0
                self.rewards["tool_agent"] = 1.0
        
        elif current_agent == "tool_agent":
            # Tool agent processes request
            observation = self.observe("tool_agent")
            response = action or self.tool_agent.act(observation)
            self.last_action = response
            self.last_agent = "tool_agent"
            
            # Give reward for successful tool execution
            if "successfully" in response.lower():
                self.rewards["tool_agent"] = 0.5
        
        # Store in conversation history
        if action:
            self.conversation_history.append((current_agent, action))
        
        # Update turn count
        self.turn_count += 1
        
        # Check for truncation
        if self.turn_count >= self.max_turns:
            self.truncations["user"] = True
            self.truncations["tool_agent"] = True
        
        # Accumulate rewards
        self._accumulate_rewards()
        
        # Move to next agent
        self._agent_selector.next()
        self.agent_selection = self._agent_selector.selected
        
        # Remove terminated agents
        if self.terminations[current_agent] or self.truncations[current_agent]:
            self._agent_selector.remove_agent(current_agent)
            self.agents.remove(current_agent)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Initial observation and info.
        """
        super().reset(seed)
        
        # Reset agents
        self.agents = self.possible_agents.copy()
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.selected
        
        # Reset agent instances
        self.user_agent.reset()
        self.tool_agent.reset()
        
        # Reset environment state
        self.turn_count = 0
        self.conversation_history = []
        self.last_action = None
        self.last_agent = None
        
        # Reset tracking dictionaries
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        # Get initial observation
        initial_obs = self.observe(self.agent_selection)
        
        return initial_obs, {}


def main():
    """Run the user-tool interaction example."""
    
    # Register the environment
    register(
        "UserToolInteraction-v0",
        entry_point=UserToolInteractionEnv,
        kwargs={
            "user_strategy": UserStrategy.SCRIPTED,
            "max_turns": 20
        }
    )
    
    # Create environment
    env = make("UserToolInteraction-v0")
    
    print("=== User-Tool Agent Interaction Demo ===\n")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation: {obs}\n")
    
    # Run interaction loop
    for agent in env.agent_iter(max_iter=40):
        observation, reward, terminated, truncated, info = env.last()
        
        if terminated or truncated:
            action = None
        else:
            # For demo, use agent's built-in policy
            if agent == "user":
                action = env.user_agent.act(observation)
            else:
                action = env.tool_agent.act(observation)
            
            print(f"{agent}: {action}")
        
        env.step(action)
        
        # Check if all agents are done
        if all(env.terminations.values()) or all(env.truncations.values()):
            break
    
    print("\n=== Interaction Complete ===")
    print(f"Final rewards: {env._cumulative_rewards}")
    print(f"Conversation turns: {env.turn_count}")


if __name__ == "__main__":
    main()