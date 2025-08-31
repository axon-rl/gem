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

"""Base agent class for multi-agent environments."""

import abc
from typing import Any, Dict, List, Optional


class BaseAgent(abc.ABC):
    """Base class for agents in multi-agent environments.
    
    This class defines the interface that all agents must implement
    to participate in multi-agent environments.
    """
    
    def __init__(self, agent_id: str, **kwargs):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent.
            **kwargs: Additional configuration parameters.
        """
        self.agent_id = agent_id
        self.messages: List[Dict[str, str]] = []
        self.observation_history: List[str] = []
        self.action_history: List[str] = []
        self.reward_history: List[float] = []
    
    @abc.abstractmethod
    def act(self, observation: str) -> str:
        """Generate an action based on the current observation.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            The action to take.
        """
        raise NotImplementedError
    
    def observe(self, observation: str) -> None:
        """Process an observation from the environment.
        
        Args:
            observation: The observation to process.
        """
        self.observation_history.append(observation)
    
    def update(self, action: str, reward: float, done: bool) -> None:
        """Update agent state after taking an action.
        
        Args:
            action: The action that was taken.
            reward: The reward received.
            done: Whether the episode is done.
        """
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def reset(self) -> None:
        """Reset the agent's state for a new episode."""
        self.messages = []
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent.
        
        Returns:
            Dictionary containing agent state information.
        """
        return {
            "agent_id": self.agent_id,
            "messages": self.messages.copy(),
            "observation_history": self.observation_history.copy(),
            "action_history": self.action_history.copy(),
            "reward_history": self.reward_history.copy(),
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the agent's state.
        
        Args:
            state: Dictionary containing agent state information.
        """
        self.messages = state.get("messages", []).copy()
        self.observation_history = state.get("observation_history", []).copy()
        self.action_history = state.get("action_history", []).copy()
        self.reward_history = state.get("reward_history", []).copy()
    
    def send_message(self, receiver: str, content: str, metadata: Optional[Dict] = None) -> Dict:
        """Send a message to another agent.
        
        Args:
            receiver: The ID of the receiving agent.
            content: The message content.
            metadata: Optional metadata to attach to the message.
            
        Returns:
            The message object.
        """
        message = {
            "sender": self.agent_id,
            "receiver": receiver,
            "content": content,
            "metadata": metadata or {}
        }
        self.messages.append(message)
        return message
    
    def receive_message(self, message: Dict) -> None:
        """Receive a message from another agent.
        
        Args:
            message: The message object.
        """
        self.messages.append(message)
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(agent_id='{self.agent_id}')"