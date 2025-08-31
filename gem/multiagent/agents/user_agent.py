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

"""User agent implementation for simulating user interactions."""

import random
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from gem.multiagent.agents.base_agent import BaseAgent


class UserStrategy(Enum):
    """Strategies for user simulation."""
    SCRIPTED = "scripted"
    LLM = "llm"
    HUMAN = "human"
    RANDOM = "random"
    VERIFY = "verify"
    REFLECTION = "reflection"


class UserAgent(BaseAgent):
    """Agent that simulates user interactions.
    
    This agent can use different strategies to generate user-like
    responses and interactions in multi-agent environments.
    """
    
    def __init__(
        self,
        agent_id: str = "user",
        strategy: UserStrategy = UserStrategy.SCRIPTED,
        task: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None,
        **kwargs
    ):
        """Initialize the user agent.
        
        Args:
            agent_id: Unique identifier for this agent.
            strategy: The strategy to use for generating responses.
            task: Task definition including instructions and expected behavior.
            model: Optional language model for LLM-based strategies.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(agent_id, **kwargs)
        self.strategy = strategy
        self.task = task or {}
        self.model = model
        self.conversation_state = "initial"
        self.turn_count = 0
        self.max_turns = kwargs.get("max_turns", 10)
        self.termination_phrase = kwargs.get("termination_phrase", "###STOP###")
        
        # For scripted strategy
        self.script = kwargs.get("script", [])
        self.script_index = 0
        
        # For verify strategy
        self.verification_attempts = 0
        self.max_verification_attempts = kwargs.get("max_verification_attempts", 3)
    
    def act(self, observation: str) -> str:
        """Generate a user action/response based on the observation.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            The user's response.
        """
        self.observe(observation)
        self.turn_count += 1
        
        # Check if we should terminate
        if self._should_terminate(observation):
            return self.termination_phrase
        
        # Generate response based on strategy
        if self.strategy == UserStrategy.SCRIPTED:
            response = self._scripted_response()
        elif self.strategy == UserStrategy.LLM:
            response = self._llm_response(observation)
        elif self.strategy == UserStrategy.HUMAN:
            response = self._human_response(observation)
        elif self.strategy == UserStrategy.RANDOM:
            response = self._random_response()
        elif self.strategy == UserStrategy.VERIFY:
            response = self._verify_response(observation)
        elif self.strategy == UserStrategy.REFLECTION:
            response = self._reflection_response(observation)
        else:
            response = "I don't understand."
        
        self.action_history.append(response)
        return response
    
    def _should_terminate(self, observation: str) -> bool:
        """Check if the conversation should terminate.
        
        Args:
            observation: The current observation.
            
        Returns:
            True if should terminate, False otherwise.
        """
        # Check turn limit
        if self.turn_count >= self.max_turns:
            return True
        
        # Check if task is completed
        if self._is_task_complete(observation):
            return True
        
        # Check for termination keywords
        termination_keywords = ["goodbye", "thank you", "that's all", "done", "finished"]
        observation_lower = observation.lower()
        if any(keyword in observation_lower for keyword in termination_keywords):
            return True
        
        return False
    
    def _is_task_complete(self, observation: str) -> bool:
        """Check if the task is complete based on observation.
        
        Args:
            observation: The current observation.
            
        Returns:
            True if task is complete, False otherwise.
        """
        if not self.task:
            return False
        
        # Check if required outputs are in the observation
        required_outputs = self.task.get("outputs", [])
        if required_outputs:
            return all(output in observation for output in required_outputs)
        
        return False
    
    def _scripted_response(self) -> str:
        """Generate a response from a predefined script.
        
        Returns:
            The scripted response.
        """
        if not self.script or self.script_index >= len(self.script):
            return self.termination_phrase
        
        response = self.script[self.script_index]
        self.script_index += 1
        return response
    
    def _llm_response(self, observation: str) -> str:
        """Generate a response using a language model.
        
        Args:
            observation: The current observation.
            
        Returns:
            The LLM-generated response.
        """
        if not self.model:
            return "I need more information."
        
        # Build prompt from conversation history
        prompt = self._build_llm_prompt(observation)
        
        # This is a placeholder - actual implementation would call the model
        # response = self.model.generate(prompt)
        response = f"Based on '{observation}', I understand. Please continue."
        
        return response
    
    def _human_response(self, observation: str) -> str:
        """Get response from a human user.
        
        Args:
            observation: The current observation.
            
        Returns:
            The human's response.
        """
        # In a real implementation, this would get input from a human
        # For now, return a placeholder
        return "Human: Please provide more details."
    
    def _random_response(self) -> str:
        """Generate a random response.
        
        Returns:
            A random response.
        """
        responses = [
            "Can you help me with this?",
            "I need more information.",
            "That's interesting, tell me more.",
            "Can you clarify that?",
            "What are my options?",
            "Please proceed.",
            "I understand.",
        ]
        return random.choice(responses)
    
    def _verify_response(self, observation: str) -> str:
        """Generate a response with verification strategy.
        
        Args:
            observation: The current observation.
            
        Returns:
            The verification response.
        """
        self.verification_attempts += 1
        
        if self.verification_attempts > self.max_verification_attempts:
            return "I've verified the information. Thank you."
        
        # Ask for verification
        verification_questions = [
            "Can you confirm that?",
            "Are you sure about this?",
            "Please verify this information.",
            "Can you double-check that?",
        ]
        
        return random.choice(verification_questions)
    
    def _reflection_response(self, observation: str) -> str:
        """Generate a response with reflection on previous interactions.
        
        Args:
            observation: The current observation.
            
        Returns:
            The reflection response.
        """
        # Reflect on conversation history
        if len(self.observation_history) < 2:
            return "Let me think about this."
        
        # Generate reflection based on history
        prev_observation = self.observation_history[-2] if len(self.observation_history) > 1 else ""
        
        if prev_observation and observation:
            return f"Based on what you said earlier about '{prev_observation[:50]}...', I now understand."
        
        return "Let me reconsider based on our conversation."
    
    def _build_llm_prompt(self, observation: str) -> str:
        """Build a prompt for the language model.
        
        Args:
            observation: The current observation.
            
        Returns:
            The formatted prompt.
        """
        prompt_parts = []
        
        # Add task instructions if available
        if self.task.get("instructions"):
            prompt_parts.append(f"Task: {self.task['instructions']}")
        
        # Add conversation history
        prompt_parts.append("Conversation history:")
        for i, obs in enumerate(self.observation_history[-5:]):  # Last 5 turns
            prompt_parts.append(f"Turn {i+1}: {obs}")
        
        # Add current observation
        prompt_parts.append(f"Current: {observation}")
        prompt_parts.append("Your response:")
        
        return "\n".join(prompt_parts)
    
    def reset(self) -> None:
        """Reset the agent's state for a new episode."""
        super().reset()
        self.conversation_state = "initial"
        self.turn_count = 0
        self.script_index = 0
        self.verification_attempts = 0