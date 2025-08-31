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

"""Example of collaborative question-answering with multiple agents.

This example demonstrates a parallel multi-agent environment where
multiple agents collaborate to answer questions by specializing in
different aspects of the task.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Any, Dict, List, Optional, Tuple
import random

from gem.multiagent.parallel_env import ParallelEnv
from gem.multiagent.agents import BaseAgent
from gem import register, make


class ResearcherAgent(BaseAgent):
    """Agent that researches information."""
    
    def act(self, observation: str) -> str:
        """Research and gather information."""
        if "question:" in observation.lower():
            # Extract question and research
            question = observation.split(":")[-1].strip()
            return f"Research findings for '{question}': The answer involves multiple factors that need validation."
        return "Researching the topic..."


class ValidatorAgent(BaseAgent):
    """Agent that validates information."""
    
    def act(self, observation: str) -> str:
        """Validate the researched information."""
        if "research findings" in observation.lower():
            return "Validation complete: The research findings are accurate and well-sourced."
        return "Validating information..."


class SynthesizerAgent(BaseAgent):
    """Agent that synthesizes final answer."""
    
    def act(self, observation: str) -> str:
        """Synthesize information into final answer."""
        if "validation complete" in observation.lower():
            return "Final Answer: Based on validated research, the comprehensive answer has been synthesized."
        return "Synthesizing answer..."


class CollaborativeQAEnv(ParallelEnv):
    """Parallel environment for collaborative question answering.
    
    Multiple agents work simultaneously to research, validate,
    and synthesize answers to questions.
    """
    
    def __init__(
        self,
        questions: Optional[List[str]] = None,
        max_rounds: int = 3,
        **kwargs
    ):
        """Initialize the environment.
        
        Args:
            questions: List of questions to answer.
            max_rounds: Maximum rounds of collaboration.
            **kwargs: Additional configuration.
        """
        super().__init__()
        
        # Define agents
        self.possible_agents = ["researcher", "validator", "synthesizer"]
        self.agents = self.possible_agents.copy()
        
        # Create agent instances
        self.agent_instances = {
            "researcher": ResearcherAgent("researcher"),
            "validator": ValidatorAgent("validator"),
            "synthesizer": SynthesizerAgent("synthesizer")
        }
        
        # Environment configuration
        self.questions = questions or [
            "What is machine learning?",
            "How do neural networks work?",
            "What is reinforcement learning?"
        ]
        self.current_question_idx = 0
        self.current_question = None
        self.max_rounds = max_rounds
        self.current_round = 0
        
        # Shared information board for agents
        self.shared_board: Dict[str, str] = {}
        
        # Define spaces
        self.observation_spaces = {agent: "Text" for agent in self.possible_agents}
        self.action_spaces = {agent: "Text" for agent in self.possible_agents}
        
        # Initialize tracking
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
    
    def step(
        self, actions: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """Execute parallel step with all agents acting simultaneously.
        
        Args:
            actions: Actions from all agents.
            
        Returns:
            Observations, rewards, terminations, truncations, and infos.
        """
        # Validate actions
        self._validate_actions(actions)
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        # Process each agent's action
        for agent, action in actions.items():
            # Update shared board
            self.shared_board[agent] = action
            
            # Calculate rewards based on collaboration
            if agent == "researcher" and "research findings" in action.lower():
                rewards[agent] = 0.3
            elif agent == "validator" and "validation complete" in action.lower():
                rewards[agent] = 0.3
            elif agent == "synthesizer" and "final answer" in action.lower():
                rewards[agent] = 0.4
                # Bonus rewards for all agents on completion
                rewards["researcher"] = rewards.get("researcher", 0) + 0.2
                rewards["validator"] = rewards.get("validator", 0) + 0.2
            else:
                rewards[agent] = 0.0
        
        # Increment round
        self.current_round += 1
        
        # Check termination conditions
        if "final answer" in self.shared_board.get("synthesizer", "").lower():
            # Task completed successfully
            for agent in self.agents:
                terminations[agent] = True
                infos[agent] = {"task_completed": True}
        elif self.current_round >= self.max_rounds:
            # Max rounds reached
            for agent in self.agents:
                truncations[agent] = True
                infos[agent] = {"max_rounds_reached": True}
        else:
            for agent in self.agents:
                terminations[agent] = False
                truncations[agent] = False
                infos[agent] = {}
        
        # Generate new observations based on shared board
        for agent in self.agents:
            # Each agent sees the shared board
            board_content = "\n".join([
                f"{a}: {msg}" for a, msg in self.shared_board.items()
                if a != agent  # Don't show agent's own message
            ])
            
            if board_content:
                observations[agent] = f"Question: {self.current_question}\n\nShared Information:\n{board_content}"
            else:
                observations[agent] = f"Question: {self.current_question}"
        
        # Store results
        self.terminations = terminations
        self.truncations = truncations
        self.rewards = rewards
        self.infos = infos
        
        # Remove terminated agents
        self._remove_dead_agents()
        
        return observations, rewards, terminations, truncations, infos
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """Reset the environment.
        
        Args:
            seed: Random seed.
            
        Returns:
            Initial observations and infos.
        """
        super().reset(seed)
        
        # Reset agents
        self.agents = self.possible_agents.copy()
        for agent in self.agent_instances.values():
            agent.reset()
        
        # Select new question
        if seed is not None:
            random.seed(seed)
            self.current_question_idx = random.randint(0, len(self.questions) - 1)
        else:
            self.current_question_idx = (self.current_question_idx + 1) % len(self.questions)
        
        self.current_question = self.questions[self.current_question_idx]
        self.current_round = 0
        self.shared_board = {}
        
        # Generate initial observations
        observations = {}
        infos = {}
        
        for agent in self.agents:
            observations[agent] = f"Question: {self.current_question}\nYour role: {agent}"
            infos[agent] = {"role": agent, "question_id": self.current_question_idx}
        
        return observations, infos


def main():
    """Run the collaborative QA example."""
    
    # Register the environment
    register(
        "CollaborativeQA-v0",
        entry_point=CollaborativeQAEnv,
        kwargs={"max_rounds": 5}
    )
    
    # Create environment
    env = make("CollaborativeQA-v0")
    
    print("=== Collaborative Question-Answering Demo ===\n")
    
    # Reset environment
    observations, infos = env.reset()
    
    print("Initial observations:")
    for agent, obs in observations.items():
        print(f"\n{agent}:\n{obs}\n")
    
    # Run collaboration rounds
    round_num = 0
    while env.agents:
        round_num += 1
        print(f"\n--- Round {round_num} ---")
        
        # Get actions from each agent (using their built-in logic)
        actions = {}
        for agent in env.agents:
            agent_instance = env.agent_instances[agent]
            action = agent_instance.act(observations[agent])
            actions[agent] = action
            print(f"{agent}: {action}")
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Print rewards
        print(f"\nRewards: {rewards}")
        
        # Check if done
        if not env.agents:
            break
    
    print("\n=== Collaboration Complete ===")
    print(f"Final shared board:\n{env.shared_board}")
    print(f"Total rounds: {env.current_round}")


if __name__ == "__main__":
    main()