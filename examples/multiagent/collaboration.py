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
from gem.multiagent import MultiAgentEnv
from gem.tools.python_code_tool import PythonCodeTool


class CollaborationEnv(MultiAgentEnv):
    
    def __init__(self, max_rounds: int = 3):
        super().__init__(simultaneous=True)
        
        self.possible_agents = ["researcher", "analyst", "reviewer"]
        
        self.max_rounds = max_rounds
        self.round_count = 0
        
        # Use a different name to avoid conflict with parent's shared_memory
        self.agent_shared_memory = {}
        self.task = "Analyze data and produce insights collaboratively"
        
        self.python_tool = PythonCodeTool()
        self.analysis_results = []
    
    def observe(self, agent: str) -> str:
        other_agents_info = []
        for other_agent, info in self.agent_shared_memory.items():
            if other_agent != agent:
                other_agents_info.append(f"{other_agent}: {info}")
        
        analysis_info = ""
        if self.analysis_results:
            analysis_info = "\nPrevious analysis results:\n" + "\n".join(
                self.analysis_results[-3:]
            )
        
        if other_agents_info:
            return f"Task: {self.task}\nRound {self.round_count}\nOther agents' actions:\n" + "\n".join(
                other_agents_info
            ) + analysis_info
        else:
            return f"Task: {self.task}\nRound {self.round_count}\nYou are the first to act this round." + analysis_info
    
    def _step_simultaneous(self, actions: Dict[str, str]) -> Tuple:
        self._validate_actions(actions)
        
        observations = {}
        rewards = {}
        
        for agent, action in actions.items():
            self.agent_shared_memory[agent] = action
            
            if agent == "analyst" and action.startswith("analyze:"):
                code = action[8:].strip()
                try:
                    is_valid, has_error, result, _ = self.python_tool.execute_action(code)
                    if is_valid and not has_error:
                        self.analysis_results.append(f"[Analysis]: {result}")
                        rewards[agent] = 1.5
                        self.infos[agent]["analysis_result"] = result
                    else:
                        self.analysis_results.append(f"[Analysis Error]: {result}")
                        rewards[agent] = -0.5
                        self.infos[agent]["analysis_error"] = result
                except Exception as e:
                    self.analysis_results.append(f"[Error]: {str(e)}")
                    rewards[agent] = -1.0
                    self.infos[agent]["error"] = str(e)
            
            elif "complete" in action.lower():
                rewards[agent] = 2.0
            
            elif "insight" in action.lower() or "recommend" in action.lower():
                rewards[agent] = 1.0
            
            else:
                rewards[agent] = 0.1
            
            if "complete" in action.lower() or "finish" in action.lower():
                self.terminations[agent] = True
        
        self.round_count += 1
        
        if self.round_count >= self.max_rounds:
            for agent in self.agents:
                self.truncations[agent] = True
        
        if all(self.terminations.values()):
            for agent in self.agents:
                self.terminations[agent] = True
        
        for agent in self.agents:
            observations[agent] = self.observe(agent)
            self.rewards[agent] = rewards.get(agent, 0.0)
            self._cumulative_rewards[agent] = self._cumulative_rewards.get(agent, 0.0) + self.rewards[agent]
        
        self._remove_dead_agents()
        
        return observations, self.rewards, self.terminations, self.truncations, self.infos
    
    def _step_sequential(self, action: str):
        raise NotImplementedError("CollaborationEnv is a simultaneous environment")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        # Call parent reset first
        _, _ = super().reset(seed)
        
        # Override shared_memory to be a dict for this env
        self.round_count = 0
        self.agent_shared_memory = {}
        self.analysis_results = []
        
        observations = {}
        for agent in self.agents:
            if agent == "researcher":
                observations[agent] = f"Task: {self.task}\nYou are the researcher. Identify data patterns and questions to explore."
            elif agent == "analyst":
                observations[agent] = f"Task: {self.task}\nYou are the analyst. Use 'analyze: <code>' to run Python analysis."
            else:
                observations[agent] = f"Task: {self.task}\nYou are the reviewer. Validate findings and provide recommendations."
        
        return observations, self.infos


def simulate_researcher(observation: str, round_num: int) -> str:
    if round_num == 1:
        return "Identifying key metrics: revenue trends, customer segments, and seasonal patterns"
    elif round_num == 2:
        return "Based on the analysis, I see a 15% growth opportunity in Q3. Let's explore customer retention metrics."
    else:
        return "Research complete - we have actionable insights on growth opportunities"


def simulate_analyst(observation: str, round_num: int) -> str:
    if round_num == 1:
        return "analyze: import numpy as np; data = np.random.randn(100); print(f'Mean: {data.mean():.2f}, Std: {data.std():.2f}')"
    elif round_num == 2:
        return "analyze: growth = [100, 115, 132, 148]; print(f'Q3 Growth Rate: {(growth[-1]/growth[-2] - 1)*100:.1f}%')"
    else:
        return "Analysis complete - all metrics computed and validated"


def simulate_reviewer(observation: str, round_num: int) -> str:
    if round_num == 1:
        return "Reviewing initial data approach. Recommend focusing on customer lifetime value."
    elif round_num == 2:
        if "[Analysis]" in observation:
            return "Good insights from the analysis. The growth rate aligns with market trends."
        else:
            return "Need more quantitative analysis to support the findings."
    else:
        return "Review complete - recommendations: 1) Focus on Q3 initiatives 2) Monitor retention metrics"


def main():
    register("CollaborationEnv-v0", entry_point="collaboration:CollaborationEnv")
    
    env = make("CollaborationEnv-v0")
    
    print("=" * 50)
    print("Multi-Agent Collaboration Example")
    print("Demonstrating simultaneous agent collaboration")
    print("=" * 50)
    
    observations, _ = env.reset()
    
    print("\nInitial observations:")
    for agent, obs in observations.items():
        print(f"\n[{agent}]:\n{obs}")
    
    round_num = 1
    while env.agents and round_num <= 3:
        print(f"\n{'=' * 20} Round {round_num} {'=' * 20}")
        
        actions = {}
        for agent in env.agents:
            if agent == "researcher":
                actions[agent] = simulate_researcher(observations.get(agent, ""), round_num)
            elif agent == "analyst":
                actions[agent] = simulate_analyst(observations.get(agent, ""), round_num)
            else:
                actions[agent] = simulate_reviewer(observations.get(agent, ""), round_num)
        
        print("\nActions:")
        for agent, action in actions.items():
            print(f"[{agent}]: {action}")
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"\nRewards: {rewards}")
        
        if any(terminations.values()) or any(truncations.values()):
            print("\nCollaboration ending...")
            break
        
        round_num += 1
    
    print("\n" + "=" * 50)
    print("Collaboration Complete")
    print(f"Total rounds: {env.round_count}")
    print(f"Final cumulative rewards: {env._cumulative_rewards}")
    print("=" * 50)


if __name__ == "__main__":
    main()