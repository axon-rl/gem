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

from typing import List, Optional


class AgentSelector:
    
    def __init__(self, agents: List[str]):
        self.agents = agents.copy()
        self._current_index = 0
        self.selected = agents[0] if agents else None
    
    def next(self) -> Optional[str]:
        if not self.agents:
            return None
        self._current_index = (self._current_index + 1) % len(self.agents)
        self.selected = self.agents[self._current_index]
        return self.selected
    
    def reset(self) -> Optional[str]:
        self._current_index = 0
        self.selected = self.agents[0] if self.agents else None
        return self.selected
    
    def is_first(self) -> bool:
        return self._current_index == 0
    
    def is_last(self) -> bool:
        if not self.agents:
            return True
        return self._current_index == len(self.agents) - 1
    
    def reinit(self, agents: List[str]) -> None:
        self.agents = agents.copy()
        self._current_index = 0
        self.selected = agents[0] if agents else None
    
    def remove_agent(self, agent: str) -> None:
        if agent not in self.agents:
            return
        
        current_agent = self.selected
        agent_index = self.agents.index(agent)
        
        self.agents.remove(agent)
        
        if not self.agents:
            self.selected = None
            self._current_index = 0
            return
        
        if current_agent == agent:
            if agent_index < len(self.agents):
                self._current_index = agent_index
            else:
                self._current_index = 0
            self.selected = self.agents[self._current_index]
        elif agent_index < self._current_index:
            self._current_index -= 1
    
    def agent_order(self) -> List[str]:
        return self.agents.copy()
    
    def __len__(self) -> int:
        return len(self.agents)

