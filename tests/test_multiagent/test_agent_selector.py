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

"""Tests for agent selector utility."""

import pytest

from gem.multiagent.agent_selector import AgentSelector


class TestAgentSelector:
    """Test AgentSelector functionality."""
    
    def test_initialization(self):
        """Test agent selector initialization."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        assert selector.selected == "agent1"
        assert len(selector) == 3
        assert selector.agent_order() == agents
    
    def test_empty_initialization(self):
        """Test initialization with empty agent list."""
        selector = AgentSelector([])
        
        assert selector.selected is None
        assert len(selector) == 0
    
    def test_next(self):
        """Test moving to next agent."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        assert selector.selected == "agent1"
        
        selector.next()
        assert selector.selected == "agent2"
        
        selector.next()
        assert selector.selected == "agent3"
        
        # Should wrap around
        selector.next()
        assert selector.selected == "agent1"
    
    def test_next_with_empty_list(self):
        """Test next with no agents."""
        selector = AgentSelector([])
        
        result = selector.next()
        assert result is None
        assert selector.selected is None
    
    def test_reset(self):
        """Test resetting to first agent."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        selector.next()
        selector.next()
        assert selector.selected == "agent3"
        
        selector.reset()
        assert selector.selected == "agent1"
    
    def test_is_first(self):
        """Test checking if current agent is first."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        assert selector.is_first() is True
        
        selector.next()
        assert selector.is_first() is False
        
        selector.next()
        assert selector.is_first() is False
        
        selector.next()
        assert selector.is_first() is True
    
    def test_is_last(self):
        """Test checking if current agent is last."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        assert selector.is_last() is False
        
        selector.next()
        assert selector.is_last() is False
        
        selector.next()
        assert selector.is_last() is True
        
        selector.next()
        assert selector.is_last() is False
    
    def test_is_last_with_empty_list(self):
        """Test is_last with no agents."""
        selector = AgentSelector([])
        assert selector.is_last() is True
    
    def test_reinit(self):
        """Test reinitializing with new agents."""
        selector = AgentSelector(["agent1", "agent2"])
        
        selector.next()
        assert selector.selected == "agent2"
        
        # Reinitialize with different agents
        selector.reinit(["agentA", "agentB", "agentC"])
        
        assert selector.selected == "agentA"
        assert len(selector) == 3
        assert selector.agent_order() == ["agentA", "agentB", "agentC"]
    
    def test_remove_agent_not_selected(self):
        """Test removing an agent that is not currently selected."""
        agents = ["agent1", "agent2", "agent3", "agent4"]
        selector = AgentSelector(agents)
        
        assert selector.selected == "agent1"
        
        # Remove agent3
        selector.remove_agent("agent3")
        
        assert len(selector) == 3
        assert "agent3" not in selector.agents
        assert selector.selected == "agent1"
        
        # Cycle through remaining agents
        selector.next()
        assert selector.selected == "agent2"
        selector.next()
        assert selector.selected == "agent4"
        selector.next()
        assert selector.selected == "agent1"
    
    def test_remove_selected_agent(self):
        """Test removing the currently selected agent."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        selector.next()  # Move to agent2
        assert selector.selected == "agent2"
        
        # Remove the selected agent
        selector.remove_agent("agent2")
        
        assert len(selector) == 2
        assert "agent2" not in selector.agents
        assert selector.selected == "agent3"  # Should move to next agent
    
    def test_remove_selected_agent_at_end(self):
        """Test removing the selected agent when it's the last one."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        selector.next()
        selector.next()  # Move to agent3
        assert selector.selected == "agent3"
        
        # Remove the last agent
        selector.remove_agent("agent3")
        
        assert len(selector) == 2
        assert selector.selected == "agent1"  # Should wrap to first
    
    def test_remove_agent_before_selected(self):
        """Test removing an agent before the selected one."""
        agents = ["agent1", "agent2", "agent3", "agent4"]
        selector = AgentSelector(agents)
        
        selector.next()
        selector.next()  # Move to agent3
        assert selector.selected == "agent3"
        
        # Remove agent1 (before selected)
        selector.remove_agent("agent1")
        
        assert len(selector) == 3
        assert selector.selected == "agent3"  # Selection should remain the same
        
        # But the index should be adjusted
        selector.next()
        assert selector.selected == "agent4"
        selector.next()
        assert selector.selected == "agent2"
    
    def test_remove_nonexistent_agent(self):
        """Test removing an agent that doesn't exist."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        # Should not raise error
        selector.remove_agent("agent4")
        
        assert len(selector) == 3
        assert selector.selected == "agent1"
    
    def test_remove_all_agents(self):
        """Test removing all agents."""
        agents = ["agent1", "agent2"]
        selector = AgentSelector(agents)
        
        selector.remove_agent("agent1")
        assert selector.selected == "agent2"
        
        selector.remove_agent("agent2")
        assert selector.selected is None
        assert len(selector) == 0
    
    def test_agent_order(self):
        """Test getting agent order."""
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)
        
        order = selector.agent_order()
        
        # Should return a copy
        assert order == agents
        assert order is not selector.agents
        
        # Modifying returned list shouldn't affect selector
        order.append("agent4")
        assert len(selector) == 3