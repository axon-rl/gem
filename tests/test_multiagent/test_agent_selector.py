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

from gem.multiagent.utils import AgentSelector


class TestAgentSelector:

    def test_initialization(self):
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)

        assert selector.selected == "agent1"
        assert len(selector) == 3
        assert selector.agent_order() == agents

    def test_empty_initialization(self):
        selector = AgentSelector([])

        assert selector.selected is None
        assert len(selector) == 0

    def test_next(self):
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)

        assert selector.selected == "agent1"

        selector.next()
        assert selector.selected == "agent2"

        selector.next()
        assert selector.selected == "agent3"

        selector.next()
        assert selector.selected == "agent1"

    def test_next_with_empty_list(self):
        selector = AgentSelector([])

        result = selector.next()
        assert result is None
        assert selector.selected is None

    def test_reset(self):
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)

        selector.next()
        selector.next()
        assert selector.selected == "agent3"

        selector.reset()
        assert selector.selected == "agent1"

    def test_is_first(self):
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
        selector = AgentSelector([])
        assert selector.is_last() is True

    def test_reinit(self):
        selector = AgentSelector(["agent1", "agent2"])

        selector.next()
        assert selector.selected == "agent2"

        selector.reinit(["agentA", "agentB", "agentC"])

        assert selector.selected == "agentA"
        assert len(selector) == 3
        assert selector.agent_order() == ["agentA", "agentB", "agentC"]

    def test_remove_agent_not_selected(self):
        agents = ["agent1", "agent2", "agent3", "agent4"]
        selector = AgentSelector(agents)

        assert selector.selected == "agent1"

        selector.remove_agent("agent3")

        assert len(selector) == 3
        assert "agent3" not in selector.agents
        assert selector.selected == "agent1"

        selector.next()
        assert selector.selected == "agent2"
        selector.next()
        assert selector.selected == "agent4"
        selector.next()
        assert selector.selected == "agent1"

    def test_remove_selected_agent(self):
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)

        selector.next()
        assert selector.selected == "agent2"

        selector.remove_agent("agent2")

        assert len(selector) == 2
        assert "agent2" not in selector.agents
        assert selector.selected == "agent3"

    def test_remove_selected_agent_at_end(self):
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)

        selector.next()
        selector.next()
        assert selector.selected == "agent3"

        selector.remove_agent("agent3")

        assert len(selector) == 2
        assert selector.selected == "agent1"

    def test_remove_agent_before_selected(self):
        agents = ["agent1", "agent2", "agent3", "agent4"]
        selector = AgentSelector(agents)

        selector.next()
        selector.next()
        assert selector.selected == "agent3"

        selector.remove_agent("agent1")

        assert len(selector) == 3
        assert selector.selected == "agent3"

        selector.next()
        assert selector.selected == "agent4"
        selector.next()
        assert selector.selected == "agent2"

    def test_remove_nonexistent_agent(self):
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)

        selector.remove_agent("agent4")

        assert len(selector) == 3
        assert selector.selected == "agent1"

    def test_remove_all_agents(self):
        agents = ["agent1", "agent2"]
        selector = AgentSelector(agents)

        selector.remove_agent("agent1")
        assert selector.selected == "agent2"

        selector.remove_agent("agent2")
        assert selector.selected is None
        assert len(selector) == 0

    def test_agent_order(self):
        agents = ["agent1", "agent2", "agent3"]
        selector = AgentSelector(agents)

        order = selector.agent_order()

        assert order == agents
        assert order is not selector.agents

        order.append("agent4")
        assert len(selector) == 3
