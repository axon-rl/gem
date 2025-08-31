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

"""Tests for agent implementations."""


from gem.multiagent.agents import BaseAgent, ToolAgent, UserAgent
from gem.multiagent.agents.user_agent import UserStrategy


class TestBaseAgent:
    """Test BaseAgent functionality."""

    def test_initialization(self):
        """Test base agent initialization."""

        class TestAgent(BaseAgent):
            def act(self, observation: str) -> str:
                return "action"

        agent = TestAgent("test_agent")

        assert agent.agent_id == "test_agent"
        assert agent.messages == []
        assert agent.observation_history == []
        assert agent.action_history == []
        assert agent.reward_history == []

    def test_observe(self):
        """Test observation processing."""

        class TestAgent(BaseAgent):
            def act(self, observation: str) -> str:
                return "action"

        agent = TestAgent("test_agent")

        agent.observe("observation1")
        agent.observe("observation2")

        assert len(agent.observation_history) == 2
        assert agent.observation_history[0] == "observation1"
        assert agent.observation_history[1] == "observation2"

    def test_update(self):
        """Test agent update."""

        class TestAgent(BaseAgent):
            def act(self, observation: str) -> str:
                return "action"

        agent = TestAgent("test_agent")

        agent.update("action1", 1.0, False)
        agent.update("action2", 0.5, True)

        assert agent.action_history == ["action1", "action2"]
        assert agent.reward_history == [1.0, 0.5]

    def test_reset(self):
        """Test agent reset."""

        class TestAgent(BaseAgent):
            def act(self, observation: str) -> str:
                return "action"

        agent = TestAgent("test_agent")

        # Add some history
        agent.observe("obs")
        agent.update("action", 1.0, False)
        agent.messages.append({"test": "message"})

        # Reset
        agent.reset()

        assert agent.messages == []
        assert agent.observation_history == []
        assert agent.action_history == []
        assert agent.reward_history == []

    def test_get_set_state(self):
        """Test state getting and setting."""

        class TestAgent(BaseAgent):
            def act(self, observation: str) -> str:
                return "action"

        agent = TestAgent("test_agent")

        # Set up some state
        agent.observe("obs1")
        agent.update("action1", 1.0, False)
        agent.messages.append({"content": "msg1"})

        # Get state
        state = agent.get_state()

        assert state["agent_id"] == "test_agent"
        assert state["observation_history"] == ["obs1"]
        assert state["action_history"] == ["action1"]
        assert state["reward_history"] == [1.0]
        assert len(state["messages"]) == 1

        # Create new agent and set state
        agent2 = TestAgent("other_agent")
        agent2.set_state(state)

        assert agent2.observation_history == ["obs1"]
        assert agent2.action_history == ["action1"]
        assert agent2.reward_history == [1.0]
        assert len(agent2.messages) == 1

    def test_messaging(self):
        """Test message sending and receiving."""

        class TestAgent(BaseAgent):
            def act(self, observation: str) -> str:
                return "action"

        agent1 = TestAgent("agent1")
        agent2 = TestAgent("agent2")

        # Send message
        message = agent1.send_message("agent2", "Hello", {"priority": "high"})

        assert message["sender"] == "agent1"
        assert message["receiver"] == "agent2"
        assert message["content"] == "Hello"
        assert message["metadata"]["priority"] == "high"
        assert message in agent1.messages

        # Receive message
        agent2.receive_message(message)
        assert message in agent2.messages


class TestUserAgent:
    """Test UserAgent functionality."""

    def test_initialization(self):
        """Test user agent initialization."""
        agent = UserAgent(
            agent_id="user",
            strategy=UserStrategy.SCRIPTED,
            script=["Hello", "Help me", "Thanks"],
            max_turns=5,
        )

        assert agent.agent_id == "user"
        assert agent.strategy == UserStrategy.SCRIPTED
        assert len(agent.script) == 3
        assert agent.max_turns == 5
        assert agent.turn_count == 0

    def test_scripted_strategy(self):
        """Test scripted user strategy."""
        agent = UserAgent(
            strategy=UserStrategy.SCRIPTED, script=["First", "Second", "Third"]
        )

        response1 = agent.act("obs1")
        assert response1 == "First"

        response2 = agent.act("obs2")
        assert response2 == "Second"

        response3 = agent.act("obs3")
        assert response3 == "Third"

        # Should return termination after script ends
        response4 = agent.act("obs4")
        assert response4 == "###STOP###"

    def test_random_strategy(self):
        """Test random user strategy."""
        agent = UserAgent(strategy=UserStrategy.RANDOM)

        responses = set()
        for _ in range(10):
            response = agent.act("observation")
            responses.add(response)

        # Should generate varied responses
        assert len(responses) > 1

    def test_verify_strategy(self):
        """Test verify user strategy."""
        agent = UserAgent(strategy=UserStrategy.VERIFY, max_verification_attempts=2)

        response1 = agent.act("observation")
        assert "verify" in response1.lower() or "confirm" in response1.lower()

        response2 = agent.act("observation")
        assert "verify" in response2.lower() or "confirm" in response2.lower()

        # After max attempts, should complete
        response3 = agent.act("observation")
        assert "verified" in response3.lower()

    def test_termination_conditions(self):
        """Test various termination conditions."""
        agent = UserAgent(max_turns=3)

        # Test turn limit
        agent.act("obs1")
        agent.act("obs2")
        response = agent.act("obs3")
        assert response == "###STOP###"

        # Test termination keywords
        agent2 = UserAgent(max_turns=10)
        response = agent2.act("Thank you, goodbye!")
        assert response == "###STOP###"

    def test_task_completion(self):
        """Test task completion detection."""
        task = {"outputs": ["answer1", "answer2"]}
        agent = UserAgent(task=task)

        # Not complete
        response = agent.act("Here is answer1")
        assert response != "###STOP###"

        # Complete
        response = agent.act("Here is answer1 and answer2")
        assert response == "###STOP###"

    def test_reset(self):
        """Test user agent reset."""
        agent = UserAgent(strategy=UserStrategy.SCRIPTED, script=["First", "Second"])

        agent.act("obs1")
        agent.turn_count = 5
        agent.conversation_state = "middle"

        agent.reset()

        assert agent.turn_count == 0
        assert agent.conversation_state == "initial"
        assert agent.script_index == 0
        assert agent.verification_attempts == 0


class TestToolAgent:
    """Test ToolAgent functionality."""

    def test_initialization(self):
        """Test tool agent initialization."""
        tools = {
            "search": lambda query: f"Results for {query}",
            "calculate": lambda expr: eval(expr),
        }

        agent = ToolAgent(agent_id="tool_agent", tools=tools, strategy="react")

        assert agent.agent_id == "tool_agent"
        assert len(agent.tools) == 2
        assert agent.strategy == "react"
        assert agent.max_tool_calls == 5

    def test_parse_tool_request_function_format(self):
        """Test parsing tool requests in function format."""
        agent = ToolAgent(tools={"search": None, "calculate": None})

        # Function call format
        request = agent._parse_tool_request('search(query="test")')
        assert request is not None
        assert request["tool"] == "search"
        assert request["parameters"]["query"] == "test"

        # Multiple parameters
        request = agent._parse_tool_request('calculate(expr="1+1", precision=2)')
        assert request is not None
        assert request["tool"] == "calculate"
        assert request["parameters"]["expr"] == "1+1"
        assert request["parameters"]["precision"] == 2

    def test_parse_tool_request_json_format(self):
        """Test parsing tool requests in JSON format."""
        agent = ToolAgent(tools={"search": None})

        request = agent._parse_tool_request(
            '{"tool": "search", "parameters": {"query": "test"}}'
        )
        assert request is not None
        assert request["tool"] == "search"
        assert request["parameters"]["query"] == "test"

    def test_parse_tool_request_natural_language(self):
        """Test parsing tool requests from natural language."""
        agent = ToolAgent(tools={"search": None})

        request = agent._parse_tool_request("Please search for Python programming")
        assert request is not None
        assert request["tool"] == "search"
        assert "Python programming" in request["parameters"].get("query", "")

    def test_execute_tool(self):
        """Test tool execution."""
        tools = {
            "search": lambda query="": f"Found: {query}",
            "error_tool": lambda: 1 / 0,  # Will raise error
        }

        agent = ToolAgent(tools=tools)

        # Successful execution
        result = agent._execute_tool(
            {"tool": "search", "parameters": {"query": "test"}}
        )
        assert result["success"] is True
        assert result["output"] == "Found: test"
        assert result["error"] is None

        # Failed execution
        result = agent._execute_tool({"tool": "error_tool", "parameters": {}})
        assert result["success"] is False
        assert result["error"] is not None
        assert "division by zero" in result["error"]

        # Non-existent tool
        result = agent._execute_tool({"tool": "nonexistent", "parameters": {}})
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_act_with_tool_request(self):
        """Test acting when tool request is detected."""
        tools = {"search": lambda query="": f"Results: {query}"}

        agent = ToolAgent(tools=tools)

        response = agent.act('search(query="Python")')
        assert "Tool 'search' executed successfully" in response
        assert "Results: Python" in response

    def test_act_strategies(self):
        """Test different action strategies."""
        agent = ToolAgent(strategy="react")
        response = agent.act("Help me find information")
        assert "Thought:" in response or "Action:" in response

        agent = ToolAgent(strategy="act")
        response = agent.act("Help me find information")
        assert "Action:" in response

        agent = ToolAgent(strategy="tool_calling")
        response = agent.act("I need to search for something")
        # Should identify search tool
        assert "search" in response.lower() or "No relevant tools" in response

    def test_tool_call_limit(self):
        """Test tool call limit."""
        tools = {"search": lambda query="": "Results"}

        agent = ToolAgent(tools=tools, max_tool_calls=2)

        # First two calls should work
        agent.act('search(query="1")')
        assert agent.tool_call_count == 1

        agent.act('search(query="2")')
        assert agent.tool_call_count == 2

        # Third call should not execute tool
        response = agent.act('search(query="3")')
        assert "executed successfully" not in response
        assert agent.tool_call_count == 2

    def test_add_remove_tools(self):
        """Test adding and removing tools."""
        agent = ToolAgent()

        assert len(agent.tools) == 0

        # Add tool
        agent.add_tool("search", lambda q: f"Search: {q}")
        assert "search" in agent.tools
        assert len(agent.tools) == 1

        # Remove tool
        agent.remove_tool("search")
        assert "search" not in agent.tools
        assert len(agent.tools) == 0

        # Remove non-existent tool (should not error)
        agent.remove_tool("nonexistent")

    def test_reset(self):
        """Test tool agent reset."""
        tools = {"search": lambda q="": "Results"}
        agent = ToolAgent(tools=tools)

        agent.act('search(query="test")')
        agent.tool_call_count = 3
        agent.execution_history.append({"test": "history"})

        agent.reset()

        assert agent.tool_call_count == 0
        assert agent.execution_history == []
        assert agent.observation_history == []
