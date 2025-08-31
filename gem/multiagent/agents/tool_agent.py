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

"""Tool agent implementation for executing tools and APIs."""

import json
import re
from typing import Any, Dict, List, Optional

from gem.multiagent.agents.base_agent import BaseAgent


class ToolAgent(BaseAgent):
    """Agent that can execute tools and APIs.
    
    This agent processes requests and executes appropriate tools
    to accomplish tasks in multi-agent environments.
    """
    
    def __init__(
        self,
        agent_id: str = "tool_agent",
        tools: Optional[Dict[str, Any]] = None,
        strategy: str = "react",
        **kwargs
    ):
        """Initialize the tool agent.
        
        Args:
            agent_id: Unique identifier for this agent.
            tools: Dictionary of available tools.
            strategy: Strategy for tool execution (react, act, tool_calling).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(agent_id, **kwargs)
        self.tools = tools or {}
        self.strategy = strategy
        self.max_tool_calls = kwargs.get("max_tool_calls", 5)
        self.tool_call_count = 0
        self.execution_history: List[Dict] = []
    
    def act(self, observation: str) -> str:
        """Generate an action based on the observation.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            The action to take (tool call or response).
        """
        self.observe(observation)
        
        # Parse the observation to determine if tool use is needed
        tool_request = self._parse_tool_request(observation)
        
        if tool_request and self.tool_call_count < self.max_tool_calls:
            # Execute the requested tool
            result = self._execute_tool(tool_request)
            self.tool_call_count += 1
            return self._format_tool_response(result)
        
        # Generate response based on strategy
        if self.strategy == "react":
            response = self._react_response(observation)
        elif self.strategy == "act":
            response = self._act_response(observation)
        elif self.strategy == "tool_calling":
            response = self._tool_calling_response(observation)
        else:
            response = self._default_response(observation)
        
        self.action_history.append(response)
        return response
    
    def _parse_tool_request(self, observation: str) -> Optional[Dict[str, Any]]:
        """Parse observation to extract tool request.
        
        Args:
            observation: The observation to parse.
            
        Returns:
            Dictionary with tool name and parameters, or None.
        """
        # Look for tool calling patterns
        # Pattern 1: Function call format - tool_name(param1="value1", param2="value2")
        func_pattern = r'(\w+)\((.*?)\)'
        match = re.search(func_pattern, observation)
        
        if match:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            if tool_name in self.tools:
                # Parse parameters
                params = self._parse_parameters(params_str)
                return {
                    "tool": tool_name,
                    "parameters": params
                }
        
        # Pattern 2: JSON format
        try:
            if "{" in observation and "}" in observation:
                json_str = observation[observation.index("{"):observation.rindex("}")+1]
                data = json.loads(json_str)
                if "tool" in data or "function" in data:
                    tool_name = data.get("tool") or data.get("function")
                    if tool_name in self.tools:
                        return {
                            "tool": tool_name,
                            "parameters": data.get("parameters", {})
                        }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Pattern 3: Natural language request
        for tool_name in self.tools:
            if tool_name.lower() in observation.lower():
                # Extract parameters from context
                params = self._extract_params_from_text(observation, tool_name)
                if params is not None:
                    return {
                        "tool": tool_name,
                        "parameters": params
                    }
        
        return None
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dictionary.
        
        Args:
            params_str: String containing parameters.
            
        Returns:
            Dictionary of parameters.
        """
        params = {}
        
        if not params_str:
            return params
        
        # Parse key=value pairs
        param_pattern = r'(\w+)\s*=\s*["\']?([^"\',]*)["\']?'
        matches = re.findall(param_pattern, params_str)
        
        for key, value in matches:
            # Try to parse value type
            if value.lower() == "true":
                params[key] = True
            elif value.lower() == "false":
                params[key] = False
            elif value.isdigit():
                params[key] = int(value)
            else:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
        
        return params
    
    def _extract_params_from_text(self, text: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """Extract parameters from natural language text.
        
        Args:
            text: The text to extract from.
            tool_name: The name of the tool.
            
        Returns:
            Dictionary of extracted parameters or None.
        """
        # This is a simplified implementation
        # In practice, this would use more sophisticated NLP
        
        params = {}
        
        # Look for common parameter patterns
        if tool_name == "search":
            # Extract search query
            query_match = re.search(r'search for ["\']?([^"\']+)["\']?', text.lower())
            if query_match:
                params["query"] = query_match.group(1)
                return params
        
        elif tool_name == "python":
            # Extract code block
            code_match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
            if code_match:
                params["code"] = code_match.group(1)
                return params
        
        # Generic parameter extraction
        if params:
            return params
        
        return None
    
    def _execute_tool(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the requested tool.
        
        Args:
            tool_request: Dictionary with tool name and parameters.
            
        Returns:
            Dictionary with execution results.
        """
        tool_name = tool_request["tool"]
        parameters = tool_request.get("parameters", {})
        
        result = {
            "tool": tool_name,
            "parameters": parameters,
            "success": False,
            "output": None,
            "error": None
        }
        
        try:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                
                # Execute tool (assuming tool is callable or has invoke method)
                if callable(tool):
                    output = tool(**parameters)
                elif hasattr(tool, "invoke"):
                    output = tool.invoke(**parameters)
                elif hasattr(tool, "execute"):
                    output = tool.execute(**parameters)
                else:
                    raise ValueError(f"Tool {tool_name} is not executable")
                
                result["success"] = True
                result["output"] = output
            else:
                result["error"] = f"Tool {tool_name} not found"
        
        except Exception as e:
            result["error"] = str(e)
        
        # Store in execution history
        self.execution_history.append(result)
        
        return result
    
    def _format_tool_response(self, result: Dict[str, Any]) -> str:
        """Format tool execution result as response.
        
        Args:
            result: The tool execution result.
            
        Returns:
            Formatted response string.
        """
        if result["success"]:
            return f"Tool '{result['tool']}' executed successfully. Output: {result['output']}"
        else:
            return f"Tool '{result['tool']}' failed. Error: {result['error']}"
    
    def _react_response(self, observation: str) -> str:
        """Generate response using ReAct strategy (Reasoning + Acting).
        
        Args:
            observation: The current observation.
            
        Returns:
            The ReAct response.
        """
        # Reasoning phase
        reasoning = self._generate_reasoning(observation)
        
        # Acting phase
        action = self._determine_action(reasoning)
        
        return f"Thought: {reasoning}\nAction: {action}"
    
    def _act_response(self, observation: str) -> str:
        """Generate response using direct acting strategy.
        
        Args:
            observation: The current observation.
            
        Returns:
            The action response.
        """
        # Directly determine action without explicit reasoning
        action = self._determine_action(observation)
        return f"Action: {action}"
    
    def _tool_calling_response(self, observation: str) -> str:
        """Generate response using tool calling strategy.
        
        Args:
            observation: The current observation.
            
        Returns:
            The tool calling response.
        """
        # Analyze which tools might be helpful
        relevant_tools = self._identify_relevant_tools(observation)
        
        if relevant_tools:
            tool = relevant_tools[0]
            params = self._generate_tool_params(tool, observation)
            return f"{tool}({params})"
        
        return "No relevant tools identified for this request."
    
    def _default_response(self, observation: str) -> str:
        """Generate a default response.
        
        Args:
            observation: The current observation.
            
        Returns:
            The default response.
        """
        return "I'll help you with that. Let me process your request."
    
    def _generate_reasoning(self, observation: str) -> str:
        """Generate reasoning about the observation.
        
        Args:
            observation: The observation to reason about.
            
        Returns:
            The reasoning string.
        """
        # Simplified reasoning generation
        if "search" in observation.lower():
            return "The user needs information. I should use the search tool."
        elif "calculate" in observation.lower() or "compute" in observation.lower():
            return "The user needs computation. I should use the python tool."
        else:
            return "I need to understand what the user wants."
    
    def _determine_action(self, context: str) -> str:
        """Determine the action to take based on context.
        
        Args:
            context: The context (observation or reasoning).
            
        Returns:
            The action to take.
        """
        # Simple action determination
        if "search" in context.lower():
            return "search(query='relevant information')"
        elif "calculate" in context.lower():
            return "python(code='# computation code here')"
        else:
            return "respond(content='How can I help you?')"
    
    def _identify_relevant_tools(self, observation: str) -> List[str]:
        """Identify which tools are relevant for the observation.
        
        Args:
            observation: The observation to analyze.
            
        Returns:
            List of relevant tool names.
        """
        relevant = []
        
        observation_lower = observation.lower()
        
        # Map keywords to tools
        tool_keywords = {
            "search": ["search", "find", "look up", "information"],
            "python": ["calculate", "compute", "code", "program"],
            "respond": ["answer", "response", "reply"],
        }
        
        for tool, keywords in tool_keywords.items():
            if tool in self.tools:
                if any(keyword in observation_lower for keyword in keywords):
                    relevant.append(tool)
        
        return relevant
    
    def _generate_tool_params(self, tool: str, observation: str) -> str:
        """Generate parameters for a tool based on observation.
        
        Args:
            tool: The tool name.
            observation: The observation to extract parameters from.
            
        Returns:
            String representation of parameters.
        """
        # Simplified parameter generation
        if tool == "search":
            # Extract key terms from observation
            terms = [word for word in observation.split() if len(word) > 3]
            query = " ".join(terms[:5])  # Use first 5 significant words
            return f"query='{query}'"
        elif tool == "python":
            return "code='# Add code here'"
        else:
            return ""
    
    def reset(self) -> None:
        """Reset the agent's state for a new episode."""
        super().reset()
        self.tool_call_count = 0
        self.execution_history = []
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names.
        
        Returns:
            List of tool names.
        """
        return list(self.tools.keys())
    
    def add_tool(self, name: str, tool: Any) -> None:
        """Add a new tool to the agent.
        
        Args:
            name: The name of the tool.
            tool: The tool object or function.
        """
        self.tools[name] = tool
    
    def remove_tool(self, name: str) -> None:
        """Remove a tool from the agent.
        
        Args:
            name: The name of the tool to remove.
        """
        if name in self.tools:
            del self.tools[name]