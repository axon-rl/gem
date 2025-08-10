"""MCP Tool implementation for connecting to any MCP server."""

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

import asyncio
import json
import logging
import os
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gem.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("GEM_LOGGING_LEVEL", "WARN"))


def is_timeout_error(error: Exception) -> bool:
    """Check if an error is a timeout-related error."""
    error_str = str(error)
    return any(
        keyword in error_str
        for keyword in ["ETIMEDOUT", "ECONNRESET", "Timeout", "Timed out"]
    )


class MCPTool(BaseTool):
    """A tool for connecting to any MCP (Model Context Protocol) server.

    This tool provides a generic interface to connect to and interact with
    MCP servers following the GEM framework's BaseTool interface.
    """

    tool_type = "mcp"

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env_vars: Optional[List[str]] = None,
        max_retries: int = 3,
        delay_between_retries: float = 1.0,
        connection_timeout: float = 10.0,
        execution_timeout: float = 30.0,
        num_workers: int = 1,
        **kwargs,
    ):
        """Initialize the MCP tool.

        Args:
            command: The command to run the MCP server
            args: Optional list of arguments for the command
            env_vars: Optional list of environment variable names to pass
            max_retries: Maximum number of retry attempts on failure
            delay_between_retries: Delay in seconds between retry attempts
            connection_timeout: Timeout in seconds for connection establishment
            execution_timeout: Timeout in seconds for tool execution
            num_workers: Number of worker processes
        """
        super().__init__(num_workers)

        # MCP server configuration
        self.params = StdioServerParameters(
            command=command,
            args=args or [],
            env={var: os.environ.get(var, "") for var in env_vars or []},
        )

        # Retry and timeout configuration
        self.max_retries = max_retries
        self.delay_between_retries = delay_between_retries
        self.connection_timeout = connection_timeout
        self.execution_timeout = execution_timeout

        # Tool discovery and caching
        self._available_tools: Optional[List[Dict[str, Any]]] = None
        self._tools_discovered = False

    async def _discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools from the MCP server."""
        if self._tools_discovered:
            return self._available_tools or []

        tools = []
        async with stdio_client(self.params) as (read, write):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=self.connection_timeout),
            ) as session:
                for attempt in range(self.max_retries):
                    try:
                        await session.initialize()
                        response = await session.list_tools()
                        for tool in response.tools:
                            tool_info = {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            }
                            tools.append(tool_info)
                        break
                    except Exception as e:
                        logger.warning(
                            f"Tool discovery attempt {attempt + 1} failed: {e}"
                        )
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.delay_between_retries)
                        else:
                            logger.error(
                                f"Failed to discover tools after {self.max_retries} attempts"
                            )

        self._available_tools = tools
        self._tools_discovered = True
        return tools

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from the MCP server (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._discover_tools())

    def _parse_action(self, action: str) -> Tuple[str, str, Dict[str, Any], bool]:
        """Parse action to extract tool name and parameters.

        Expected format: <mcp_tool name="tool_name">{"param1": "value1"}</mcp_tool>

        Args:
            action: Raw action string from agent

        Returns:
            tuple: (tool_name, parsed_action, parameters_dict, is_valid)
        """
        pattern = r'<mcp_tool\s+name="([^"]+)">(.*?)</mcp_tool>'
        match = re.search(pattern, action, re.DOTALL)

        if not match:
            return "", "", {}, False

        tool_name = match.group(1).strip()
        params_str = match.group(2).strip()
        parsed_action = action[: match.end()]

        try:
            if params_str:
                parameters = json.loads(params_str)
            else:
                parameters = {}
            return tool_name, parsed_action, parameters, True
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse parameters JSON: {e}")
            return tool_name, parsed_action, {}, False

    async def _execute_mcp_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> str:
        """Execute a specific MCP tool with given parameters."""
        response = ""

        async with stdio_client(self.params) as (read, write):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=self.connection_timeout),
            ) as session:
                for attempt in range(self.max_retries):
                    try:
                        await session.initialize()
                        result = await session.call_tool(
                            tool_name,
                            arguments=parameters,
                            read_timeout_seconds=timedelta(
                                seconds=self.execution_timeout
                            ),
                        )
                        response = result.content[0].text if result.content else ""
                        if attempt > 0:
                            logger.info(
                                f"MCP tool execution succeeded on attempt {attempt + 1}"
                            )
                        break
                    except Exception as e:
                        if is_timeout_error(e) and attempt < self.max_retries - 1:
                            logger.warning(
                                f"MCP tool execution attempt {attempt + 1} failed: {e}"
                            )
                            await asyncio.sleep(self.delay_between_retries)
                        else:
                            response = f"MCP tool execution failed: {e}"
                            logger.error(response)
                            break

        return response

    def instruction_string(self) -> str:
        """Return instruction string for using the MCP tool."""
        tools = self.get_available_tools()
        if not tools:
            return (
                "MCP server connection failed. No tools available.\n"
                "Please check your MCP server configuration."
            )

        tool_descriptions = []
        for tool in tools:
            desc = f"- {tool['name']}: {tool['description']}"
            if tool.get("parameters", {}).get("properties"):
                params = list(tool["parameters"]["properties"].keys())
                desc += f" (Parameters: {', '.join(params)})"
            tool_descriptions.append(desc)

        return (
            "You have access to MCP (Model Context Protocol) tools to help with various tasks.\n\n"
            "Available MCP tools:\n"
            + "\n".join(tool_descriptions)
            + "\n\nTo use an MCP tool, format your request as:\n"
            '<mcp_tool name="tool_name">{"parameter1": "value1", "parameter2": "value2"}</mcp_tool>\n\n'
            "The tool will execute and return results to help with your task."
        )

    def execute_action(self, action: str) -> Tuple[bool, bool, str, str]:
        """Execute the MCP tool action.

        Args:
            action: Raw action string containing MCP tool call

        Returns:
            tuple: (is_valid, has_error, observation, parsed_action)
        """
        tool_name, parsed_action, parameters, is_valid = self._parse_action(action)

        if not is_valid:
            return False, True, "", ""

        # Check if the requested tool exists
        available_tools = self.get_available_tools()
        tool_names = [tool["name"] for tool in available_tools]

        if tool_name not in tool_names:
            error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(tool_names)}"
            return False, True, error_msg, parsed_action

        try:
            # Execute the MCP tool
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self._execute_mcp_tool(tool_name, parameters)
            )
            has_error = "failed" in response.lower()
            observation = f"\n<mcp_result>\n{response}\n</mcp_result>\n"
            return True, has_error, observation, parsed_action

        except Exception as e:
            error_msg = f"MCP tool execution failed: {e}"
            logger.error(error_msg)
            return False, True, error_msg, parsed_action
