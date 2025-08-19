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
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fastmcp import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.client.logging import LogMessage
from fastmcp.client.sampling import RequestContext, SamplingMessage, SamplingParams
from fastmcp.exceptions import ClientError, ToolError

from gem.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)
# silence the underlying HTTP and MCP client loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)


def _run_async(coro):
    """Run an async coroutine from both sync and already-async contexts safely.

    - If there is no running loop, uses asyncio.run
    - If called inside a running loop, spins up a dedicated thread with its own loop
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")


def is_timeout_error(error: Exception) -> bool:
    """Check if an error is a timeout-related error."""
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return True
    error_str = str(error).lower()
    return any(
        keyword in error_str
        for keyword in ["etimedout", "econnreset", "timeout", "timed out"]
    )


class MCPTool(BaseTool):
    """A tool for connecting to MCP servers.

    This tool provides a unified configuration-based interface to connect to and 
    interact with MCP servers following the GEM framework's BaseTool interface.
    
    Uses FastMCP client with configuration-based setup for reliable MCP 
    communication. Supports both simple HTTP servers and complex multi-server 
    configurations.
    
    Examples:
        # Simple HTTP server (most common case)
        tool = MCPTool("https://api.example.com/mcp")
        
        # HTTP server with authentication
        tool = MCPTool({
            "mcpServers": {
                "main": {
                    "transport": "http",
                    "url": "https://api.example.com/mcp",
                    "headers": {"Authorization": "Bearer your-token"}
                }
            }
        })
        
        # Multi-server configuration with tool transformations
        tool = MCPTool({
            "mcpServers": {
                "weather": {
                    "transport": "http",
                    "url": "https://weather-api.example.com/mcp"
                },
                "assistant": {
                    "transport": "http", 
                    "url": "https://assistant-api.example.com/mcp",
                    "tools": {
                        "ask": {"name": "assistant_ask"}  # Rename tool
                    }
                }
            }
        })
        
        # From configuration file
        tool = MCPTool.from_config_file("mcp_servers.json")
        
        # HTTP with custom authentication and callbacks
        tool = MCPTool(
            "https://api.example.com/mcp",
            auth="bearer-token",
            headers={"X-Custom": "value"},
            log_handler=my_log_handler
        )
    """

    tool_type = "mcp"

    def __init__(
        self,
        config: Union[str, Dict[str, Any]],
        # Optional authentication and headers for simple HTTP case
        auth: Optional[Union[str, BearerAuth]] = None,
        headers: Optional[Dict[str, str]] = None,
        # Optional callback handlers
        log_handler: Optional[Callable[[LogMessage], None]] = None,
        progress_handler: Optional[Callable[[float, Optional[float], Optional[str]], None]] = None,
        sampling_handler: Optional[Callable[[List[SamplingMessage], SamplingParams, RequestContext], str]] = None,
        # Retry and timeout configuration
        max_retries: int = 3,
        delay_between_retries: float = 1.0,
        execution_timeout: float = 30.0,
        num_workers: int = 1,
        **kwargs,
    ):
        """Initialize the MCP tool using configuration.

        Args:
            config: MCP server configuration. Can be:
                - URL string: "https://api.example.com/mcp" (auto-converted to config)
                - MCP config dict: Full configuration with multiple servers
            auth: Bearer token for authentication (only used with URL string input)
            headers: Custom headers for HTTP requests (only used with URL string input)
            log_handler: Handler for server log messages
            progress_handler: Handler for progress updates during long operations
            sampling_handler: Handler for server LLM sampling requests
            max_retries: Maximum number of retry attempts on failure
            delay_between_retries: Delay in seconds between retry attempts
            execution_timeout: Timeout in seconds for tool execution
            num_workers: Number of worker processes
        """
        super().__init__(num_workers)

        # Store original config and normalize it
        self.raw_config = config
        self.normalized_config = self._normalize_config(config, auth, headers)
        
        # Create FastMCP client with normalized configuration
        self.client = self._create_client(
            log_handler,
            progress_handler,
            sampling_handler,
            execution_timeout
        )

        # Retry and timeout configuration
        self.max_retries = max_retries
        self.delay_between_retries = delay_between_retries
        self.execution_timeout = execution_timeout

        # Tool discovery and caching
        self._available_tools: Optional[List[Dict[str, Any]]] = None
        self._tools_discovered = False
        
    def _normalize_config(
        self, 
        config: Union[str, Dict[str, Any]], 
        auth: Optional[Union[str, BearerAuth]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Convert any config input to standard MCP configuration format.
        
        Args:
            config: URL string or MCP configuration dictionary
            auth: Optional authentication for URL string input
            headers: Optional headers for URL string input
            
        Returns:
            Standard MCP configuration dictionary
        """
        if isinstance(config, str):
            # Auto-convert URL string to simple HTTP config
            server_config = {
                "transport": "http",
                "url": config
            }
            
            # Add authentication if provided
            if auth or headers:
                server_headers = headers or {}
                if auth:
                    if isinstance(auth, str):
                        server_headers["Authorization"] = f"Bearer {auth}"
                    # BearerAuth will be handled in client creation
                if server_headers:
                    server_config["headers"] = server_headers
                    
            return {
                "mcpServers": {
                    "default": server_config
                }
            }
        elif isinstance(config, dict):
            # Ensure all servers default to HTTP transport if not specified
            normalized = config.copy()
            if "mcpServers" in normalized:
                for server_name, server_config in normalized["mcpServers"].items():
                    if "transport" not in server_config:
                        server_config["transport"] = "http"
            return normalized
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
            
    def _create_client(
        self,
        log_handler: Optional[Callable[[LogMessage], None]],
        progress_handler: Optional[Callable[[float, Optional[float], Optional[str]], None]],
        sampling_handler: Optional[Callable[[List[SamplingMessage], SamplingParams, RequestContext], str]],
        timeout: float
    ) -> Client:
        """Create FastMCP client with normalized configuration."""
        # If no log_handler provided, use a silent handler by default
        if log_handler is None:
            log_handler = lambda msg: None  # Silent handler - does nothing with log messages
            
        client_kwargs = {
            'timeout': timeout,
            'log_handler': log_handler,
        }
        
        # Add optional handlers if provided
        if progress_handler:
            client_kwargs['progress_handler'] = progress_handler
        if sampling_handler:
            client_kwargs['sampling_handler'] = sampling_handler
            
        return Client(self.normalized_config, **client_kwargs)

    def _discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools from the MCP server (synchronous wrapper)."""
        if self._tools_discovered:
            return self._available_tools or []

        tools = _run_async(self._async_discover_tools())
        self._available_tools = tools
        self._tools_discovered = True
        return tools

    async def _async_discover_tools(self) -> List[Dict[str, Any]]:
        """Discover tools using fastMCP client."""
        tools: List[Dict[str, Any]] = []
        
        for attempt in range(self.max_retries):
            try:
                async with self.client:
                    mcp_tools = await self.client.list_tools()
                    for tool in mcp_tools:
                        tool_info = {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        }
                        tools.append(tool_info)
                break
                
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Tool discovery attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.delay_between_retries)
                else:
                    logger.error(f"Failed to discover tools after {self.max_retries} attempts")
        
        return tools

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from the MCP server."""
        return self._discover_tools()

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

    def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute a specific MCP tool with given parameters (synchronous wrapper)."""
        return _run_async(self._async_execute_tool(tool_name, parameters))

    async def _async_execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute tool using FastMCP client with enhanced result handling."""
        for attempt in range(self.max_retries):
            try:
                async with self.client:
                    result = await self.client.call_tool(
                        tool_name, 
                        parameters, 
                        timeout=self.execution_timeout,
                        raise_on_error=False
                    )
                    
                    # Check for errors using FastMCP's structured error detection
                    if result.is_error:
                        error_content = []
                        for content in result.content:
                            if hasattr(content, 'text'):
                                error_content.append(content.text)
                            else:
                                error_content.append(str(content))
                        return f"[Tool execution error: {' '.join(error_content)}]"
                    
                    # Log success if this was a retry attempt
                    if attempt > 0:
                        logger.info(f"Tool execution succeeded on attempt {attempt + 1}")
                    
                    # Use FastMCP's structured data handling
                    if result.data is not None:
                        # FastMCP provides fully hydrated Python objects
                        return str(result.data)
                    elif result.content:
                        # Fallback to content blocks when no structured data
                        parts = []
                        for content in result.content:
                            if hasattr(content, 'text'):
                                parts.append(content.text)
                            elif hasattr(content, 'data'):
                                parts.append(f"Binary data: {len(content.data)} bytes")
                            else:
                                parts.append(str(content))
                        return "\n".join(parts)
                    else:
                        # No content available
                        return "Tool execution completed with no output"
                
            except ClientError as e:
                error_msg = f"[Tool execution error: {e}]"
                logger.error(error_msg)
                return error_msg
                
            except Exception as e:  # noqa: BLE001
                if is_timeout_error(e) and attempt < self.max_retries - 1:
                    logger.warning(f"Tool execution attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.delay_between_retries)
                else:
                    error_msg = f"[Tool execution failed: {e}]"
                    logger.error(error_msg)
                    return error_msg
        
        return "[Tool execution failed after all retry attempts]"

    @classmethod
    def from_url(
        cls, 
        url: str, 
        auth: Optional[str] = None, 
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'MCPTool':
        """Create MCPTool for HTTP server with authentication.
        
        Args:
            url: HTTP URL of the MCP server
            auth: Bearer token for authentication
            headers: Custom headers to send with requests
            **kwargs: Additional arguments to pass to MCPTool constructor
            
        Returns:
            MCPTool instance configured for HTTP transport
        """
        return cls(config=url, auth=auth, headers=headers, **kwargs)
        
    @classmethod
    def from_config_file(cls, config_path: str, **kwargs) -> 'MCPTool':
        """Create MCPTool from MCP configuration file.
        
        Args:
            config_path: Path to MCP configuration JSON file
            **kwargs: Additional arguments to pass to MCPTool constructor
            
        Returns:
            MCPTool instance configured from the file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(config=config, **kwargs)
        
    @classmethod
    def from_multi_server(cls, servers: Dict[str, str], **kwargs) -> 'MCPTool':
        """Create MCPTool for multiple HTTP servers.
        
        Args:
            servers: Dictionary mapping server names to URLs
            **kwargs: Additional arguments to pass to MCPTool constructor
            
        Returns:
            MCPTool instance configured for multiple servers
        """
        config = {
            "mcpServers": {
                name: {
                    "transport": "http",
                    "url": url
                }
                for name, url in servers.items()
            }
        }
        return cls(config=config, **kwargs)

    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self.client, 'close'):
                _run_async(self.client.close())
        except Exception:
            # Ignore cleanup errors - the client may already be closed
            # or the event loop may have been closed
            pass

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'client') and self.client:
            try:
                self.close()
            except Exception:
                pass  # Ignore cleanup errors during deletion

    def instruction_string(self) -> str:
        """Return instruction string for using the MCP tool."""
        tools = self.get_available_tools()
        if not tools:
            return (
                f"MCP server connection failed. No tools available.\n"
                f"Please check your MCP server configuration: {self._get_server_description()}"
            )

        tool_descriptions = []
        for tool in sorted(tools, key=lambda t: t.get("name", "")):
            desc = f"- {tool['name']}: {tool['description']}"
            if tool.get("parameters", {}).get("properties"):
                params = list(tool["parameters"]["properties"].keys())
                desc += f" (Parameters: {', '.join(params)})"
            tool_descriptions.append(desc)

        server_desc = self._get_server_description()
        return (
            f"You have access to MCP (Model Context Protocol) tools from {server_desc}.\n\n"
            "Available MCP tools:\n"
            + "\n".join(tool_descriptions)
            + "\n\nTo use an MCP tool, format your request as:\n"
            '<mcp_tool name="tool_name">{"parameter1": "value1", "parameter2": "value2"}</mcp_tool>\n\n'
            "The tool will execute via MCP protocol and return structured results to help with your task."
        )
        
    def _get_server_description(self) -> str:
        """Get a human-readable description of the server configuration."""
        if isinstance(self.raw_config, str):
            return f"HTTP server at {self.raw_config}"
        elif isinstance(self.raw_config, dict) and "mcpServers" in self.raw_config:
            servers = list(self.raw_config["mcpServers"].keys())
            if len(servers) == 1:
                server_config = self.raw_config["mcpServers"][servers[0]]
                url = server_config.get("url", "unknown")
                return f"HTTP server '{servers[0]}' at {url}"
            else:
                return f"multi-server configuration with {len(servers)} servers: {', '.join(servers)}"
        else:
            return str(self.raw_config)

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
            # Execute the MCP tool (synchronously)
            response = self._execute_mcp_tool(tool_name, parameters)
            ERROR_PREFIXES = ("[Tool execution error", "[Tool execution failed")
            has_error = response.startswith(ERROR_PREFIXES)
            observation = f"\n<mcp_result>\n{response}\n</mcp_result>\n"
            return True, has_error, observation, parsed_action

        except Exception as e:
            error_msg = f"MCP tool execution failed: {e}"
            logger.error(error_msg)
            return False, True, error_msg, parsed_action
