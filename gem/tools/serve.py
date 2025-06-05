# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
"""
Tool Server - A FastAPI server to manage and execute tools based on incoming requests.
Using asyncio for concurrent processing.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from tqdm import tqdm
from dataclasses import dataclass

import fire
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .utils import hash_requests
from collections import defaultdict

from gem.tools.python_code import PythonCodeTool

ALL_TOOLS = {
    "python_code": PythonCodeTool,
 } # TODO: add propertool registration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ToolOutput:
    observations: List[str]
    dones: List[bool]
    valids: List[bool]


# ---- Tool Management ----

class AsyncToolManager:
    """Manages all tools and their execution using asyncio"""
    
    def __init__(self, tool_types: Tuple[str], num_workers_per_tool: int = 4, use_tqdm: bool = False):
        """
        Initialize the tool manager with specified tools
        Args:
            tool_types: Tuple of tool type names to initialize
            num_workers_per_tool: Number of workers for each tool
        """
        self.tools: Dict[str, Any] = {}
        self.use_tqdm = use_tqdm
        self._initialize_tools(tool_types, num_workers_per_tool)
        
    def _initialize_tools(self, tool_types: Tuple[str], num_workers: int) -> None:
        """Initialize all tools based on tool types"""
        for tool_type in tool_types:
            try:
                self.tools[tool_type] = ALL_TOOLS[tool_type](num_workers=num_workers)
                logger.info(f"Initialized tool: {tool_type}")
            except Exception as e:
                logger.error(f"Failed to initialize tool {tool_type}: {e}")
        
        # Log available vs. active tools with emoji indicators
        logger.info("Available Tools:")
        for tool in ALL_TOOLS:
            if tool in self.tools:
                status = "active 🟢"  # Green circle for active tools
                logger.info(f"  - {tool}: {status}")
            else:
                status = "inactive ⚪"  # White circle for inactive tools
                logger.info(f"  - {tool}: {status}")
    
    async def process_actions(
        self, 
        actions: List[str], 
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """
        Process a batch of actions asynchronously using appropriate tools
        Args: actions: List of action strings
        Returns: Tuple of (observations, dones, valids) lists
        """
        raise NotImplementedError("Not implemented")
