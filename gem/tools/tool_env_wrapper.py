from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env, EnvWrapper
from gem.tools.base_tool import BaseTool


class ToolEnvWrapper(EnvWrapper):
    def __init__(
        self,
        env: Env,
        tools: List[BaseTool],
        tool_use_reward: float = 0.1,
        max_tool_uses: Optional[int] = 10,
    ):
        super().__init__(env)
        self.tools = tools
        self.tool_use_reward = tool_use_reward
        self.max_tool_uses = (
            max_tool_uses if max_tool_uses is not None else float("inf")
        )
        self.tool_use_counter = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        prev_ep_tool_uses = self.tool_use_counter
        self.tool_use_counter = 0
        obs, info = self.env.reset(seed=seed)
        tool_instructions = "\n".join(
            [tool.instruction_string() for tool in self.tools]
        )
        if len(self.tools) > 1:
            tool_instructions = f"Available tools:\n{tool_instructions}"
        obs = f"{obs}\n{tool_instructions}"
        info["tool_use_counter"] = self.tool_use_counter
        info["prev_ep_tool_use_counter"] = prev_ep_tool_uses
        return obs, info

    def step(
        self, action: str,
        verbose: bool = False,
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        # try to execute the action with each tool
        tool_executed = False
        if self.tool_use_counter < self.max_tool_uses:
            for tool in self.tools:
                observation, terminated, tool_executed = tool.execute_action(action)
                if tool_executed:
                    break

        if tool_executed:
            self.tool_use_counter += 1
            reward = self.tool_use_reward
            truncated = False
            info = {}
            if verbose:
                print(f"Tool executed: {tool.name}, tool use count: {self.tool_use_counter}")
        # if no tool was executed, step the environment
        else:
            observation, reward, terminated, truncated, info = self.env.step(action)

        info["tool_use_counter"] = self.tool_use_counter
        return observation, reward, terminated, truncated, info
