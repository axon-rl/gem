from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env, EnvWrapper
from gem.tools.base_tool import BaseTool


class ToolEnvWrapper(EnvWrapper):
    def __init__(
        self,
        env: Env,
        tools: List[BaseTool],
        tool_reward: float = 0.05,
        tool_success_reward: float = 0.25,
        max_tool_uses: Optional[int] = 10,
    ):
        super().__init__(env)
        self.tools = tools
        self.tool_reward = tool_reward
        self.tool_success_reward = tool_success_reward
        self.max_tool_uses = (
            max_tool_uses if max_tool_uses is not None else float("inf")
        )
        self.tool_use_counter = 0
        self.tool_success_counter = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        prev_ep_tool_uses = self.tool_use_counter
        prev_ep_tool_success = self.tool_success_counter
        self.tool_use_counter = 0
        self.tool_success_counter = 0
        obs, info = self.env.reset(seed=seed)
        tool_instructions = "\n".join(
            [tool.instruction_string() for tool in self.tools]
        )
        if len(self.tools) > 1:
            tool_instructions = f"Available tools:\n{tool_instructions}"
        obs = f"{obs}\n{tool_instructions}"
        info["tool_use_counter"] = self.tool_use_counter
        info["prev_ep_tool_use_counter"] = prev_ep_tool_uses
        info["tool_success_counter"] = self.tool_success_counter
        info["prev_ep_tool_success_counter"] = prev_ep_tool_success
        info["use_tool"] = False  # The initial context is not a tool result
        return obs, info

    def step(
        self,
        action: str,
        verbose: bool = False,
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        # try to execute the action with each tool
        tool_parsed = False
        if self.tool_use_counter < self.max_tool_uses:
            for tool in self.tools:
                tool_parsed, tool_execute_error, observation, parsed_action = (
                    tool.execute_action(action)
                )
                if tool_parsed and (not tool_execute_error):
                    break

        reward = 0
        if tool_parsed:
            self.tool_use_counter += 1
            if self.tool_use_counter == self.max_tool_uses:
                observation = f"{observation}\n\nReached the maximum number of tool use. Please output final answer directly."
            reward += self.tool_reward
            terminated, truncated = False, False
            info = {"parsed_action": parsed_action, "tool_type": tool.tool_type}
            if verbose:
                print(
                    f"Tool parsed: {tool.name}, tool use count: {self.tool_use_counter}"
                )
            if not tool_execute_error:
                self.tool_success_counter += 1
                reward += self.tool_success_reward
                if verbose:
                    print(
                        f"Tool executed: {tool.name}, tool use count: {self.tool_use_counter}"
                    )
        # if no tool was executed, step the environment
        else:
            observation, reward, terminated, truncated, info = self.env.step(action)

        info["tool_use_counter"] = self.tool_use_counter
        info["tool_success_counter"] = self.tool_success_counter
        info["use_tool"] = True
        return observation, reward, terminated, truncated, info
