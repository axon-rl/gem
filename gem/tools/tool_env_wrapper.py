from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env, EnvWrapper
from gem.tools.base_tool import BaseTool


class ToolEnvWrapper(EnvWrapper):
    def __init__(
        self,
        env: Env,
        tools: List[BaseTool],
        tool_use_reward: float = 0.1,
        max_tool_uses: Optional[int] = None,
    ):
        super().__init__(env)
        self.tools = tools
        self.tool_use_reward = tool_use_reward
        self.max_tool_uses = (
            max_tool_uses if max_tool_uses is not None else float("inf")
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        self.tool_use_counter = 0
        return self.env.reset(seed=seed)

    def step(
        self, action: str, verbose: bool = False
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        # try to execute the action with each tool
        for tool in self.tools:
            observation, terminated, valid = tool.execute_action(action)
            if valid:
                if verbose:
                    print(f"Action {action!r} executed by tool {tool.tool_type}")
                self.tool_use_counter += 1
                reward = self.tool_use_reward
                truncated = False
                info = {}
                break
            else:
                if verbose:
                    print(f"Action {action!r} not executed by tool {tool.tool_type}")

        # if the action does not work with any tool, execute it in the environment
        if not valid or self.tool_use_counter >= self.max_tool_uses:
            observation, reward, terminated, truncated, info = self.env.step(action)
        if valid and self.tool_use_counter >= self.max_tool_uses:
            if verbose:
                print(
                    f"Tried to use tool {tool.tool_type} but max tool uses {self.tool_use_counter} has been reached."
                )
            observation = (
                f"Tool {tool.tool_type} has reached its maximum usage limit of {self.max_tool_uses}. "
                + observation
            )
        return observation, reward, terminated, truncated, info
