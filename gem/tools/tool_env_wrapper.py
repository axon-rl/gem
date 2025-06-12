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

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        self.tool_use_counter = 0
        obs, info = self.env.reset(seed=seed)
        tool_instructions = "\n".join(
            [tool.instruction_string() for tool in self.tools]
        )
        if len(tool_instructions) > 1:
            tool_instructions = f"{obs}\n\nAvailable tools:\n{tool_instructions}"
        obs = f"{obs}\n{tool_instructions}"
        return obs, info


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
        print(f"Tool use counter: {self.tool_use_counter}, Max tool uses: {self.max_tool_uses}")
        return observation, reward, terminated, truncated, info
