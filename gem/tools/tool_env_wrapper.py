from typing import Any, Optional, SupportsFloat, Tuple, TypeVar, List
from gem.core import EnvWrapper, Env
from gem.tools.base_tool import BaseTool

class ToolEnvWrapper(EnvWrapper):
    def __init__(self, env: Env, tools: List[BaseTool], tool_use_reward: float = 0.1):
        super().__init__(env)
        self.tools = tools
        self.tool_use_reward = tool_use_reward

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        # try to execute the action with each tool
        for tool in self.tools:
            observation, terminated, valid = tool.execute_action(action)
            if valid:
                print(f"Action {action} executed by tool {tool.tool_type}")
                reward = self.tool_use_reward
                truncated = False
                info = {}
                break
        # if the action does not work with any tool, execute it in the environment
        if not valid:
            observation, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.append(observation)
        return self._get_wrapped_obs(), reward, terminated, truncated, info