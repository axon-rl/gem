#!/usr/bin/env python3
import json
from typing import Any, Dict, List

from litellm import completion


class TauBenchAgent:
    """Agent using OpenRouter-style tool calling pattern"""

    def __init__(
        self, model: str = "gpt-4o", provider: str = "openai", temperature: float = 0.0
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(
        self, env, task_index: int = 0, max_num_steps: int = 30
    ) -> Dict[str, Any]:
        observations, infos = env.reset(task_index=task_index)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": observations["assistant"]},
        ]

        reward = 0.0
        num_steps = 0

        for _ in range(max_num_steps):
            request = {
                "model": self.model,
                "messages": messages,
                "tools": env.tool_definitions,
                "temperature": self.temperature,
            }

            response = completion(custom_llm_provider=self.provider, **request)
            response_message = response.choices[0].message
            messages.append(response_message.model_dump())

            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    action_json = json.dumps({"name": tool_name, "kwargs": tool_args})
                    observations, rewards, terminations, truncations, env_infos = (
                        env.step({"assistant": action_json})
                    )

                    reward = rewards.get("assistant", 0.0)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": observations["assistant"],
                        }
                    )

                    num_steps += 1
                    if terminations.get("assistant", False):
                        break
            else:
                content = response_message.content or ""
                action_json = json.dumps(
                    {"name": "respond", "kwargs": {"content": content}}
                )

                observations, rewards, terminations, truncations, env_infos = env.step(
                    {"assistant": action_json}
                )

                reward = rewards.get("assistant", 0.0)
                messages.append({"role": "user", "content": observations["assistant"]})
                num_steps += 1

            if terminations.get("assistant", False):
                break

        return {
            "reward": reward,
            "task_id": env.task.user_id,
            "task_index": task_index,
            "num_steps": num_steps,
        }
