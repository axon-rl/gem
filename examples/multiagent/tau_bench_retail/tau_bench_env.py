#!/usr/bin/env python3
import json
import os
import sys
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion
from pydantic import BaseModel

from gem.envs.multiagent import MultiAgentEnv
from gem.envs.multiagent.multi_agent_env import AgentSelector

TAU_BENCH_PATH = os.environ.get("TAU_BENCH_PATH", os.path.join(os.path.dirname(__file__), "tau-bench"))
ASSETS_PATH = os.path.join(TAU_BENCH_PATH, "tau_bench/envs/retail")

if not os.path.exists(ASSETS_PATH):
    raise FileNotFoundError(
        f"TAU-bench repository not found. Please either:\n"
        f"1. Clone https://github.com/sierra-research/tau-bench to {TAU_BENCH_PATH}\n"
        f"2. Set TAU_BENCH_PATH environment variable to the cloned repository path"
    )

if ASSETS_PATH not in sys.path:
    sys.path.insert(0, ASSETS_PATH)

from data import load_data
from tools import ALL_TOOLS
from wiki import WIKI


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]


class Task(BaseModel):
    user_id: str
    actions: List[Action]
    instruction: str
    outputs: List[str]


class TauBenchEnv(MultiAgentEnv):
    """TAU-bench Retail environment using GEM MultiAgentEnv API"""

    def __init__(
        self,
        task_split: str = "test",
        user_model: str = "gpt-4o",
        user_provider: str = "openai",
    ):
        super().__init__()
        self.task_split = task_split
        self.user_model = user_model
        self.user_provider = user_provider

        self.possible_agents = ["assistant"]
        self.agent_selector = AgentSelector(self.possible_agents, mode="sequential")

        self.data = load_data()
        self.wiki = WIKI
        self.tool_definitions = [tool.get_info() for tool in ALL_TOOLS]
        self.tools_map = {
            tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS
        }
        self.tasks = self._load_tasks()
        self.terminate_tools = ["transfer_to_human_agents"]

        self.task_index = 0
        self.task = None
        self.user_messages = []
        self.actions_taken = []

    def _load_tasks(self) -> List[Task]:
        if self.task_split == "test":
            from tasks_test import TASKS_TEST

            return TASKS_TEST
        elif self.task_split == "train":
            from tasks_train import TASKS_TRAIN

            return TASKS_TRAIN
        else:
            from tasks_dev import TASKS_DEV

            return TASKS_DEV

    def reset(
        self, seed: Optional[int] = None, task_index: Optional[int] = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        observations, infos = super().reset(seed=seed)

        self.task_index = task_index if task_index is not None else 0
        self.task = self.tasks[self.task_index]
        self.data = load_data()
        self.actions_taken = []

        user_system_prompt = f"""You are a user interacting with an agent.

Instruction: {self.task.instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

        self.user_messages = [
            {"role": "system", "content": user_system_prompt},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]

        initial_user_obs = self._simulate_user()
        observations["assistant"] = initial_user_obs
        infos["assistant"] = {"task": self.task.model_dump()}

        return observations, infos

    def observe(self, agent: str) -> str:
        if agent == "assistant" and self.user_messages and len(self.user_messages) > 2:
            return self.user_messages[-1]["content"]
        return ""

    def _simulate_user(self) -> str:
        response = completion(
            model=self.user_model,
            custom_llm_provider=self.user_provider,
            messages=self.user_messages,
        )
        msg = response.choices[0].message
        self.user_messages.append(msg.model_dump())
        return msg.content

    def _process_actions(self, actions: Dict[str, str]) -> Tuple[
        Dict[str, str],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        observations = {}
        rewards = {"assistant": 0.0}
        terminations = {"assistant": False}
        truncations = {"assistant": False}
        infos = {"assistant": {}}

        if "assistant" in actions:
            action_str = actions["assistant"]

            try:
                action_dict = json.loads(action_str)
                action = Action(**action_dict)
            except:
                action = Action(name="respond", kwargs={"content": action_str})

            self.actions_taken.append(action)

            if action.name == "respond":
                self.user_messages.append(
                    {"role": "user", "content": action.kwargs["content"]}
                )
                user_response = self._simulate_user()

                observations["assistant"] = user_response
                infos["assistant"]["source"] = "user"

                if "###STOP###" in user_response:
                    terminations["assistant"] = True
                    rewards["assistant"] = self._calculate_reward()

            elif action.name in self.tools_map:
                try:
                    observation = self.tools_map[action.name].invoke(
                        data=self.data, **action.kwargs
                    )
                except Exception as e:
                    observation = f"Error: {e}"

                observations["assistant"] = observation
                infos["assistant"]["source"] = action.name

                if action.name in self.terminate_tools:
                    terminations["assistant"] = True
                    rewards["assistant"] = self._calculate_reward()

            else:
                observations["assistant"] = f"Unknown action {action.name}"
                infos["assistant"]["source"] = action.name

        return observations, rewards, terminations, truncations, infos

    def _calculate_reward(self) -> float:
        def to_hashable(item):
            if isinstance(item, dict):
                return tuple(
                    (key, to_hashable(value)) for key, value in sorted(item.items())
                )
            elif isinstance(item, list):
                return tuple(to_hashable(element) for element in item)
            elif isinstance(item, set):
                return tuple(sorted(to_hashable(element) for element in item))
            else:
                return item

        def get_data_hash():
            return sha256(str(to_hashable(self.data)).encode("utf-8")).hexdigest()

        data_hash = get_data_hash()
        self.data = load_data()
        saved_user_messages = self.user_messages[:]

        for gt_action in self.task.actions:
            if gt_action.name not in self.terminate_tools:
                self.actions_taken.append(gt_action)
                if gt_action.name != "respond" and gt_action.name in self.tools_map:
                    try:
                        self.tools_map[gt_action.name].invoke(
                            data=self.data, **gt_action.kwargs
                        )
                    except:
                        pass

        self.user_messages = saved_user_messages
        gt_data_hash = get_data_hash()
        reward = 1.0 if data_hash == gt_data_hash else 0.0

        if len(self.task.outputs) > 0:
            for output in self.task.outputs:
                found = any(
                    action.name == "respond"
                    and output.lower()
                    in action.kwargs.get("content", "").lower().replace(",", "")
                    for action in self.actions_taken
                )
                if not found:
                    reward = 0.0
                    break

        return reward
