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

"""
Adds support for enforcing specific response formats from the environment.

Each wrapper caters to a different kind of format enforcement,
although for now only EncapsulateWrapper is implemented, where
the response from the environment is encapsulated in some regexable syntax.
"""
# import re
from abc import abstractmethod
from typing import Any, Optional, Tuple

from gem.core import ActType, Env, EnvWrapper
from gem.wrappers.observation_wrapper import WrapperObsType

# TODO: Refactor
def maybe_add_new_line(text: str):
    if text and not text.endswith("\n"):
        return text + "\n"
    return text

class FormatWrapper(EnvWrapper):
    MALFORMED_ACTION = "MALFORMED_ACTION"
    def __init__(self, env: Env):
        super().__init__(env)

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        formatted_obs = self._get_format_instruction(obs)
        return formatted_obs, info
    
    def step(
        self, raw_action: ActType
    ) -> Tuple[WrapperObsType, Any, bool, bool, dict[str, Any]]:
        try:
            enforced_action = self._enforce_format(raw_action)
        except ValueError as e:
            enforced_action = self.MALFORMED_ACTION
        return self.env.step(enforced_action)
    
    def sample_random_action(self):
        valid_action = self.env.sample_random_action()
        return self._get_format(valid_action)
    
    @abstractmethod
    def _get_format(self, action: ActType) -> ActType:
        """
        Applies the format to a VALID action.
        Args:
            action (ActType): A valid action that can be wrapped in the expected format.

        Returns:
            ActType: The action wrapped in the expected format.
        """
        pass

    @abstractmethod
    def _get_format_instruction(self, observation: WrapperObsType) -> WrapperObsType:
        """Adds the format rule to the observation.

        Args:
            observation (WrapperObsType): The original observation from the environment.

        Returns:
            WrapperObsType: The observation with the format rule added.
        """
        pass

    @abstractmethod
    def _enforce_format(self, raw_action: ActType) -> ActType:
        """
        Parses the raw action, under the assumption that the action follows the enforced format.

        Args:
            action (ActType): The raw action from the agent.

        Returns:
            ActType: The extracted action that conforms to the expected format.
        """
        pass

class EncapsulateWrapper(FormatWrapper):
    def __init__(
        self,
        env: Env,
        prefix: str = "<<<",
        suffix: str = ">>>",
    ):
        super().__init__(env)
        self.prefix = prefix
        self.suffix = suffix
        # self.regex = re.compile(f"{re.escape(self.prefix)}(.*?){re.escape(self.suffix)}", re.DOTALL)

    def find_last_encapsulated(self, text: str) -> Optional[str]:
        """
        Extracts the last and innermost encapsulated substring from the given text.

        Args:
            text (str): The input string containing encapsulated substrings.
        Returns:
            str or None: The extracted substring, or None if not found.
        """

        # Note that regex is incapable of matching nested patterns (correctly and quickly).
        # So instead we use the same approach as `extract_last_boxed_only_string`.
        idx = text.rfind(self.prefix)
        if idx < 0:
            return None
        end_idx = len(text)
        while True:
            next_end_tag_idx = text.rfind(self.suffix, 0, end_idx)
            if next_end_tag_idx < idx:
                break
            end_idx = next_end_tag_idx
        return text[idx + len(self.prefix) : end_idx].strip()

    def _get_format(self, action: ActType) -> ActType:
        """
        Encapsulates a valid action within the specified prefix and suffix.

        Args:
            action (ActType): A valid action that can be wrapped in the expected format.
        Returns:
            ActType: The action wrapped in the expected format.
        """
        return f"{self.prefix}{action}{self.suffix}"
    
    def _get_format_instruction(self, observation: WrapperObsType) -> WrapperObsType:
        """
        Encapsulates the observation in the specified prefix and suffix.

        Args:
            observation (WrapperObsType): The original observation from the environment.
        Returns:
            WrapperObsType: The observation, including a specification that the response
            should be encapsulated within the given prefix and suffix.
        """
        new_obs = maybe_add_new_line(observation)
        new_obs += (
            f"Please give your FINAL response in the following format: {self.prefix}<your response>{self.suffix}\n"
            "If you give multiple responses, only the last (and innermost) one will be considered. "
        )
        return new_obs
    
    def _enforce_format(self, raw_action: ActType) -> ActType:
        """
        Extracts the action encapsulated within the specified prefix and suffix.

        Args:
            raw_action (ActType): The raw action from the agent.
        Returns:
            ActType: The extracted action within the prefix and suffix.
        """
        enforced_action = self.find_last_encapsulated(raw_action)
        if enforced_action is None:
            raise ValueError(
                "Action does not follow the expected format: "
                f"{self.prefix}<your response>{self.suffix}"
            )
        return enforced_action