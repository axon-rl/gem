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

import fire
from typing import Any, Dict, Tuple

from gem.core import Env
from gem.wrappers.format_wrapper import EncapsulateWrapper
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY, get_wrapper_fns

class DummyEnv(Env):
    """A simple dummy environment for testing wrappers."""

    def __init__(self):
        super().__init__()
        self.received_action = None
        self.turn_count = 0

    def reset(self, seed=None) -> Tuple[str, Dict[str, Any]]:
        self.received_action = None
        self.turn_count = 0
        return "initial_observation", {}
    
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.received_action = action
        self.turn_count += 1
        done = self.turn_count >= 5
        obs = f"observation_{self.turn_count}"
        if done:
            obs += "_(final)"
        return obs, 0.0, done, done, {}
    
    def sample_random_action(self) -> str:
        return "random action"

def test_encapsulate_wrapper():
    """
    Test that EncapsulateWrapper correctly 
    1. accepts only actions that follow the specified format,
    2. modifies sampled random actions to follow the specified format.
    """
    print("Testing EncapsulateWrapper...")
    env = DummyEnv()
    
    # 1. Test answer_boxed
    # TEST 1A: Expects action "x" to be rejected for incorrect format
    boxed_wrapper = get_wrapper_fns(
        "answer_boxed"
    )[0]
    wrapped_env = boxed_wrapper(env)
    assert isinstance(wrapped_env, EncapsulateWrapper), (
        "Wrapper is not an instance of EncapsulateWrapper. "
        f"Got {type(wrapped_env)} instead."
    )
    wrapped_env.reset()

    ortti = wrapped_env.step("This is an invalid action")
    assert wrapped_env.env.received_action == wrapped_env.MALFORMED_ACTION, (
        "(TEST 1A) Invalid action was not rejected properly. "
        f"Expected \"\", got: \"{wrapped_env.env.received_action}\""
    )
    
    # TEST 1B: Expects action "\\boxed{valid action}" to be accepted
    valid_action = "\\boxed{valid action}"
    ortti = wrapped_env.step(valid_action)
    assert wrapped_env.env.received_action == "valid action", (
        "(TEST 1B) Valid action was not accepted and parsed properly. "
        f"Expected \"valid action\", got: \"{wrapped_env.env.received_action}\""
    )

    # TEST 1C: Test that sampled random action is modified to follow format
    random_action = wrapped_env.sample_random_action()
    assert random_action.startswith("\\boxed{") and random_action.endswith("}"), (
        "(TEST 1C) Sampled random action was not formatted correctly. "
        f"Got: \"{random_action}\""
    )

    # TEST 1D: Test edge cases with multiple boxes
    multi_boxed_action = r"\boxed{first} some text \boxed{second}"
    ortti = wrapped_env.step(multi_boxed_action)
    assert wrapped_env.env.received_action == "second", (
        "(TEST 1D) Action with multiple boxes was not parsed correctly. "
        f"Expected \"second\", got: \"{wrapped_env.env.received_action}\""
    )

    # TEST 1E: Test edge case with nested boxes
    nested_boxed_action = r"\boxed{outer \boxed{inner}}"
    ortti = wrapped_env.step(nested_boxed_action)
    assert wrapped_env.env.received_action == r"inner", (
        "(TEST 1E) Action with nested boxes was not parsed correctly. "
        f"Expected \"outer \\boxed{{inner}}\", got: \"{wrapped_env.env.received_action}\""
    )

    # Ditto for `answer_tags`
    # TEST 2A: Expects action "x" to be rejected for incorrect format
    tags_wrapper = get_wrapper_fns(
        "answer_tags"
    )[0]
    wrapped_env = tags_wrapper(env)
    assert isinstance(wrapped_env, EncapsulateWrapper), (
        "Wrapper is not an instance of EncapsulateWrapper. "
        f"Got {type(wrapped_env)} instead."
    )
    wrapped_env.reset() 
    ortti = wrapped_env.step("This is an invalid action")
    assert wrapped_env.env.received_action == wrapped_env.MALFORMED_ACTION, (
        "(TEST 2A) Invalid action was not rejected properly. "
        f"Expected \"\", got: \"{wrapped_env.env.received_action}\""
    )   
    # TEST 2B: Expects action "<answer>valid action</answer>" to be accepted
    valid_action = "<answer>valid action</answer>"
    ortti = wrapped_env.step(valid_action)
    assert wrapped_env.env.received_action == "valid action", (
        "(TEST 2B) Valid action was not accepted and parsed properly. "
        f"Expected \"valid action\", got: \"{wrapped_env.env.received_action}\""
    )
    # TEST 2C: Test that sampled random action is modified to follow format
    random_action = wrapped_env.sample_random_action()
    assert random_action.startswith("<answer>") and random_action.endswith("</answer>"), (
        "(TEST 2C) Sampled random action was not formatted correctly. "
        f"Got: {random_action}"
    )

    # TEST 2D: Test edge cases with multiple boxes
    multi_ans_action = "<answer>first</answer> some text <answer>second</answer>"
    ortti = wrapped_env.step(multi_ans_action)
    assert wrapped_env.env.received_action == "second", (
        "(TEST 2D) Action with multiple boxes was not parsed correctly. "
        f"Expected \"second\", got: \"{wrapped_env.env.received_action}\""
    )

    # TEST 2E: Test edge case with nested boxes
    nested_ans_action = "<answer>outer <answer>inner</answer></answer>"
    ortti = wrapped_env.step(nested_ans_action)
    assert wrapped_env.env.received_action == "inner", (
        "(TEST 2E) Action with nested boxes was not parsed correctly. "
        f"Expected \"outer \\boxed{{inner}}\", got: \"{wrapped_env.env.received_action}\""
    )

    print("EncapsulateWrapper tests passed.")

def test_all():
    test_encapsulate_wrapper()

if __name__ == "__main__":

    '''
    Run with:
    python -m tests.test_env.test_wrappers test_encapsulate_wrapper
    python -m tests.test_env.test_wrappers test_all
    '''

    fire.Fire({
        "test_encapsulate_wrapper": test_encapsulate_wrapper,
        "test_all": test_all,
    })