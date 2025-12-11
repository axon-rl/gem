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
from transformers import AutoTokenizer

import gem
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY


class HumanAgent:
    def __call__(self, observation):
        action = input(f"Observation: {observation}\nAction: ")
        random_string = "random_string"
        result = f"{random_string} \\boxed{{{action}}}"
        return result


def test(env_name: str = "webshop:Webshop-test"):
    policy = HumanAgent()
    env = gem.make(env_name)
    print("\n" * 5, "EPISODE 2: CONCATENATED OBSERVATION")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = WRAPPER_FACTORY["concat_with_action_tag"](env, tokenizer=tokenizer)
    run_and_print_episode(wrapped_env, policy)


if __name__ == "__main__":
    fire.Fire(test)
    print("\n\nAll tests run.\n\n")

    """Run with:
        python -m tests.test_env.test_webshop --env_name webshop:Webshop-test-text_rich
        python -m tests.test_env.test_webshop --env_name webshop:Webshop-train
    """
