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

"""Env for math datasets."""

import logging
import multiprocessing
from typing import Any, Optional, SupportsFloat, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from math_verify import parse, verify

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer

logger = logging.getLogger(__name__)


class MathEnv(Env):
    """Built upon a dataset, serving as a single-turn env (contextual bandits)."""

    def __init__(
        self,
        dataset_name: Optional[str] = "",
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        question_key: str = "problem",
        answer_key: str = "answer",
        seed: int = 0,
        **_,
    ):
        super().__init__()
        self.seed = seed
        self.question_key = question_key
        self.answer_key = answer_key
        if dataset is None:
            dataset = load_dataset(dataset_name)
            logger.info(f"Loaded: {dataset=}")
        if isinstance(dataset, DatasetDict):
            if split is not None:
                dataset = dataset[split]
            elif len(list(dataset.keys())) == 1:
                dataset = dataset[list(dataset.keys())[0]]
            else:
                raise ValueError(
                    f"Dataset {dataset_name} has multiple splits. "
                    f"Please specify a split: {list(dataset.keys())}"
                )
        assert isinstance(dataset, Dataset), f"Expected a Dataset, got {type(dataset)}"
        self.dataset = dataset.shuffle(seed=self.seed)
        self.idx = 0
        self.epoch = 0
        # Process pool is used to enable the timeout mechanism for answer grading in a potential distributed training setup
        self.mp_pool = multiprocessing.Pool(1)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = extract_last_boxed_answer(action)
        if model_answer is None:
            reward = 0
        else:
            res = self.mp_pool.apply_async(
                self.check_correct, (model_answer, self.answer)
            )
            try:
                is_correct = res.get(timeout=1)
            except multiprocessing.context.TimeoutError:
                is_correct = False
            reward = 1.0 if is_correct else 0
        return TERMINAL_STATE, reward, True, True, {}

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        del seed

        if self.idx == len(self.dataset):
            self.epoch += 1
            self.dataset = self.dataset.shuffle(seed=self.seed + self.epoch)
            self.idx = 0

        data = self.dataset[self.idx]
        self.first_obs = data[self.question_key]
        self.answer = data[self.answer_key]
        self.idx += 1
        return self.first_obs, {}

    @staticmethod
    def check_correct(model_answer: str, gt_answer: str) -> bool:
        """Check if the action is correct."""
        # parse with math_verify
        model_answer = parse(model_answer)

        # get correct answers from the dataset entry
        if isinstance(gt_answer, (str, float, int)):
            correct_answers = [str(gt_answer)]
        elif isinstance(gt_answer, list):
            correct_answers = gt_answer
        else:
            raise ValueError(f"Unexpected answer type: {type(gt_answer)}")

        # check against all possible correct answers
        # (math_verify.parse handles extraction e.g. from \\boxed{...})
        is_correct = False
        for correct_answer in correct_answers:
            correct_answer = parse(str(correct_answer))
            if verify(correct_answer, model_answer):
                is_correct = True
                break
        return is_correct

    def sample_random_action(self) -> str:
        """Sample a random action."""
        return "\\boxed{42}"


if __name__ == "__main__":
    ans1 = parse("${1,2,3,4}$")
    ans2 = parse("${1,3} \\cup {2,4}$")
    ans3 = parse("\\boxed{1,2,3,4}")
    ans4 = parse("none")
    ans5 = parse(None)
    ans6 = parse({1, 2, 3, 4})
    ans7 = parse(str({1, 2, 3, 4}))

    print(f"{ans1} == {ans2}: {verify(ans1, ans2)}")  # >> True
    print(f"{ans2} == {ans1}: {verify(ans2, ans1)}")  # >> True
    print(f"{ans1} == {ans3}: {verify(ans1, ans3)}")  # >> True
    print(f"{ans3} == {ans1}: {verify(ans3, ans1)}")  # >> True
    print(f"{ans2} == {ans3}: {verify(ans2, ans3)}")  # >> True
    print(f"{ans3} == {ans2}: {verify(ans3, ans2)}")  # >> True
    print(f"{ans1} == {ans4}: {verify(ans1, ans4)}")  # >> False
    print(f"{ans1} == {ans5}: {verify(ans1, ans5)}")  # >> False
    print(f"{ans1} == {ans6}: {verify(ans1, ans6)}")  # >> False
    print(f"{ans1} == {ans7}: {verify(ans1, ans7)}")  # >> False

    ans1 = parse("1")
    # ans2 = parse(2) # >> `TypeError: expected string or bytes-like object`
    ans3 = parse("2k+2")
    ans4 = parse("\\frac9{19}")
