"""Env for math datasets."""

import logging
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
        self.dataset_iter = iter(self.dataset)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = extract_last_boxed_answer(action)
        if model_answer is None:
            reward = -0.1
        else:
            is_correct = self._check_correct(model_answer)
            reward = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, True, {}

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        del seed

        try:
            data = next(self.dataset_iter)
        except StopIteration:
            self.dataset = self.dataset.shuffle(seed=self.seed)
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)

        self.first_obs = data[self.question_key]
        self.answer = data[self.answer_key]
        return self.first_obs, {}

    def _check_correct(self, model_answer: str) -> bool:
        """Check if the action is correct."""
        # parse with math_verify
        model_answer = parse(model_answer)

        # get correct answers from the dataset entry
        if isinstance(self.answer, (str, float, int)):
            correct_answers = [str(self.answer)]
        elif isinstance(self.answer, list):
            correct_answers = self.answer
        else:
            raise ValueError(f"Unexpected answer type: {type(self.answer)}")

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
