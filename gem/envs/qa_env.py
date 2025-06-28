"""Env for question answering datasets."""

import logging
from typing import Any, Optional, SupportsFloat, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_tagged_answer
from gem.utils.qa_em import em_check

logger = logging.getLogger(__name__)


def apply_prompt(example):
    prompt_template = "Answer the given question. Question: {question}\n"
    example["question"] = prompt_template.format(question=example["question"])
    return example


class QaEnv(Env):
    """Built upon a dataset, serving as a single-turn env (contextual bandits)."""

    def __init__(
        self,
        dataset_name: Optional[str] = "",
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        question_key: str = "question",
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
        dataset = dataset.map(apply_prompt)
        self.dataset = dataset.shuffle(seed=self.seed)
        self.idx = 0
        self.epoch = 0

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = extract_last_tagged_answer(action)
        if model_answer is None:
            reward = -0.1
        else:
            is_correct = self.check_correct(model_answer, self.answer)
            reward = 1.0 if is_correct else 0.0
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
    def check_correct(model_answer: str, gt_answer: Any) -> bool:
        """Check if the action is correct using exact match."""
        # get correct answers from the dataset entry
        if isinstance(gt_answer, (str, float, int)):
            correct_answers = [str(gt_answer)]
        elif isinstance(gt_answer, list):
            correct_answers = [str(ans) for ans in gt_answer]
        else:
            raise ValueError(f"Unexpected answer type: {type(gt_answer)}")

        is_correct = em_check(model_answer, correct_answers)
        return is_correct

    def sample_random_action(self) -> str:
        """Sample a random action."""
        return "<answer>Hello World!</answer>"


if __name__ == "__main__":
    print("--- Testing QaEnv.check_correct (which uses em_check) ---")

    assert QaEnv.check_correct("Paris", ["Paris"]) is True
    print("Test 1 Passed: Simple correct answer.")

    assert QaEnv.check_correct("London", ["Paris"]) is False
    print("Test 2 Passed: Simple incorrect answer.")

    assert QaEnv.check_correct("paris", ["Paris"]) is True
    print("Test 3 Passed: Correct answer with different casing.")

    assert QaEnv.check_correct("  Paris  ", ["Paris"]) is True
    print("Test 4 Passed: Correct answer with whitespace.")

    assert QaEnv.check_correct("Berlin", ["Paris", "Berlin", "Rome"]) is True
    print("Test 5 Passed: One of multiple answers is correct.")

    assert QaEnv.check_correct("42", [42, "some other answer"]) is True
    print("Test 6 Passed: Integer answer treated as string.")

    assert QaEnv.check_correct("43", [42, "some other answer"]) is False
    print("Test 7 Passed: Incorrect integer answer.")

    print("--- All tests passed! ---")
