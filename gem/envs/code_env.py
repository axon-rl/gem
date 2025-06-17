"""Env for code datasets."""

import logging
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any, Optional, SupportsFloat, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_code_from_model
from gem.utils.sandbox import run_python

logger = logging.getLogger(__name__)


class CodeEnv(Env):
    """Built upon a dataset, serving as a single-turn env (contextual bandits)."""

    def __init__(
        self,
        dataset_name: Optional[str] = "",
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        question_key: str = "problem",
        test_key: str = "tests",
        seed: int = 0,
        max_workers: int = 5,
        max_tests: int = 12,
        **_,
    ):
        super().__init__()
        self.seed = seed
        self.question_key = question_key
        self.test_key = test_key

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

        self.dataset_name = dataset_name
        self.dataset = dataset.shuffle(seed=self.seed)
        self.dataset_iter = iter(self.dataset)

        self.thread_pool_executer = ThreadPoolExecutor(max_workers=max_workers)
        self.max_tests = max_tests

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:

        model_code = extract_code_from_model(action)
        if model_code is None:
            return -0.1
        else:
            time_st = time()
            is_correct = self._check_correct(model_code)
            logger.debug(time() - time_st)
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
        self.tests = data[self.test_key]
        return self.first_obs, {}

    def _check_correct(self, model_code: str) -> bool:
        assert any(
            [x in self.dataset_name.lower() for x in ["taco", "apps", "codecontests"]]
        )
        tests = self.tests

        # format of tests: List[Dictionary] - Codeforces, LiveCodeBench
        # format of tests: Dictionary[Lists] - CodeContests, Taco/Apps
        if isinstance(tests, list):
            raise NotImplementedError
            total_tests = len(tests)
            if total_tests > self.max_tests:
                # Sort indices by test input length and take the max_tests longest ones
                selected_indices = sorted(
                    range(total_tests),
                    key=lambda i: len(tests[i]["input"]),
                    reverse=True,
                )[: self.max_tests]
                tests = [tests[i] for i in selected_indices]
            num_tests = len(tests)
        else:
            total_tests = len(tests["inputs"])
            if total_tests > self.max_tests:
                # Select the tests with the longest input length.
                selected_indices = sorted(
                    range(total_tests),
                    key=lambda i: len(tests["inputs"][i]),
                    reverse=True,
                )[: self.max_tests]
                # Create a new dict with only the selected test cases
                selected_tests = {
                    "inputs": [tests["inputs"][i] for i in selected_indices],
                    "outputs": [tests["outputs"][i] for i in selected_indices],
                }
                tests = selected_tests
            num_tests = len(tests["inputs"])

        code_and_tests = [(model_code, test) for test in tests["inputs"]]
        results = list(
            self.thread_pool_executer.map(
                lambda args: run_python(*args), code_and_tests
            )
        )

        successes, stdouts, stderrs = zip(*results)

        if not all(successes):
            return False
        for gt, pred in zip(tests["outputs"], stdouts):
            if gt != pred:
                return False
        return True
