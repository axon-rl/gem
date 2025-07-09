import reasoning_gym as rg

from gem.envs.registration import register

# Register games from our implementation of TextArena
# GuessTheNumber
register(
    "ta:GuessTheNumber-v0",
    "gem.envs.textarena.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=20,
    max_turns=10,
)
register(
    "ta:GuessTheNumber-v0-hard",
    "gem.envs.textarena.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=50,
    max_turns=7,
)
register(
    "ta:GuessTheNumber-v0-easy",
    "gem.envs.textarena.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=10,
    max_turns=4,
)
register(
    "ta:GuessTheNumber-v0-random",
    "gem.envs.textarena.guess_the_number:GuessTheNumberEnv",
    min_number=None,
    max_number=None,
    max_turns=None,
)
# Mastermind
register(
    "ta:Mastermind-v0",
    "gem.envs.textarena.mastermind:MastermindEnv",
    code_length=4,
    num_numbers=6,
    max_turns=20,
    duplicate_numbers=False,
)
register(
    "ta:Mastermind-v0-hard",
    "gem.envs.textarena.mastermind:MastermindEnv",
    code_length=4,
    num_numbers=8,
    max_turns=30,
    duplicate_numbers=False,
)
register(
    "ta:Mastermind-v0-random",
    "gem.envs.textarena.mastermind:MastermindEnv",
    code_length=None,
    num_numbers=None,
    max_turns=50,
    duplicate_numbers=False,
)
register(
    "ta:Mastermind-v0-easy",
    "gem.envs.textarena.mastermind:MastermindEnv",
    code_length=2,
    num_numbers=6,
    max_turns=6,
    duplicate_numbers=False,
)
# Minesweeper
register(
    "ta:Minesweeper-v0",
    "gem.envs.textarena.minesweeper:MinesweeperEnv",
    rows=8,
    cols=8,
    num_mines=10,
    max_turns=100,
)
register(
    "ta:Minesweeper-v0-veryeasy",
    "gem.envs.textarena.minesweeper:MinesweeperEnv",
    rows=3,
    cols=3,
    num_mines=1,
    max_turns=10,
)
register(
    "ta:Minesweeper-v0-easy",
    "gem.envs.textarena.minesweeper:MinesweeperEnv",
    rows=5,
    cols=5,
    num_mines=3,
    max_turns=10,
)
register(
    "ta:Minesweeper-v0-hard",
    "gem.envs.textarena.minesweeper:MinesweeperEnv",
    rows=12,
    cols=12,
    num_mines=30,
    max_turns=100,
)
register(
    "ta:Minesweeper-v0-random",
    "gem.envs.textarena.minesweeper:MinesweeperEnv",
    rows=None,
    cols=None,
    num_mines=None,
    max_turns=None,
)
# Wordle
register(
    "ta:Wordle-v0",
    "gem.envs.textarena.wordle:WordleEnv",
    word_length=5,
    hardcore=False,
    only_real_words=True,
    max_turns=6,
)
register(
    "ta:Wordle-v0-easy",
    "gem.envs.textarena.wordle:WordleEnv",
    word_length=3,
    hardcore=False,
    only_real_words=True,
    max_turns=6,
)
register(
    "ta:Wordle-v0-random",
    "gem.envs.textarena.wordle:WordleEnv",
    word_length=None,
    hardcore=False,
    only_real_words=True,
    max_turns=8,
)
register(
    "ta:Wordle-v0-lenient",
    "gem.envs.textarena.wordle:WordleEnv",
    word_length=5,
    hardcore=False,
    only_real_words=False,
    max_turns=8,
)
register(
    "ta:FifteenPuzzle-v0",
    "gem.envs.textarena.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=3,
    max_turns=20,
)
register(
    "ta:FifteenPuzzle-v0-easy",
    "gem.envs.textarena.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=2,
    max_turns=10,
)
register(
    "ta:FifteenPuzzle-v0-hard",
    "gem.envs.textarena.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=4,
    max_turns=50,
)


# Register math dataset environments

# import logging
# import time

# from datasets import load_dataset

# _wait_time = 5
# _num_retries = 10

# for i in range(_num_retries):
#     # Retry in case network error when accessing HF.
#     try:
#         math_12k_dataset = load_dataset("axon-rl/MATH-12k", split="train")
#         break
#     except Exception as e:
#         # In case of timeout.
#         time.sleep(_wait_time)
#         _wait_time *= 1.2
#         logging.warning(f"{e}")
#         logging.warning(f"Try {i}/{_num_retries}. Trying again...")
# else:
#     raise RuntimeError("Cannot load axon-rl/MATH-12k dataset")
# register(
#     "math:Math12k-v0",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/MATH-12k",
#     dataset=math_12k_dataset,
#     question_key="problem",
#     answer_key="answer",
#     unformatted_penalty=-0.1,
#     formatted_reward=0.0,
#     is_correct_reward=1.0,
# )
# register(
#     "math:Math12k-v1",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/MATH-12k",
#     dataset=math_12k_dataset,
#     question_key="problem",
#     answer_key="answer",
#     unformatted_penalty=0.0,
#     formatted_reward=0.1,
#     is_correct_reward=1.0,
# )

# for i in range(_num_retries):
#     # Retry in case network error when accessing HF.
#     try:
#         asdiv_2k_dataset = load_dataset("axon-rl/ASDIV-2k", split="train")
#         break
#     except Exception as e:
#         # In case of timeout.
#         time.sleep(_wait_time)
#         _wait_time *= 1.2
#         logging.warning(f"{e}")
#         logging.warning(f"Try {i}/{_num_retries}. Trying again...")
# else:
#     raise RuntimeError("Cannot load axon-rl/ASDIV-2k dataset")
# register(
#     "math:ASDIV2k-v0",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/ASDIV-2k",
#     dataset=asdiv_2k_dataset,
#     question_key="problem",
#     answer_key="answer",
#     unformatted_penalty=-0.1,
#     formatted_reward=0.0,
#     is_correct_reward=1.0,
# )
# register(
#     "math:ASDIV2k-v1",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/ASDIV-2k",
#     dataset=asdiv_2k_dataset,
#     question_key="problem",
#     answer_key="answer",
#     unformatted_penalty=0.0,
#     formatted_reward=0.1,
#     is_correct_reward=1.0,
# )


# for i in range(_num_retries):
#     # Retry in case network error when accessing HF.
#     try:
#         gsm_8k_dataset = load_dataset("axon-rl/GSM-8k", split="train")
#         break
#     except Exception as e:
#         # In case of timeout.
#         time.sleep(_wait_time)
#         _wait_time *= 1.2
#         logging.warning(f"{e}")
#         logging.warning(f"Try {i}/{_num_retries}. Trying again...")
# else:
#     raise RuntimeError("Cannot load axon-rl/GSM-8k dataset")
# register(
#     "math:GSM8k-v0",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/GSM-8k",
#     dataset=gsm_8k_dataset,
#     question_key="problem",
#     answer_key="answer",
#     unformatted_penalty=-0.1,
#     formatted_reward=0.0,
#     is_correct_reward=1.0,
# )
# register(
#     "math:GSM8k-v1",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/GSM-8k",
#     dataset=gsm_8k_dataset,
#     question_key="problem",
#     answer_key="answer",
#     unformatted_penalty=0.0,
#     formatted_reward=0.1,
#     is_correct_reward=1.0,
# )

# for i in range(_num_retries):
#     # Retry in case network error when accessing HF.
#     try:
#         orz_57k_dataset = load_dataset("axon-rl/ORZ-57k", split="train")
#         break
#     except Exception as e:
#         # In case of timeout.
#         time.sleep(_wait_time)
#         _wait_time *= 1.2
#         logging.warning(f"{e}")
#         logging.warning(f"Try {i}/{_num_retries}. Trying again...")
# else:
#     raise RuntimeError("Cannot load axon-rl/ORZ-57k dataset")
# register(
#     "math:ORZ57k-v0",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/ORZ-57k",
#     dataset=orz_57k_dataset,
#     question_key="problem",
#     answer_key="answer",
#     unformatted_penalty=-0.1,
#     formatted_reward=0.0,
#     is_correct_reward=1.0,
# )
# register(
#     "math:ORZ57k-v1",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/ORZ-57k",
#     dataset=orz_57k_dataset,
#     question_key="problem",
#     answer_key="answer",
#     unformatted_penalty=0.0,
#     formatted_reward=0.1,
#     is_correct_reward=1.0,
# )

register(
    "math:GSM8K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/GSM-8k",
    question_key="problem",
    answer_key="answer",
)

# Register code dataset environments

register(
    "code:CodeContest",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/CodeContest",
    split="train",
    question_key="problem",
    test_key="tests",
)

register(
    "code:Taco8k",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/TACO-8k",
    split="train",
    question_key="problem",
    test_key="tests",
)

register(
    "code:PrimeIntellect15k",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/PrimeIntellect-15k",
    split="train",
    question_key="problem",
    test_key="tests",
)

# Register qa dataset environments

for i in [0, 1, 2, 3, 5]:
    register(
        f"logic:RuleTaker-d{i}",
        "gem.envs.qa_env:QaEnv",
        dataset_name=f"axon-rl/RuleTaker-d{i}-70k",
        split="train",
        extract_boxed=True,
        question_key="question",
        answer_key="answer",
    )

register(
    "qa:NaturalQuestions",
    "gem.envs.qa_env:QaEnv",
    dataset_name="/path/to/project/gem/resources/datasets/NaturalQuestions",
    split="train",
    question_key="problem",
    answer_key="answer",
)

register(
    "qa:HotpotQA",
    "gem.envs.qa_env:QaEnv",
    dataset_name="/path/to/project/gem/resources/datasets/HotpotQA",
    split="train",
    question_key="problem",
    answer_key="answer",
)

register(
    "qa:NaturalQuestions-v1",
    "gem.envs.qa_env:QaEnv",
    dataset_name="/path/to/project/gem/resources/datasets/NaturalQuestions",
    split="train",
    question_key="problem",
    answer_key="answer",
    unformatted_penalty=0.0,
    formatted_reward=0.0,
    is_correct_reward=1.0,
)

register(
    "qa:HotpotQA-v1",
    "gem.envs.qa_env:QaEnv",
    dataset_name="/path/to/project/gem/resources/datasets/HotpotQA",
    split="train",
    question_key="problem",
    answer_key="answer",
    unformatted_penalty=0.0,
    formatted_reward=0.0,
    is_correct_reward=1.0,
)

register(
    "qa:NaturalQuestions-v2",
    "gem.envs.qa_env:QaEnv",
    dataset_name="/path/to/project/gem/resources/datasets/NaturalQuestions",
    split="train",
    question_key="problem",
    answer_key="answer",
    unformatted_penalty=0.0,
    formatted_reward=0.2,
    is_correct_reward=1.0,
)

register(
    "qa:HotpotQA-v2",
    "gem.envs.qa_env:QaEnv",
    dataset_name="/path/to/project/gem/resources/datasets/HotpotQA",
    split="train",
    question_key="problem",
    answer_key="answer",
    unformatted_penalty=0.0,
    formatted_reward=0.2,
    is_correct_reward=1.0,
)

# Register datasets from ReasoningGym

for name in rg.factory.DATASETS.keys():
    register(
        f"rg:{name}",
        "gem.envs.reasoning_gym:ReasoningGymEnv",
        name=name,
        size=500,
        seed=42,
    )

# Register evaluation datasets

## MATH500
# for i in range(_num_retries):
#     # Retry in case network error when accessing HF.
#     try:
#         math_500_dataset = load_dataset("axon-rl/Eval-MATH500", split="test")
#         break
#     except Exception as e:
#         # In case of timeout.
#         time.sleep(_wait_time)
#         _wait_time *= 1.2
#         logging.warning(f"{e}")
#         logging.warning(f"Try {i}/{_num_retries}. Trying again...")
# else:
#     raise RuntimeError("Cannot load axon-rl/Eval-MATH500 dataset")
# register(
#     "eval:MATH500",
#     "gem.envs.math_env:MathEnv",
#     # dataset_name="axon-rl/Eval-MATH500",
#     # split="test"
#     dataset=math_500_dataset,
#     question_key="problem",
#     answer_key="answer",
# )

## The test split of deepmind/code_contests, with merged test cases.
register(
    "eval:CodeContest",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/CodeContest",
    split="test",
    question_key="problem",
    test_key="tests",
)

## QaOpen
register(
    "eval:QaOpen",
    "gem.envs.qa_env:QaEnv",
    dataset_name="google-research-datasets/nq_open",
    split="validation",
    question_key="question",
    answer_key="answer",
)

# Dummy env
register(
    "dummy:DummyEnv",
    "gem.envs.dummy_env:DummyEnv",
)
