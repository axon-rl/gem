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
    "ta:GuessTheNumber-v0-hard_wo_maxturns",
    "gem.envs.textarena.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=50,
    max_turns=50,
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
    only_real_words=True,
    max_turns=15,
)
register(
    "ta:Wordle-v0-easy",
    "gem.envs.textarena.wordle:WordleEnv",
    word_length=3,
    only_real_words=True,
    max_turns=15,
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

register(
    "ta:Hangman-v0",
    "gem.envs.textarena.hangman:HangmanEnv",
    word_length=5,
    hardcore=False,
    max_turns=15,
)
register(
    "ta:Hangman-v0-easy",
    "gem.envs.textarena.hangman:HangmanEnv",
    word_length=3,
    hardcore=False,
    max_turns=15,
)
register(
    "ta:Hangman-v0-hard",
    "gem.envs.textarena.hangman:HangmanEnv",
    word_length=7,
    hardcore=False,
    max_turns=20,
)
register(
    "ta:Sudoku-v0-easy",
    "gem.envs.textarena.sudoku:SudokuEnv",
    clues=10,
    max_turns=15,
    scale=4,
)
register(
    "ta:Sudoku-v0-easy-v2",
    "gem.envs.textarena.sudoku:SudokuEnv",
    clues=10,
    max_turns=15,
    scale=4,
    invalid_action_reward_type="v2",
)
register(
    "ta:Sudoku-v0",
    "gem.envs.textarena.sudoku:SudokuEnv",
    clues=50,
    max_turns=50,
    scale=9,
)
register(
    "ta:TowerofHanoi-v0-easy",
    "gem.envs.textarena.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=3,
    max_turns=10,
)
register(
    "ta:TowerofHanoi-v0",
    "gem.envs.textarena.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=4,
    max_turns=20,
)
register(
    "ta:TowerofHanoi-v0-hard",
    "gem.envs.textarena.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=5,
    max_turns=50,
)

# Register math dataset environments

register(
    "math:ASDiv2K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/ASDIV-2k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:GSM8K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/GSM-8k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:Math12K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/MATH-12k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:Orz57K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/ORZ-57k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:DeepScaleR40K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/DeepScaleR-40K",
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
    dataset_name="axon-rl/NaturalQuestions",
    split="train",
    question_key="problem",
    answer_key="answer",
)

register(
    "qa:HotpotQA",
    "gem.envs.qa_env:QaEnv",
    dataset_name="axon-rl/HotpotQA",
    split="train",
    question_key="problem",
    answer_key="answer",
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
register(
    "eval:MATH500",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/Eval-MATH500",
    question_key="problem",
    answer_key="answer",
)

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
