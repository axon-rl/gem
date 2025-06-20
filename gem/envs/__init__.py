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


# Register math dataset environments

register(
    "eval:MATH500",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/Eval-MATH500",
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

# Register code dataset environments

## The test split of deepmind/code_contests, with merged test cases.
register(
    "eval:CodeContest",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/CodeContest",
    split="test",
    question_key="problem",
    test_key="tests",
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
