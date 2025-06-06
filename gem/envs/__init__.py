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
    "ta:GuessTheNumber-v0-sanitycheck",
    "gem.envs.textarena.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=5,
    max_turns=2,
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
    "ta:Mastermind-v0-extreme",
    "gem.envs.textarena.mastermind:MastermindEnv",
    code_length=6,
    num_numbers=12,
    max_turns=50,
    duplicate_numbers=True,
)
register(
    "ta:Mastermind-v0-random",
    "gem.envs.textarena.mastermind:MastermindEnv",
    code_length=None,
    num_numbers=None,
    max_turns=20,
    duplicate_numbers=False,
)
register(
    "ta:Mastermind-v0-easy",
    "gem.envs.textarena.mastermind:MastermindEnv",
    code_length=2,
    num_numbers=6,
    max_turns=20,
    duplicate_numbers=False,
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
