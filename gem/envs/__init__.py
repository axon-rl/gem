# import reasoning_gym as rg

from gem.envs.registration import register

# Register games from our implementation of TextArena
register(
    "ta:GuessTheNumber-v0",
    "gem.envs.textarena.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=20,
    max_turns=20,
)

# # Register datasets from ReasoningGym
# for name in rg.factory.DATASETS.keys():
#     register(
#         f"rg:{name}",
#         "gem.envs.reasoning_gym:ReasoningGymEnv",
#         name=name,
#         size=500,
#         seed=42,
#     )
