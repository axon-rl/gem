from gem.utils.constants import LanguageGameReward
# Proposal 1: Punish the model for making illegal moves
# class WikiGameReward(LanguageGameReward):
#     # the reward for successfully completing the task
#     success_reward: float = 1.0
#     # the reward for a step that does not end the game, e.g. a hint in Guess the Number
#     internal_step_reward: float = 0.0
#     # the reward for failing the task, e.g. hit a mine in Minesweeper
#     fail_reward: float = 0.0
#     # the reward for an invalid action, e.g. guessing a number outside the range in Guess the Number
#     invalid_action_reward: float = -0.2
#     # the reward for a format error, e.g. cannot parse the action
#     format_error_reward: float = -0.2
# 
# Proposal 2: Do not.
# class WikiGameReward(LanguageGameReward):
#     # the reward for successfully completing the task
#     success_reward: float = 1.0
#     # the reward for a step that does not end the game, e.g. a hint in Guess the Number
#     internal_step_reward: float = 0.0
#     # the reward for failing the task, e.g. hit a mine in Minesweeper
#     fail_reward: float = 0.0
#     # the reward for an invalid action, e.g. guessing a number outside the range in Guess the Number
#     invalid_action_reward: float = 0.0
#     # the reward for a format error, e.g. cannot parse the action
#     format_error_reward: float = -0.2

# Proposal 3: While the model hasn't won, cook it.
# class WikiGameReward(LanguageGameReward):  
#     # the reward for successfully completing the task
#     success_reward: float = 1.0
#     # the reward for a step that does not end the game, e.g. a hint in Guess the Number
#     internal_step_reward: float = -0.05
#     # the reward for failing the task, e.g. hit a mine in Minesweeper
#     fail_reward: float = 0.0
#     # the reward for an invalid action, e.g. guessing a number outside the range in Guess the Number
#     invalid_action_reward: float = 0.0
#     # the reward for a format error, e.g. cannot parse the action
#     format_error_reward: float = -0.2

# Proposal 4: ALL OF THE ABOVE.
class WikiGameReward(LanguageGameReward):  
    # the reward for successfully completing the task
    success_reward: float = 1.0
    # the reward for a step that does not end the game, e.g. a hint in Guess the Number
    internal_step_reward: float = -0.05
    # the reward for failing the task, e.g. hit a mine in Minesweeper
    fail_reward: float = 0.0
    # the reward for an invalid action, e.g. guessing a number outside the range in Guess the Number
    invalid_action_reward: float = -0.2
    # the reward for a format error, e.g. cannot parse the action
    format_error_reward: float = -0.2
