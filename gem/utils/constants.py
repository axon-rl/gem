TERMINAL_STATE = "<｜TERMINAL_STATE｜>"

# the reward for successfully completing the task
SUCCESS_REWARD = 1.0
# the reward for successful milestone, e.g. reveal a empty space in Minesweeper
SUCCESS_INTERNAL_REWARD = 0.1
# the reward for a step that does not end the game, e.g. a hint in Guess the Number
INTERNAL_STEP_REWARD = 0.0
# the reward for failing the task, e.g. hit a mine in Minesweeper
FAIL_REWARD = 0.0  # -0.02
# the reward for an invalid action, e.g. guessing a number outside the range in Guess the Number
INVALID_ACTION_REWARD = -0.05
# the reward for a format error, e.g. cannot parse the action
FORMAT_ERROR_REWARD = -0.1
