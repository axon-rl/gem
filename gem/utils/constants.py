TERMINAL_STATE = "<｜TERMINAL_STATE｜>"


class TextArenaGameReward:
    # the reward for successfully completing the task
    success_reward: float = 1.0
    # the reward for successful milestone, e.g. reveal a empty space in Minesweeper
    success_internal_reward: float = 0.1
    # the reward for a step that does not end the game, e.g. a hint in Guess the Number
    internal_step_reward: float = 0.0
    # the reward for failing the task, e.g. hit a mine in Minesweeper
    fail_reward: float = 0.0  # -0.02
    # the reward for an invalid action, e.g. guessing a number outside the range in Guess the Number
    invalid_action_reward: float = -0.05
    # the reward for a format error, e.g. cannot parse the action
    format_error_reward: float = -0.1
