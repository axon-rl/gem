from gem.utils.constants import LanguageGameReward

class WikiGameReward(LanguageGameReward):
    '''
    Rewards for Wikipedia Game environment, which
    assume infinite-horizon RL task formulation.

    I can't guarantee they are optimal but this is the reasoning:
    - success_reward: large positive reward to encourage solving the task
    - internal_step_reward: encourages LMs to solve the task with less meandering.
    - format_error_reward and invalid_action_reward: 
        encourage model to act meaningfully and in a communicable manner.
    - fail_reward: 
        neutral since failing is already differentiated from being able to reach the goal.
    '''

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
