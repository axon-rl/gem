"""Guess the number game environment."""

import random
from typing import Any, Optional, Tuple

from gem.envs.multi_turn import MultiTurnEnv
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer


class GuessTheNumberEnv(MultiTurnEnv):

    def __init__(self, min_number: int = 1, max_number: int = 20, max_turns: int = 20):
        super().__init__()
        self.min_number = min_number
        self.max_number = max_number
        self.max_turns = max_turns

    def get_task_prefix(self) -> str:
        return (
            f"You are playing Guess The Number.\n"
            f"You have to guess the number between {self.min_number} and {self.max_number} (inclusive) within {self.max_turns} turns.\n"
            "As you enter your guess, the game will provide you with hints such as the target number is 'higher' or 'lower'.\n"
            "You may provide your response in any manner. Only the number that is wrapped inside \\boxed{} will be considered as your guess. For example, \\boxed{5}.\n"
            "As you play, the history of your guesses will be appended below. Use the information to complete the game before you run out of guesses.\n"
        )

    def get_task_suffix(self) -> str:
        return "Enter your guess."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        self.game_number = random.randint(self.min_number, self.max_number)
        self.guessed_numbers = set()
        self.turn_count = 0
        return self.get_task_prefix() + self.get_task_suffix(), {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        self.turn_count += 1

        clean_action = extract_last_boxed_answer(action)
        try:
            player_guess = int(clean_action) if clean_action else None
        except Exception:
            player_guess = None

        if not player_guess:
            return TERMINAL_STATE, -1, True, self.turn_count == self.max_turns, {}
        else:
            if self.turn_count >= self.max_turns:
                distance = abs(player_guess - self.game_number)
                reward = 1 - (distance / (self.max_number - self.min_number))
                return TERMINAL_STATE, reward, True, True, {}

            if player_guess < self.min_number or player_guess > self.max_number:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, which is outside the range specified."
                reward, terminated, truncated = -0.1, False, False
            elif player_guess in self.guessed_numbers:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, which has been already guessed before."
                reward, terminated, truncated = -0.1, False, False
            else:
                self.guessed_numbers.add(player_guess)
                if player_guess == self.game_number:
                    next_obs = f"Congratulations! You guessed the correct number {self.game_number} in {self.turn_count} turns."
                    reward, terminated, truncated = 1, True, False
                else:
                    hint = "lower" if player_guess > self.game_number else "higher"
                    next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, and the target number is {hint} than {player_guess}."
                    reward, terminated, truncated = 0.1, False, False
        return next_obs, reward, terminated, truncated, {}

    def sample_random_action(self):
        return "\\boxed{" + str(random.randint(self.min_number, self.max_number)) + "}"
