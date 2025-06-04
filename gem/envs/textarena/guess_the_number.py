"""Guess the number game environment"""

import random
import re
from typing import Any, Optional, Tuple

from gem.envs.textarena.base import TextArenaEnv
from gem.utils.constants import TERMINAL_STATE


class GuessTheNumberEnv(TextArenaEnv):

    def __init__(self, min_number: int = 1, max_number: int = 20, max_turns: int = 20):
        super().__init__()
        self.min_number = min_number
        self.max_number = max_number
        self.max_turns = max_turns

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        self.game_number = random.randint(self.min_number, self.max_number)
        self.guessed_numbers = set()
        self.game_history = []
        self.turn_count = 0
        return self._get_obs(), {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        self.turn_count += 1
        action_search_pattern = re.compile(r"\[(\d+)\]")  # e.g. [5]
        match = action_search_pattern.search(action)

        if not match:
            return TERMINAL_STATE, -1, True, self.turn_count == self.max_turns, {}
        else:
            player_guess = int(match.group(1))

            if self.turn_count >= self.max_turns:
                distance = abs(player_guess - self.game_number)
                reward = 1 - (distance / (self.max_number - self.min_number))
                return TERMINAL_STATE, reward, True, True, {}

            if player_guess < self.min_number or player_guess > self.max_number:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, which is outside the range specified."
                self.game_history.append(next_obs)
                return self._get_obs(), -0.1, False, False, {}
            elif player_guess in self.guessed_numbers:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, which has been already guessed before."
                self.game_history.append(next_obs)
                return self._get_obs(), -0.1, False, False, {}
            else:
                self.guessed_numbers.add(player_guess)
                if player_guess == self.game_number:
                    next_obs = f"Congratulations! You guessed the correct number {self.game_number} in {self.turn_count} turns."
                    self.game_history.append(next_obs)
                    return self._get_obs(), 1, True, False, {}
                else:
                    hint = "lower" if player_guess > self.game_number else "higher"
                    next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, and the target number is {hint} than {player_guess}."
                    self.game_history.append(next_obs)
                    return self._get_obs(), 0.1, False, False, {}

    def sample_random_action(self):
        return "[" + str(random.randint(self.min_number, self.max_number)) + "]"

    def _get_obs(self) -> str:
        game_prefix = (
            f"You are playing Guess The Number.\n"
            f"You have to guess the number between {self.min_number} and {self.max_number} (inclusive) within {self.max_turns} turns.\n"
            "As you enter your guess, the game will provide you with hints such as the target number is 'higher' or 'lower'.\n"
            "You may provide your response in any manner. Only the number that is wrapped in square brackets will be considered as your guess. For example, [5].\n"
            "As you play, the history of your guesses will be appended below. Use the information to complete the game before you run out of guesses.\n"
        )
        if self.game_history:
            for obs in self.game_history:
                game_prefix += f"{obs}\n"
        prompt = game_prefix + "Enter your guess."
        return prompt
