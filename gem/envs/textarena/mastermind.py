"""Mastermind game environment"""

import random
from typing import Any, Optional, Tuple, List
import re

from gem.envs.multi_turn import MultiTurnEnv
from gem.utils.constants import TERMINAL_STATE


class MastermindEnv(MultiTurnEnv):

    def __init__(
        self,
        code_length: Optional[int] = 4,
        num_numbers: Optional[int] = 6,
        duplicate_numbers: Optional[bool] = False,
        max_turns: Optional[int] = 20,
    ):
        super().__init__()
        self.code_length = code_length
        self.num_numbers = num_numbers
        self.duplicate_numbers = duplicate_numbers
        self.max_turns = max_turns
        self.is_random = code_length is None or num_numbers is None

    def get_task_prefix(self) -> str:
        return (
            f"You are playing Mastermind.\n"
            f"You have to guess the secret {self.code_length} digit code within {self.max_turns} turns.\n"
            f"The code consists of digits from 1 to {self.num_numbers} (inclusive). "
            f"Duplicate numbers are {'allowed' if self.duplicate_numbers else 'not allowed'}.\n"
            "After you enter your guess, I will say mark your guess with black and white pegs, where a black peg indicates a correct digit in the correct position, while a white peg indicates a correct digit in the wrong position.\n"
            "After thinking, format your final answer inside \\boxed{...},"
            f" for example, {self.example_action}.\n"
            "As you play, the history of your guesses will be appended below. Use the information to complete the game before you run out of guesses.\n"
        )

    def get_task_suffix(self) -> str:
        return "Enter your guess."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        if self.is_random:
            self.code_length = random.randint(2, 4)
            self.num_numbers = random.randint(self.code_length, 6)
        available_numbers = list(range(1, self.num_numbers + 1))
        sample_fn = random.choices if self.duplicate_numbers else random.sample
        self.game_code = sample_fn(available_numbers, k=self.code_length)
        self.guessed_numbers = set()
        self.turn_count = 0
        self.example_action = self.sample_random_action()
        return self.get_task_prefix() + self.get_task_suffix(), {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        self.turn_count += 1

        action_search_pattern = re.compile(r"\\boxed{(\d+(?:\s+\d+)*)}")
        matches = list(action_search_pattern.finditer(action))
        clean_action = matches[-1] if matches else None
        try:
            player_guess = list(map(int, clean_action.group(1).split())) if clean_action is not None else None
        except Exception:
            player_guess = None

        if not player_guess:
            return TERMINAL_STATE, -1, True, self.turn_count == self.max_turns, {}
        else:
            length_correct, black_pegs, white_pegs = self._evaluate_guess(player_guess)
            if self.turn_count >= self.max_turns:
                reward = -1 if not length_correct else (black_pegs + 0.5 * white_pegs) / self.code_length
                return TERMINAL_STATE, reward, True, True, {}

            if len(player_guess) != self.code_length:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess} which has {len(player_guess)} entires but the code has length {self.code_length}."
                reward, terminated, truncated = -0.1, False, False
            elif any(num < 1 or num > self.num_numbers for num in player_guess):
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, which has numbers outside the range 1 to {self.num_numbers}."
                reward, terminated, truncated = -0.1, False, False
            elif tuple(player_guess) in self.guessed_numbers:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, which has been already guessed before."
                reward, terminated, truncated = -0.1, False, False
            else:
                self.guessed_numbers.add(tuple(player_guess))
                if black_pegs == self.code_length:
                    next_obs = f"Congratulations! You guessed the correct code {self.game_code} in {self.turn_count} turns."
                    reward, terminated, truncated = 1, True, False
                else:
                    next_obs = f"At turn {self.turn_count}, you guessed {player_guess}. This guess recieves {black_pegs} black peg(s) and {white_pegs} white peg(s)."
                    reward, terminated, truncated = 0.1, False, False
        return next_obs, reward, terminated, truncated, {}

    def sample_random_action(self):
        return "\\boxed{" + " ".join(map(str, random.sample(range(1, self.num_numbers + 1), self.code_length))) + "}"

    def _evaluate_guess(
        self, player_guess: List[int]
    ) -> Tuple[int, int]:
        """
        Evaluates the player's guess and returns the number of black and white pegs.
        Black peg: correct digit in the correct position.
        White peg: correct digit in the wrong position.

        Args: player_guess (List[int]): The player's guess.
        Returns: Tuple[int, int]: Number of black and white pegs.
        """
        length_correct = len(player_guess) == self.code_length
        black_pegs = 0
        white_pegs = 0

        # Create copies to mark matched positions
        code_copy = self.game_code.copy()
        guess_copy = player_guess.copy()

        # First pass: count black pegs and mark them as None
        for i in range(self.code_length):
            if guess_copy[i] == code_copy[i]:
                black_pegs += 1
                code_copy[i] = None
                guess_copy[i] = None

        # Second pass: count white pegs using the remaining numbers
        for i in range(self.code_length):
            if guess_copy[i] is not None and guess_copy[i] in code_copy:
                white_pegs += 1
                # Remove the first occurrence to prevent over-counting
                code_copy[code_copy.index(guess_copy[i])] = None

        return length_correct, black_pegs, white_pegs