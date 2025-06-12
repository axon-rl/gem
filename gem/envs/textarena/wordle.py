"""Wordle game environment"""

import random
import re
from typing import Any, List, Optional, Tuple

import nltk
from nltk.corpus import words

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class WordleEnv(Env):

    def __init__(
        self,
        word_length: Optional[int] = 5,
        hardcore: Optional[bool] = False,
        only_real_words: Optional[bool] = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        nltk.download("words")
        self.word_length = word_length
        self.hardcore = hardcore
        self.only_real_words = only_real_words
        self.max_turns = max_turns
        self.is_random = word_length is None
        self.all_words = words.words("en") if hardcore else words.words("en-basic")
        self.reset()

    def _get_instructions(self) -> str:
        return (
            f"You are playing Wordle.\n"
            f"You have to guess the secret {self.word_length}-letter word within {self.max_turns} turns.\n"
            "After you enter your guess, I will say mark your guess as follows:\n"
            "  - G (green): correct letter in the correct position\n"
            "  - Y (yellow): letter exists in the word but in the wrong position\n"
            "  - X (wrong): letter is not in the word\n"
            "After thinking, format your final answer inside \\boxed{...},"
            f" for example, {self.sample_random_action()}.\n"
            "As you play, the history of your guesses will be appended below. Use the information to complete the game before you run out of guesses.\n"
            "Enter your first guess to start the game.\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        if self.is_random:
            self.word_length = random.randint(3, 6)
        available_words = [
            word
            for word in self.all_words
            if len(word) == self.word_length and word.isalpha() and word.islower()
        ]
        self.secret_word = random.choice(available_words).upper()
        self.previous_guesses = set()
        self.turn_count = 0
        return self._get_instructions(), {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        self.turn_count += 1

        action_search_pattern = re.compile(r"\\boxed{([a-zA-Z]+(?:\s+[a-zA-Z]+)*)}")
        matches = list(action_search_pattern.finditer(action))
        clean_action = matches[-1] if matches else None
        try:
            if clean_action is not None:
                player_guess = clean_action.group(1).upper().replace(" ", "")
            else:
                player_guess = None
        except Exception:
            player_guess = None

        if not player_guess:
            return TERMINAL_STATE, -0.1, True, self.turn_count == self.max_turns, {}
        else:
            length_correct, feedback = self._evaluate_guess(player_guess)
            if self.turn_count >= self.max_turns:
                reward = (
                    -0.1
                    if not length_correct
                    else (feedback.count("G") + 0.5 * feedback.count("Y"))
                    / self.word_length
                )
                return TERMINAL_STATE, reward, True, True, {}

            if len(player_guess) != self.word_length:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess} which has {len(player_guess)} letters but the secret word has {self.word_length} letters."
                reward, terminated, truncated = -0.05, False, False
            elif player_guess.lower() not in self.all_words and self.only_real_words:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess}. This word is not a valid English word. All guesses must be valid English words."
                reward, terminated, truncated = -0.05, False, False
            elif player_guess in self.previous_guesses:
                next_obs = f"At turn {self.turn_count}, you guessed {player_guess}, which has been already guessed before."
                reward, terminated, truncated = -0.05, False, False
            else:
                self.previous_guesses.add(player_guess)
                if feedback.count("G") == self.word_length:
                    next_obs = f"Congratulations! You guessed the secret word {self.secret_word} in {self.turn_count} turns."
                    reward, terminated, truncated = 1, True, False
                else:
                    next_obs = f"At turn {self.turn_count}, you guessed {player_guess}.\nFeedback:\n"
                    next_obs += " ".join(player_guess) + "\n"
                    next_obs += " ".join(feedback)
                    reward, terminated, truncated = 0, False, False
        
        if not terminated:
            next_obs += "\nEnter your next guess."
        return next_obs, reward, terminated, truncated, {}

    def sample_random_action(self):
        """Samples a random word"""
        return f"\\boxed{{{random.choice(self.all_words).upper()}}}"

    def _evaluate_guess(self, player_guess: str) -> List[str]:
        """Evaluates the player's guess against the secret word"""
        length_correct = len(player_guess) == self.word_length
        if not length_correct:
            return length_correct, None
        feedback = [None] * self.word_length
        secret_list = list(self.secret_word.upper())
        guess_list = list(player_guess.upper())

        # First pass: mark correct letters in the correct position (green)
        for i in range(self.word_length):
            if guess_list[i] == secret_list[i]:
                feedback[i] = "G"
                secret_list[i] = None  # Mark this letter as accounted for

        # Second pass: mark correct letters in the wrong position (yellow) or wrong letters
        for i in range(self.word_length):
            if feedback[i] is None:
                if guess_list[i] in secret_list:
                    feedback[i] = "Y"
                    # Remove the first occurrence of guess_list[i] from secret_list
                    index = secret_list.index(guess_list[i])
                    secret_list[index] = None
                else:
                    feedback[i] = "X"

        return length_correct, feedback
