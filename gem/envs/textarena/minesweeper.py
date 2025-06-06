"""Mine Sweeper game environment."""

import random
from typing import Any, Optional, Tuple
import re
import numpy as np
from collections import deque

from gem.envs.multi_turn import MultiTurnEnv
from gem.utils.constants import TERMINAL_STATE

class MinesweeperEnv(MultiTurnEnv):

    def __init__(self, rows: int = 8, cols: int = 8, num_mines: int = 10, max_turns: int = 20):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.max_turns = max_turns
        self.reset()

    def get_task_prefix(self) -> str:
        reveal_r = int(self.example_reveal.split(" ")[1])
        reveal_c = int(self.example_reveal.split(" ")[2].split("}")[0])
        flag_r = int(self.example_flag.split(" ")[1])
        flag_c = int(self.example_flag.split(" ")[2].split("}")[0])
        return (
            f"You are playing the Minesweeper game.\n"
            "The objective of the game is to reveal all cells that do not contain mines.\n"
            "To make a move, you can either reveal a cell or place a flag on a suspected mine location using one of the following commands:\n"
            "- 'reveal': Reveal the contents of a specific cell.\n"
            "- 'flag': Place or remove a flag on a specific cell to mark it as a potential mine.\n"
            "To submit your move, type the command followed by the row and column in \\boxed{}.\n"
            "For example:\n"
            f"- {self.example_reveal} to reveal the cell in Row {reveal_r}, Column {reveal_c}.\n"
            f"- {self.example_flag} to place or remove a flag on the cell in Row {flag_r}, Column {flag_c}.\n"
            "The current board layout is shown below. Cells that are unrevealed are represented by a dot ('.'), revealed numbers show the count of adjacent mines, and flagged cells are marked with an 'F'.\n"
            "Use logic and deduction to avoid revealing cells with mines!\n"
            "Be mindful not to choose revealed or flagged cells.\n"
            "Here is the current board layout:\n"
            f"{self._render_board(is_start=True)}\n"
        )

    def get_task_suffix(self) -> str:
        return "Enter your guess."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        super().reset(seed)
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.revealed = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.flags = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.first_reveal = True  # Track if it's the first move to ensure playability
        self.turn_count = 0
        self.example_reveal = self.sample_random_action(reveal_or_flag="reveal")
        self.example_flag = self.sample_random_action(reveal_or_flag="flag")
        return self.get_task_prefix() + self.get_task_suffix(), {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        self.turn_count += 1

        action_search_pattern = re.compile(r"\\boxed{([a-zA-Z]+)\s(\d+)\s(\d+)}") # e.g. \\boxed{reveal 3 2}
        matches = list(action_search_pattern.finditer(action))
        clean_action = matches[-1] if matches else None
        try:
            action_type = clean_action.group(1).lower() if clean_action else None
            row = int(clean_action.group(2)) if clean_action else None
            col = int(clean_action.group(3)) if clean_action else None
        except Exception:
            action_type, row, col = None, None, None

        if action_type is None or row is None or col is None:
            return TERMINAL_STATE, -1, True, self.turn_count == self.max_turns, {}
        else:
            if self.turn_count >= self.max_turns:
                num_revealed = np.sum(self.revealed)
                reward = num_revealed / (self.rows * self.cols - self.num_mines)
                return TERMINAL_STATE, reward, True, True, {}
            
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                next_obs = f"At turn {self.turn_count}, you chose cell ({row}, {col}), which is outside the bounds of the grid."
                reward, terminated, truncated = -0.1, False, False
            elif action_type == "reveal":
                if self.first_reveal:
                    self._setup_mines(row, col)
                    self.first_reveal = False

                if self.grid[row][col] == -1:
                    ## If the cell is a mine, end the game
                    next_obs = f"Game over! You hit a mine at ({row}, {col})."
                    reward, terminated, truncated = -1, True, False
                elif self.revealed[row][col] or self.flags[row][col]:
                    ## If already revealed or flagged
                    next_obs = f"At turn {self.turn_count}, you chose to reveal cell ({row}, {col}), which has already been revealed or flagged."
                    reward, terminated, truncated = -0.1, False, False
                else:
                    self._update_grid(row, col)  # Update the grid and reveal cells
                    if self._is_solved():
                        ## If the game is solved
                        next_obs = f"Congratulations! You have successfully cleared the Minesweeper board."
                        reward, terminated, truncated = 1, True, False
                    else:
                        next_obs = (
                            f"At turn {self.turn_count}, successfully revealed cell ({row}, {col}).\n"
                            f"Here is the updated board:\n{self._render_board()}\n"
                        )
                        reward, terminated, truncated = 0.1, False, False

            elif action_type == "flag":
                if self.revealed[row][col]:
                    next_obs = f"At turn {self.turn_count}, you chose to flag cell ({row}, {col}), which has already been revealed."
                    reward, terminated, truncated = -0.1, False, False
                else:
                    self.flags[row][col] = not self.flags[row][col] # Toggle flag
                    next_obs = (
                        f"At turn {self.turn_count}, you "
                        f"{'added' if self.flags[row][col] else 'removed'} "
                        f"a flag on cell ({row}, {col}).\n"
                        f"Here is the updated board:\n{self._render_board()}\n"
                    )
                    reward, terminated, truncated = 0.1, False, False

            else:
                ## If the action is not recognized
                next_obs = f"At turn {self.turn_count}, you chose an invalid action '{action_type}'. Valid actions are 'reveal' or 'flag'."
                reward, terminated, truncated = -0.1, False, False

            return next_obs, reward, terminated, truncated, {}

    def sample_random_action(self, reveal_or_flag=None) -> str:
        if reveal_or_flag is None:
            reveal_or_flag = random.choice(["reveal"] * 9 + ["flag"])
        rand_row = random.randint(0, self.rows - 1)
        rand_col = random.randint(0, self.cols - 1)
        return f"\\boxed{{{reveal_or_flag} {rand_row} {rand_col}}}"

    def _setup_mines(self, safe_row: int, safe_col: int):
        mines = set()
        while len(mines) < self.num_mines:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            # Avoid placing mines in the safe zone
            if (r, c) not in mines and (
                r < safe_row - 1
                or r > safe_row + 1
                or c < safe_col - 1
                or c > safe_col + 1
            ):
                mines.add((r, c))
                self.grid[r][c] = -1  # -1 represents a mine
        self._calculate_adjacent_numbers()

    def _calculate_adjacent_numbers(self):
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == -1:
                    continue
                mine_count = sum(
                    (
                        0 <= r + dr < self.rows
                        and 0 <= c + dc < self.cols
                        and self.grid[r + dr][c + dc] == -1
                    )
                    for dr, dc in directions
                )
                self.grid[r][c] = mine_count

    def _update_grid(self, row: int, col: int):
        queue = deque([(row, col)]) # Start with the initial cell in the queue
        self.revealed[row][col] = True # Mark the initial cell as revealed immediately

        while queue:
            current_row, current_col = queue.popleft()

            # Check it's not a mine
            assert self.grid[current_row][current_col] != -1, (
                f"Env error: Hit mine at ({current_row}, {current_col}) - this should not happen.")

            # If the cell has no adjacent mines, add its neighbors to the queue
            if self.grid[current_row][current_col] == 0:
                for dr, dc in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]:
                    neighbor_row, neighbor_col = (
                        current_row + dr,
                        current_col + dc,
                    )
                    # Only add to the queue if within bounds and not revealed or flagged
                    if (
                        0 <= neighbor_row < self.rows
                        and 0 <= neighbor_col < self.cols
                    ):
                        if (
                            not self.revealed[neighbor_row][neighbor_col]
                            and not self.flags[neighbor_row][neighbor_col]
                        ):
                            self.revealed[neighbor_row][neighbor_col] = (
                                True  # Mark as revealed when adding to queue
                            )
                            queue.append((neighbor_row, neighbor_col))

    def _is_solved(self) -> bool:
        """
        Check if the board is in a solved state.

        Returns:
            bool: True if the board is in a solved state, False otherwise.
        """
        return all(
            (self.grid[r][c] == -1 and self.flags[r][c])
            or (self.grid[r][c] != -1 and self.revealed[r][c])
            for r in range(self.rows)
            for c in range(self.cols)
        )

    def _render_board(self, is_start: bool = False) -> str:
        """
        Render the game board.

        Returns:
            str: The rendered game board.
        """
        board_str = "   " + " ".join([str(c).rjust(2) for c in range(self.cols)]) + "\n"
        for r in range(self.rows):
            row_str = f"{r:2} "
            for c in range(self.cols):
                if is_start:
                    row_str += " . "
                else:
                    if self.revealed[r][c]:
                        if self.grid[r][c] == -1:
                            row_str += " * "
                        else:
                            row_str += f" {self.grid[r][c]} "
                    elif self.flags[r][c]:
                        row_str += " F "
                    else:
                        row_str += " . "
            board_str += row_str + "\n"
        return board_str

    

