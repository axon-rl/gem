# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fifteen Puzzle environment"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from gem.core import Env
from gem.utils.constants import LanguageGameReward


class FifteenPuzzleEnv(Env):
    '''
    The Fifteen Puzzle is a sequential, deterministic, single-agent game
    where the agent is presented with a 4x4 grid containing 15 tiles (1-15)
    and one empty space, and must arrange them in row-major ascending order by 
    sliding tiles into the empty space.

    For generalization and convenience purposes, we parameterize the puzzle size
    (num_rows) and the maximum number of turns to solve the puzzle (max_turns).

    Special care is taken to make the board solvable by generating
    configurations which respect the parity constraints of the puzzle 
    (https://doi.org/10.2307/2369492).
    '''
    VALID_INITIALIZATION_ALGOS = ["monte_carlo", "pure_random"]

    def __init__(self, max_turns: Optional[int] = 20, num_rows: Optional[int] = 2, init_algo = "monte_carlo", mc_rand_moves = 100, **_):
        super().__init__()
        self.max_turns = max_turns
        self.num_rows = num_rows
        self.init_algo = init_algo
        assert init_algo in self.VALID_INITIALIZATION_ALGOS, (
            f"Initialization algorithm {init_algo} not recognized. "
            f"Valid options are: {self.VALID_INITIALIZATION_ALGOS}"
        )
        self.mc_rand_moves = mc_rand_moves
        assert mc_rand_moves > 0, "Number of random moves for Monte Carlo initialization must be positive."
        self._is_random = num_rows is None or max_turns is None
        self.greatest_num = self.num_rows**2 - 1
        self.reset()

    def _get_instructions(self) -> str:
        return (
            f"You are playing the {self.greatest_num}-Puzzle game.\n"
            f"You have to arrange the numbered tiles in ascending order from 1 to {self.greatest_num}, with the empty space located in the bottom-right corner.\n"
            "To make a move, you can slide a tile into the empty space (represented by a double underscore, e.g. __) by using one of the following commands:\n"
            "- 'up': Move the tile below the empty space up.\n"
            "- 'down': Move the tile above the empty space down.\n"
            "- 'left': Move the tile to the right of the empty space left.\n"
            "- 'right': Move the tile to the left of the empty space right.\n"
            "To submit your move, type the direction (e.g., 'up', 'down', 'left', or 'right') in \\boxed{...}.\n"
            "The current board layout is shown below\n"
            "Use logic and planning to solve the puzzle.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Here is the current board layout: \n{self._render_board()}\n"
            "Enter your move."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if self._is_random:
            candidates = [(2, 10), (3, 20), (4, 50)]
            self.num_rows, self.max_turns = random.choice(candidates)

        self.greatest_num = self.num_rows**2 - 1
        self.goal_board = self._generate_goal_board()
        self.goal_parity = self._get_board_parity(self.goal_board)
        self.board = self._generate_solvable_board()
        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        action_search_pattern = re.compile(
            r"\\boxed{(up|down|left|right)}"
        )  # e.g. \\boxed{up}
        matches = list(action_search_pattern.finditer(action))
        clean_action = matches[-1] if matches else None

        try:
            player_guess = clean_action.group(1).lower() if clean_action else None
        except Exception:
            player_guess = None

        if player_guess is None:
            terminate_obs = (
                f"At turn {self.turn_count}, you did not provide a valid guess"
            )
            return (
                terminate_obs,
                LanguageGameReward.format_error_reward,
                True,
                self.turn_count == self.max_turns,
                {"suffix": self.get_task_suffix()},
            )
        else:
            is_valid_move = self._move(player_guess)
            if not is_valid_move:  # invalid action
                next_obs = f"At turn {self.turn_count}, you chose a move {player_guess} that is outside the bounds of the board."
                reward = LanguageGameReward.invalid_action_reward
            else:
                if self._is_solved():
                    terminate_obs = "Congratulations! You have solved the puzzle!"
                    reward = LanguageGameReward.success_reward
                    return (
                        terminate_obs,
                        reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                else:
                    next_obs = f"At turn {self.turn_count}, you made a valid move: {player_guess}.\n"
                    reward = LanguageGameReward.internal_step_reward

        if self.turn_count >= self.max_turns:
            terminate_obs = "You have reached the maximum number of turns."
            reward += self._get_soft_reward()
            return terminate_obs, reward, True, True, {"suffix": self.get_task_suffix()}
        return next_obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _get_soft_reward(self) -> float:
        def _is_equal(a, b) -> bool:
            return a == b or (a is None and b is None)

        correct_tiles = list(range(1, self.greatest_num + 1)) + [None]
        current_tiles = [tile for row in self.board for tile in row]
        reward = 0
        for cor, cur in zip(correct_tiles, current_tiles):
            if _is_equal(cor, cur):
                reward += 1 / (self.greatest_num + 1)
        return reward
    
    def _has_correct_tiles(self, board: List[List[Optional[int]]]) -> bool:
        '''
        Check, as a necessary condition, that the board contains all tiles from 1 to greatest_num.
        '''
        correct_tiles = set(range(1, self.greatest_num + 1)) | {None}
        current_tiles = {tile for row in board for tile in row}
        return correct_tiles == current_tiles


    def _get_board_parity(self, board: List[List[Optional[int]]]) -> int:
        '''
        Computes the invariant of the current board,
        which is defined as the parity of the sum of two components: 
        - parity of tile permutations (incl. empty tile)
        - the Manhattan distance of the empty space from 
        the top-left corner.

        This completely determines whether the board is solvable,
        since all moves preserve this invariant.
        '''
        # Collect tiles in row-major order.
        tiles = [tile for row in board for tile in row]

        tps = 0
        # Count the number of transpositions required to sort the tiles.
        for i in range(len(tiles) - 1):
            # Search for tile i + 1, then swap it to position i
            pos_of_tile = tiles.index(i + 1)
            if pos_of_tile == i:
                continue
            tiles[i], tiles[pos_of_tile] = tiles[pos_of_tile], tiles[i]
            tps += 1
        tile_parity = tps % 2
        
        # Compute Manhattan distance of empty space from top-left corner
        tiles_empty_idx = [tile for row in board for tile in row].index(None)
        empty_row, empty_col = tiles_empty_idx // self.num_rows, tiles_empty_idx % self.num_rows
        mh_dist = empty_row + empty_col
        empty_parity = mh_dist % 2

        return (tile_parity + empty_parity) % 2

    def _is_solvable(self, board: List[List[Optional[int]]]) -> bool:
        '''
        Checks if the given board configuration is solvable,
        based on the parity invariant.
        '''
        if not self._has_correct_tiles(board):
            return False
        
        return self._get_board_parity(board) == self.goal_parity

    def _generate_solvable_board(self) -> List[List[Optional[int]]]:

        # Empirically speaking, pure random sampling produces extremely difficult boards.
        # An alternative is to perform a series of random valid moves
        # (which by definition are invariant-preserving).
        # Only constraint is we don't "waste" moves; i.e. make those that 
        # immediately reverse the previous move.

        if self.init_algo == "pure_random":
            tiles = list(range(1, self.greatest_num + 1)) + [None]
            random.shuffle(tiles)
            # If parity not good, swap 2 random non-empty tiles.
            # This guarantees equal distribution of solvable boards.
            cand_board = [
                tiles[i : i + self.num_rows]
                for i in range(0, self.greatest_num + 1, self.num_rows)
            ]
            if not self._is_solvable(cand_board):
                tile1, tile2 = random.sample(
                    [t for t in tiles if t is not None], 2
                )
                idx1, idx2 = tiles.index(tile1), tiles.index(tile2)
                r1, c1 = idx1 // self.num_rows, idx1 % self.num_rows
                r2, c2 = idx2 // self.num_rows, idx2 % self.num_rows
                cand_board[r1][c1], cand_board[r2][c2] = cand_board[r2][c2], cand_board[r1][c1]
            return cand_board
        elif self.init_algo == "monte_carlo":
            board = self._generate_goal_board()
            opposite_moves = {"up": "down", "down": "up", "left": "right", "right": "left"}
            last_move = None
            for _ in range(self.mc_rand_moves):
                curr_empty_row, curr_empty_col = -1, -1
                for i in range(self.num_rows):
                    for j in range(self.num_rows):
                        if board[i][j] is None:
                            curr_empty_row, curr_empty_col = i, j
                            break
                possible_moves = [
                    move 
                    for move in ["up", "down", "left", "right"]
                    if (
                        move != (opposite_moves[last_move] if last_move else None)
                        and (
                            (move == "up" and curr_empty_row < self.num_rows - 1)
                            or (move == "down" and curr_empty_row > 0)
                            or (move == "left" and curr_empty_col < self.num_rows - 1)
                            or (move == "right" and curr_empty_col > 0)
                        )
                    )
                ]
                move = random.choice(possible_moves)
                tgt_row, tgt_col = curr_empty_row, curr_empty_col
                if move == "up":
                    tgt_row += 1
                elif move == "down": 
                    tgt_row -= 1
                elif move == "left":
                    tgt_col += 1
                elif move == "right":
                    tgt_col -= 1
                # Swap tiles
                board[curr_empty_row][curr_empty_col], board[tgt_row][tgt_col] = (
                    board[tgt_row][tgt_col],
                    board[curr_empty_row][curr_empty_col],
                )
                last_move = move

            return board

    
    def _generate_goal_board(self) -> List[List[Optional[int]]]:
        '''
        Defines the goal board configuration for the puzzle.

        This is used for:
        - Checking if the puzzle is solved.
        - Computing soft rewards based on tile positions.
        - Computing board parity for solvability checks.
        '''
        tiles = list(range(1, self.greatest_num + 1)) + [None]
        return [
            tiles[i : i + self.num_rows]
            for i in range(0, self.greatest_num + 1, self.num_rows)
        ]

    def _render_board(self) -> str:
        rendered_board = ""
        for row in self.board:
            rendered_board += (
                " ".join(["__" if x is None else f"{x:2}" for x in row]) + "\n"
            )
        return rendered_board

    def _move(self, direction: str) -> bool:
        # Moves are expressed in an 'intuitive' manner, in the sense
        # that tiles are moved 'up'/'down'/'left'/'right' INTO the empty space.
        # Correspondingly, the empty space moves in the opposite direction.

        empty_row, empty_col = self._get_empty_position()

        target_row, target_col = empty_row, empty_col

        if direction == "up" and empty_row < self.num_rows - 1:
            target_row += 1
        elif direction == "down" and empty_row > 0:
            target_row -= 1
        elif direction == "left" and empty_col < self.num_rows - 1:
            target_col += 1
        elif direction == "right" and empty_col > 0:
            target_col -= 1
        else:  # invalid move
            return False

        self.board[empty_row][empty_col], self.board[target_row][target_col] = (
            self.board[target_row][target_col],
            self.board[empty_row][empty_col],
        )
        return True

    def _get_empty_position(self):
        for r in range(self.num_rows):
            for c in range(self.num_rows):
                if self.board[r][c] is None:
                    return r, c

    def _is_solved(self) -> bool:
        # Perform deep comparison of current board with goal board.
        return all(
            self.board[r][c] == self.goal_board[r][c]
            for r in range(self.num_rows)
            for c in range(self.num_rows)
        )

    def sample_random_action(self) -> str:
        return random.choice(["up", "down", "left", "right"])
