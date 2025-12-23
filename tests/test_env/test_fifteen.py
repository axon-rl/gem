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

# Few things to watch out for:
# - Ensure transitions are logically sound.
# - Ensure solvability checks are correct.
# - Ensure generated boards are valid AND solvable (via A*).

import fire

from copy import deepcopy
from heapq import heappop, heappush
from itertools import product
from typing import Dict, List, Optional, Callable, Tuple
from gem.envs.game_env.fifteen_puzzle import FifteenPuzzleEnv

def classic_puzzle():
    # Returns the classic 4x4 fifteen puzzle environment.
    # You can edit this function to change parameters.
    return FifteenPuzzleEnv(max_turns = 100, num_rows = 4, init_algo = "monte_carlo", mc_rand_moves = 20)

def fix_state(env: FifteenPuzzleEnv, board: List[List[Optional[int]]], steps: int = 0):
    env.reset()
    # Fixes the environment to the given board state.
    env.board = deepcopy(board)
    env.blank_pos = env._get_empty_position()
    env.turn_count = steps

def transitions_work_properly():
    # 1. Goal board looks exactly as expected.
    # 2. Each move should only move the blank tile one position.
    # 3. Tiles should change positions appropriately.
    # 4. The number of blank tiles should remain one.
    env = classic_puzzle()

    def reset_goal_state():
        fix_state(env, env._generate_goal_board())
        
    reset_goal_state()

    GOAL_REFERENCE = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, None],
    ]
    assert all(
        env.board[r][c] == GOAL_REFERENCE[r][c]
        for r in range(env.num_rows)
        for c in range(env.num_rows)
    ), (
        f"Initial board does not match goal reference. Expected\n"
        f"{''.join(str(row) + '\n' for row in GOAL_REFERENCE)}, got\n"
        f"{''.join(str(row) + '\n' for row in env.board)}"
    )
    assert env._get_empty_position() == (3, 3), (
        f"Expected blank tile at position (3, 3), got {env._get_empty_position()}"
    )

    # Test 1: Starting from the goal state.
    # Test 1a: Move 'down'
    reset_goal_state()
    env.step(r'\\boxed{down}')

    expected = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, None],
        [13, 14, 15, 12],
    ]

    assert all(
        env.board[r][c] == expected[r][c]
        for r in range(env.num_rows)
        for c in range(env.num_rows)
    ), (
        f"After moving 'down' from goal state, expected\n"
        f"{''.join(str(row) + '\n' for row in expected)}, got\n"
        f"{''.join(str(row) + '\n' for row in env.board)}"
    )

    # Test 1b: Move 'right'
    reset_goal_state()
    env.step(r'\\boxed{right}')
    expected = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, None, 15],
    ]

    assert all(
        env.board[r][c] == expected[r][c]
        for r in range(env.num_rows)
        for c in range(env.num_rows)
    ), (
        f"After moving 'right' from goal state, expected\n"
        f"{''.join(str(row) + '\n' for row in expected)}, got\n"
        f"{''.join(str(row) + '\n' for row in env.board)}"
    )

    # Test 1c: Make sure 'up' is reported as invalid.
    reset_goal_state()
    valid = env._move(r'up')
    assert not valid, "Expected 'up' move from goal state to be invalid."

    # Test 1d: Make sure 'left' is reported as invalid.
    reset_goal_state()
    valid = env._move(r'left')
    assert not valid, "Expected 'left' move from goal state to be invalid."

    # Test 2: Starting from a non-goal state.
    NG_BOARD = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, None, 11, 12],
            [13, 10, 14, 15],
    ]
    NG_BLANK_POS = (2, 1)
    def reset_non_goal_state():
        env.reset()
        env.board = deepcopy(NG_BOARD)
        env.blank_pos = NG_BLANK_POS
    
    # Test 2a: All moves should be valid.
    # Test 2b: The blank tile should move correctly.
    for move, blank_tile_delta in [
        (r'up', (1, 0)),
        (r'down', (-1, 0)),
        (r'left', (0, 1)),
        (r'right', (0, -1)),
    ]:
        reset_non_goal_state()
        valid = env._move(move)
        assert valid, f"Expected move '{move}' to be valid from non-goal state."
        assert env._get_empty_position() == (
            NG_BLANK_POS[0] + blank_tile_delta[0],
            NG_BLANK_POS[1] + blank_tile_delta[1],
        ), (
            f"After move '{move}', expected blank tile position to be "
            f"{(NG_BLANK_POS[0] + blank_tile_delta[0], NG_BLANK_POS[1] + blank_tile_delta[1])}, "
            f"got {env.blank_pos}."
        )
    print("All transition tests passed.")

def solvability_checks_work_properly():
    env = classic_puzzle()

    # Known solvable board
    SOLVABLE_BOARD = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, None, 15],
    ]
    assert env._is_solvable(SOLVABLE_BOARD), "Expected known solvable board to be solvable."

    # Known unsolvable board
    UNSOLVABLE_BOARD = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 15, 14, None],
    ]
    assert not env._is_solvable(UNSOLVABLE_BOARD), "Expected known unsolvable board to be unsolvable."

    print("All solvability check tests passed.")

def generated_boards_are_valid_and_solvable(num_tests: int = 10):

    def manhattan_lc_heuristic(board: List[List[Optional[int]]]) -> int:
        '''
        Beyond just letting tiles move with impunity, we consider how
        the blank tile must also move to facilitate these moves.

        The Linear Conflict (https://michael.kim/blog/puzzle) technique
        allows us to conservatively consider additional moves needed when
        two tiles are in their goal row/column but are reversed.
        '''
        pos = {} # Map actual position to the goal position
        for r in range(len(board)):
            for c in range(len(board[0])):
                pos[(r, c)] = (
                    (len(board) - 1, len(board[0]) - 1)
                    if board[r][c] is None 
                    else ((board[r][c] - 1) // len(board[0]), (board[r][c] - 1) % len(board[0]))
                )

        mh_dist = sum(
            abs(r - pos[(r, c)][0]) + abs(c - pos[(r, c)][1])
            for r in range(len(board))
            for c in range(len(board[0]))
            if board[r][c] is not None
        )

        # 1 additional blank move is needed for any two tiles which are out of place (row-wise)
        num_out_of_place = 0
        for r1, c1 in product(range(len(board)), range(len(board[0]))):
            if board[r1][c1] is None:
                continue
            for r2 in range(len(board)):
                if board[r2][c1] is None or (r1 == r2):
                    continue
                # Out of place: i.e. goal order is different from row order
                if (
                    (
                        (r1 < r2 and pos[(r1, c1)][0] > pos[(r2, c1)][0])
                        or (r1 > r2 and pos[(r1, c1)][0] < pos[(r2, c1)][0])
                    ) 
                    and (pos[(r1, c1)][1] == pos[(r2, c1)][1])
                    and pos[(r1, c1)][1] == c1
                ):
                    num_out_of_place += 1

            for c2 in range(len(board[0])):
                if board[r1][c2] is None or (c1 == c2):
                    continue
                # Out of place: i.e. goal order is different from column order
                if (
                    (
                        (c1 < c2 and pos[(r1, c1)][1] > pos[(r1, c2)][1])
                        or (c1 > c2 and pos[(r1, c1)][1] < pos[(r1, c2)][1])
                    ) 
                    and (pos[(r1, c1)][0] == pos[(r1, c2)][0])
                    and pos[(r1, c1)][0] == r1
                ):
                    num_out_of_place += 1
        return mh_dist + num_out_of_place

    def inversion_heuristic(board: List[List[Optional[int]]]) -> int:
        '''
        Second heuristic: Count number of inversions on the board.
        (https://michael.kim/blog/puzzle)

        Vertical moves of the blank tile fix up to 3 inversions, (good lower bound!)
        while horizontal ones leave them unchanged. (cf. how we compute the invariant)
        '''
        flat_board = [
            board[r][c]
            for r in range(len(board))
            for c in range(len(board[0]))
            if board[r][c] is not None
        ]
        num_inversions = 0
        
        # Unfortunatly we need an efficient inversion algorithm since
        # the manhattan + LC is O(n^{3/2}).
        def mod_mergesort(flat_board: List[int]) -> Tuple[List[int], int]:
            if len(flat_board) <= 1:
                return flat_board, 0
            mid = len(flat_board) // 2
            left, left_inv = mod_mergesort(flat_board[:mid])
            right, right_inv = mod_mergesort(flat_board[mid:])
            merged = []
            i = j = 0
            inversions = left_inv + right_inv

            # Merge routine: This is the only place where new inversions can be added.
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    inversions += len(left) - i
                    j += 1
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged, inversions
        
        _, num_inversions = mod_mergesort(flat_board)
        return num_inversions // 3
    
    def improved_manhattan_heuristic(board: List[List[Optional[int]]]) -> int:
        return max(
            manhattan_lc_heuristic(board),
            inversion_heuristic(board)
        )

    def a_star_solve(
            start_board: List[List[Optional[int]]], 
            heuristic: Callable[[List[List[Optional[int]]]], int]
        ) -> Optional[List[str]]:
        # A* solver to verify solvability of generated boards.

        def state_hash(board: List[List[Optional[int]]]) -> str:
            return '\n'.join(
                ','.join(
                    str(board[r][c]) if board[r][c] is not None else 'X'
                    for c in range(len(board[0]))
                )
                for r in range(len(board))
            )
        
        def state_unhash(s: str) -> List[List[Optional[int]]]:
            return [
                [
                    int(x) if x != 'X' else None
                    for x in row.split(',')
                ]
                for row in s.split('\n')
            ]

        dummy_env = classic_puzzle()
        dummy_env.reset()
        fix_state(dummy_env, start_board)

        pq: List[Tuple[int, int, List[List[Optional[int]]]]] = []
        parent: Dict[str, Optional[str]] = {state_hash(start_board): None}
        heappush(pq, (heuristic(start_board), 0, 0, 0, state_hash(start_board), None))
        visited = {}

        n_exp = 0
        while pq:
            f, sec_prio, g, curr_turn, board_hash, last_move = heappop(pq)
            board = state_unhash(board_hash)

            fix_state(dummy_env, board, steps = curr_turn)

            if board_hash in visited and visited[board_hash] <= g:
                continue
            elif curr_turn >= dummy_env.max_turns:
                continue
            elif dummy_env._is_solved():
                print(f"Solution found. Cost: {g} moves. Nodes expanded: {n_exp}.")
                # Reconstruct path
                path = []
                shash = state_hash(board)
                while parent[shash] is not None:
                    path.append(parent[shash][1])  # move description
                    shash = parent[shash][0]
                return path[::-1], g # Return reversed path and cost
            n_exp += 1
            if n_exp % 1000 == 0:
                print(f"Expanded {n_exp} nodes so far. Frontier size: {len(pq)}. Frontier min f: {pq[0][0]}")
            visited[board_hash] = g
            
            # While entirely possible to use a faster representation,
            # we respect the existing environment structure for clarity.
            for move in ["up", "down", "left", "right"]:
                if (
                    (last_move == "down" and move == "up")
                    or (last_move == "up" and move == "down")
                    or (last_move == "left" and move == "right")
                    or (last_move == "right" and move == "left")
                ):
                    # Don't immediately undo the last move.
                    continue
                fix_state(dummy_env, board, steps = curr_turn)
                valid = dummy_env._move(move)
                if not valid:
                    continue
                new_board = deepcopy(dummy_env.board)
                assert any (
                    new_board[r][c] != board[r][c]
                    for r in range(len(board))
                    for c in range(len(board[0]))
                ), "Move did not change the board state."
                new_board_hash = state_hash(new_board)
                new_g = g + 1
                if new_board_hash not in visited or visited[new_board_hash] > new_g:
                    parent[new_board_hash] = (board_hash, move)
                    new_h = heuristic(new_board)
                    new_f = new_g + new_h
                    heappush(
                        pq, 
                        (
                            new_f,
                            -new_g,
                            new_g,
                            curr_turn + 1,
                            state_hash(new_board),
                            move
                        )
                    )

        return None, None  # No solution found

    # Use the A* Algorithm to verify solvability.
    # On the classic problem, we expect this to take about... 50 moves???
    # Not sure, but as long as some solution pops up, we are good.
    env = classic_puzzle()

    # # Preliminary test: Check that A* works on a known solvable board.
    # KNOWN_SOLVABLE = [[7, 9, None, 12], [1, 3, 5, 2], [6, 13, 11, 8], [15, 14, 4, 10]]
    # solution, cost = a_star_solve(KNOWN_SOLVABLE, heuristic = manhattan_heuristic)
    # assert solution is not None, "A* failed to find a solution on a known solvable board."
    # print(f"A* preliminary test passed. Solution cost: {cost} moves.")
    # print(f"Solution moves: {solution}")

    # Now test generated boards.

    for i in range(num_tests):
        env.reset()
        board = deepcopy(env.board)
        solution, cost = a_star_solve(board, heuristic = improved_manhattan_heuristic)
        assert solution is not None, (
            f"Generated board on test {i} is not solvable:\n"
            f"{''.join(str(row) + '\n' for row in board)}"
        )

        verifier = classic_puzzle()
        fix_state(verifier, board, steps = 0)
        for move in solution:
            ortti = verifier.step(f"\\boxed{{{move}}}")
            print(f"Applied move: {move}")
            print(f"Legality: {ortti[0]}, Reward: {ortti[1]}, Done: {ortti[2]}, Info: {ortti[3]}")
            print(f"Current board:\n{''.join(str(row) + '\n' for row in verifier.board)}")

        assert verifier._is_solved(), (
            f"Solution provided does not solve the board. Expected board:\n"
            f"{''.join(str(row) + '\n' for row in verifier._generate_goal_board())}, got\n"
            f"{''.join(str(row) + '\n' for row in verifier.board)}"
        )

        print(f"Test {i + 1}/{num_tests} passed: Generated board is valid and solvable.")
        print(f"Solution length: {len(solution)} moves.")
        print(f"Start board:\n{''.join(str(row) + '\n' for row in board)}")
    
    print("All generated board tests passed.")
        

def other_environment_checks():
    # Check for other things like step limits, reset functionality, etc.
    env = classic_puzzle()

    # Test 1: Ensure reset removes turn counts.
    for _ in range(10):
        env.step(env.sample_random_action())
    env.reset()
    assert env.turn_count == 0, f"Expected current turn to be 0 after reset, got {env.turn_count}."

    # Test 2: Ensure step limits are enforced.
    env.reset()
    ortti = None
    for _ in range(env.max_turns):
        ortti = env.step(f"\\boxed{{{env.sample_random_action()}}}")
        print(ortti)
    assert ortti[2] or ortti[3], f"Expected environment to be done after max turns. {ortti}"
    print("All other environment checks passed.")

def test_all():
    transitions_work_properly()
    solvability_checks_work_properly()
    generated_boards_are_valid_and_solvable()
    other_environment_checks()

if __name__ == "__main__":

    fire.Fire({
        "transitions_work_properly": transitions_work_properly,
        "solvability_checks_work_properly": solvability_checks_work_properly,
        "generated_boards_are_valid_and_solvable": generated_boards_are_valid_and_solvable,
        "other_environment_checks": other_environment_checks,
        "test_all": test_all,
    })

    '''
    Usage:
    python -m tests.test_env.test_fifteen <TEST_NAME>
    '''