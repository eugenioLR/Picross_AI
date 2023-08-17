from __future__ import annotations
from copy import copy, deepcopy 
import numpy as np
import random
from functools import reduce
from ..permutation_utils import *
from ..PicrossPuzzle import PicrossPuzzle, display_solution

def solve_random_backtrack(puzzle: PicrossPuzzle, progress = None, verbose = False) -> PicrossPuzzle:
    """
    Calls the recursive randomized backtracking solver and returns the solution.
    """

    # Pointer to the solution, lists in python work as pointers when passed as an argument.
    solved_map = [np.full([puzzle.height, puzzle.width], -1, dtype=np.byte)]

    # Initialize the necesary data structures and variables
    already_solved = False
    if progress is None:
        base_map = np.full([puzzle.height, puzzle.width], -1, dtype=np.byte)
    else:
        if already_solved := puzzle.verify_solution(progress):
            solved_map[0] = progress        
        base_map = progress

    # Call backtrack algorithm
    if not already_solved:
        if progress is None:
            n_cells = base_map.size
        else:
            n_cells = np.count_nonzero(progress == -1)

        if verbose:
            print(f"Estimation of complexity:")
            print(f"- Number of cells {n_cells}")
            print(f"- Total number of combinations {2**n_cells} = {2**n_cells:e}")
            print()
        
        _random_backtrack_rec(puzzle, base_map, solved_map, puzzle.height, puzzle.width)

    return solved_map[0]

def _random_backtrack_rec(puzzle: PicrossPuzzle, bitmap: np.ndarray, solved_map: np.ndarray,
                        max_y: int, max_x: int, y: int = 0, x: int = 0):
    """
    Solves the puzzle using a randomized backtracking approach
    changing the value of each cell each step.
    """

    # A solution has already been found
    if np.all(solved_map[0] != -1):
        return
    
    # Check if we have the solution of the puzzle or we have reached the end of the array
    if puzzle.verify_solution(bitmap):
        solved_map[0] = np.copy(bitmap)
        return 
    elif y >= max_y:
        return

    # Main backtracking section.
    # Try to set the current cell as 0 or 1, continue to the next cell.
    if bitmap[y, x] == -1:
        prev_value = bitmap[y, x]

        rand_bit = random.randint(0,1)
        possibilities = [rand_bit, 1-rand_bit]
        for i in possibilities:
            bitmap[y, x] = i
            _random_backtrack_rec(puzzle, bitmap, solved_map, max_y, max_x, y + ((x + 1) // max_x), (x + 1) % max_x)
        bitmap[y, x] = prev_value
    else:
        _random_backtrack_rec(puzzle, bitmap, solved_map, max_y, max_x, y + ((x + 1) // max_x), (x + 1) % max_x)