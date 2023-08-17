from __future__ import annotations
from copy import copy, deepcopy 
import numpy as np
import random
from functools import reduce
from ..permutation_utils import *
from ..PicrossPuzzle import PicrossPuzzle, display_solution

def solve_naive_backtrack(puzzle: PicrossPuzzle, progress = None) -> PicrossPuzzle:
    """
    Calls the recursive naive backtracking solver and returns the solution.
    """

    # Pointer to the solution, lists in python work as pointers when passed as an argument.
    solved_map = [np.full([puzzle.height, puzzle.width], -1, dtype=np.byte)]

    already_solved = False
    if progress is None:
        base_map = np.full([puzzle.height, puzzle.width], -1, dtype=np.byte)
    else:
        if already_solved := puzzle.verify_solution(progress):
            solved_map[0] = progress        
        base_map = progress

    if not already_solved: 
        _naive_backtrack_rec(puzzle, base_map, solved_map, puzzle.height, puzzle.width)
    
        if np.any(solved_map[0] == -1):
            solved_map[0] = None

    return solved_map[0]

def _naive_backtrack_rec(puzzle: PicrossPuzzle, bitmap: np.ndarray, solved_map: np.ndarray,
                        max_y: int, max_x: int, y: int = 0, x: int = 0):
    """
    Solves the puzzle using a naive backtracking approach
    changing the value of each cell each step.
    """

    if np.all(solved_map[0] != -1):
        return
    
    if puzzle.verify_solution(bitmap):
        solved_map[0] = np.copy(bitmap)
        return
    elif y >= max_y:
        return

    if bitmap[y, x] == -1:
        prev_value = bitmap[y, x]
        for i in [True, False]:
            bitmap[y, x] = i
            _naive_backtrack_rec(puzzle, bitmap, solved_map, max_y, max_x, y + ((x + 1) // max_x), (x + 1) % max_x)
        bitmap[y, x] = prev_value
    else:
        _naive_backtrack_rec(puzzle, bitmap, solved_map, max_y, max_x, y + ((x + 1) // max_x), (x + 1) % max_x)