from __future__ import annotations
from copy import copy, deepcopy 
import numpy as np
import random
from functools import reduce
from .iterated_intersections import iterated_intersections
from ..permutation_utils import *
from ..PicrossPuzzle import PicrossPuzzle, display_solution

def solve_rand_row_backtrack(puzzle: PicrossPuzzle, progress: np.ndarray = None, verbose = True):
    """
    Calls the recursive randomized row-wise backtracking solver and returns a solution.

    All the valid combinations of cells on each row are generated and stored 
    in memory so be careful using this solver for large puzzles.
    """

    solved_map = [np.full([puzzle.height, puzzle.width], -1, dtype=np.byte)]

    already_solved = False
    if progress is None:
        progress = np.full([puzzle.height, puzzle.width], -1, dtype=np.byte)
        base_map = np.zeros([puzzle.height, puzzle.width], dtype=np.byte)
    else:
        if already_solved := puzzle.verify_solution(progress):
            solved_map[0] = progress
        base_map = progress
    
    if not already_solved:
        row_costs = np.zeros(puzzle.height).astype(int)
        for i in range(puzzle.height):
            row_costs[i] = n_line_perms(puzzle.width, puzzle.height, puzzle.hints[0][i], progress[i, :])
        
        row_order = row_costs.argsort()

        n_possibilities = reduce(lambda x, y: int(x)*int(y), row_costs)
        
        if verbose:
            print(f"Estimation of complexity:")
            print(f"- Combinations per row: {row_costs}")
            print(f"- Total number of combinations: {n_possibilities} = {n_possibilities:e}")
            print(f"- Chosen order of rows: {row_order} with costs {row_costs[row_order]}")
            print()

        _rand_row_backtrack_rec(puzzle, base_map, solved_map, progress, row_order)

    return solved_map[0]

def _rand_row_backtrack_rec(puzzle: PicrossPuzzle, bitmap: np.ndarray, solved_map: np.ndarray,
                       progress: np.ndarray, row_order: np.ndarray):
    """
    Solves the puzzle using a randomized row-wise backtracking approach checking every valid row.
    """

    if np.all(solved_map[0] != -1):
        return
    
    if puzzle.verify_solution(bitmap):
        solved_map[0] = np.copy(bitmap)
        return
    elif row_order.size == 0:
        return

    row = row_order[0]
    aux = bitmap[row, :].copy()
    possibilities = list(line_perms(puzzle.width, puzzle.height, puzzle.hints[0][row], progress[row, :]))
    random.shuffle(possibilities)
    for perm in possibilities:
        bitmap[row, :] = perm
        _rand_row_backtrack_rec(puzzle, bitmap, solved_map, progress, row_order[1:])
    bitmap[row, :] = aux