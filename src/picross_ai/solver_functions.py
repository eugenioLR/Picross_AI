from __future__ import annotations
from copy import copy, deepcopy 
import numpy as np
import random
from functools import reduce
from itertools import tee
from .permutation_utils import *
from .PicrossPuzzle import PicrossPuzzle, display_solution

def solve_optimized_backtrack(puzzle: PicrossPuzzle, progress: np.ndarray = None):
    """
    Calls the recursive optimized backtracking solver and returns the solution.
    Accepts a partial solution to work from.
    """

    solved_map = [np.full([puzzle.height, puzzle.width], -1, dtype=np.byte)]

    already_solved = False
    if progress is None:
        base_map = np.zeros([puzzle.height, puzzle.width], dtype=np.byte)
        progress = np.full([puzzle.height, puzzle.width], -1, dtype=np.byte)

        row_costs = np.zeros(puzzle.height).astype(int)
        for i in range(puzzle.height):
            row_costs[i] = n_line_perms(puzzle.width, puzzle.height, puzzle.hints[0][i], progress[i, :])
        n_ones = np.count_nonzero(row_costs == 1)
        row_order = row_costs.argsort()[::-1]

        print(f"Estimation of complexity:")
        print(f"- Combinations per row: {row_costs}")
        print(f"- Total number of combinations: {reduce(lambda x, y: x*y, row_costs)}")
        print(f"- Chosen order of rows: {row_order} with costs {row_costs[row_order]}")
    else:
        if already_solved := puzzle.verify_solution(progress):
            solved_map[0] = progress
        
        base_map = progress
        row_costs = np.zeros(puzzle.height).astype(int)
        for i in range(puzzle.height):
            row_costs[i] = n_line_perms(puzzle.width, puzzle.height, puzzle.hints[0][i], progress[i, :])
        row_costs = row_costs[row_costs != 1]
        row_order = row_costs.argsort()

    
    if not already_solved:
        _optimized_backtrack_rec(puzzle, base_map, solved_map, progress, row_order)

    if np.any(solved_map[0] == -1):
        print()
        print("No solution found.")
        # solved_map[0] = None
    else:
        print()
        print("Solution found.")

    return solved_map[0]

def _optimized_backtrack_rec(puzzle: PicrossPuzzle, bitmap: np.ndarray, solved_map: np.ndarray,
                            progress: np.ndarray, row_order: List, solved = False, i: int = 0):
    """
    Solves the puzzle using an optimized backtracking approach.
    It starts from a partial solution.
    Each step it checks all the possible permutations of the rows not already completed.
    The algorithms starts from the most constrained row.
    Each step the linear solver is called.
    """

    # if i < row_order.shape[0]:
    #     print(row_order[i], end=",", flush=True)

    if np.all(solved_map[0] != -1):
        return
    
    if puzzle.verify_solution(bitmap):
        solved_map[0] = np.copy(bitmap)
        return 
    elif i >= row_order.shape[0] or not puzzle.still_works_vert(bitmap):
        return

    row = row_order[i]
    aux = np.copy(bitmap[row, :])
    for perm in line_perms(puzzle.width, puzzle.height, puzzle.hints[0][row], progress[row, :]):
        bitmap[row, :] = perm
        (new_bitmap, solvable) = partial_solution(puzzle, 5, bitmap)
        if solvable:
            _optimized_backtrack_rec(puzzle, bitmap, solved_map, new_bitmap, row_order, i + 1)
        # else:
            # _optimized_backtrack_rec(puzzle, bitmap, solved_map, progress, row_order, i + 1)
    bitmap[row, :] = aux

def partial_solution(puzzle, times = 1, progress = None, solvable = True):
    """
    Calculates the overlap between all possible permutations of each row.
    """

    if progress is None or not puzzle.verify_solution(progress):
        pre_solution = np.zeros([puzzle.height, puzzle.width])

        for i in range(puzzle.height):
            if solvable:
                if progress is None:
                    perm_gen = line_perms(puzzle.width, puzzle.height, puzzle.hints[0][i])
                else:
                    perm_gen = line_perms(puzzle.width, puzzle.height, puzzle.hints[0][i], progress[i,:])
                
                
                if solvable := perm_gen is not None:
                    permutations = list(perm_gen)
                    if solvable := permutations != []:
                        pre_solution[i,:] = common_from_perms(permutations)

        if solvable and times > 1 and not np.all(progress == pre_solution) and not puzzle.verify_solution(pre_solution):
            puzzle.transpose()
            pre_solution, solvable = partial_solution(puzzle, times-1, pre_solution.T, solvable)
            pre_solution = pre_solution.T
            puzzle.transpose()
        return (pre_solution, solvable)
    else:
        return (progress, True)

def solve(puzzle):
    """
    Nothing -> matrix(byte)
    Solves the puzzle.
    """

    print(puzzle.hints)

    # Preprocess the puzzle to solve it parcialy
    progress = partial_solution(puzzle, 100)[0]

    print("partial solution")
    display_solution(progress)

    # Decide if swaping the puzzle will result in a simpler problem
    horiz_cost = 1.0
    for i in range(len(puzzle.hints[0])):
        horiz_cost *= n_line_perms(puzzle.width, puzzle.height,puzzle.hints[0][i], progress[i, :], isHoriz = True)
        #horiz_cost *= len(line_perms(puzzle.width, puzzle.height,puzzle.hints[0][i], progress[i, :], isHoriz = True))

    progress_t = progress.T
    vert_cost = 1.0
    for i in range(len(puzzle.hints[1])):
        vert_cost *= n_line_perms(puzzle.width, puzzle.height,puzzle.hints[1][i], progress_t[i, :], isHoriz = False)
        #vert_cost *= len(line_perms(puzzle.width, puzzle.height,puzzle.hints[1][i], progress_t[i, :], isHoriz = False))

    print("horiz: {:}".format(round(horiz_cost)), "vert: {:}".format(round(vert_cost)))

    #solution = puzzle.solve_optimized_backtrack();
    #solution = puzzle.solve_naive_backtrack();

    if vert_cost < horiz_cost:
       puzzle.transpose()
       progress = progress.T
       print("Transposition is beneficial")

    # Solve the puzzle
    solution = solve_optimized_backtrack(puzzle, progress);
    #solution = puzzle.solve_naive_backtrack();

    # Restore the puzzle
    if vert_cost < horiz_cost:
       puzzle.transpose()
       progress = progress.T
       if solution is not None:
           solution = solution.T

    # Display the puzzle
    if solution is None:
        print("The result that should have appeared")
        display_solution(progress)
    else:
        display_solution(solution)