from __future__ import annotations
from copy import copy, deepcopy 
import numpy as np
import random
from functools import reduce
from ..permutation_utils import *
from ..PicrossPuzzle import PicrossPuzzle, display_solution

def iterated_intersections(puzzle, times = 1, progress = None, solvable = True):
    """
    Repeatedly calculates the overlap between all possible permutations of each row
    and column.
    """

    if progress is not None and puzzle.verify_solution(progress):
        return (progress, True)
    
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

    if solvable and times > 1 and np.any(progress != pre_solution) and not puzzle.verify_solution(pre_solution):
        puzzle.transpose()
        pre_solution, solvable = iterated_intersections(puzzle, times-1, pre_solution.T, solvable)
        pre_solution = pre_solution.T
        puzzle.transpose()
    
    return (pre_solution, solvable)
        