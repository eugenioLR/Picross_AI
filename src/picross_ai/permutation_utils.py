from __future__ import annotations
import numpy as np

def partitions_limited_count_rec(n, min_elems, max_elems, solution, part_solution: List = None):
    """
    Recursively generates all the integer parpartitions of `n` that has more than `min_elems` and less than `max_elems`.
    The solution is modified directly over the `solution` variable. 

    time complexity O(2^n)
    space complexity O(2^n)
    """

    if part_solution is None:
        part_solution = []

    #ordered partitions
    if max_elems >= 0:
        if n > 0:
            # for unordered partitions change 1 with the i of the previous call
            for i in range(1, n + 2 - min_elems):
                partitions_limited_count_rec(n - i, min_elems-1, max_elems-1, solution, part_solution + [i])
        elif n == 0 and min_elems <= 0:
            solution.append(part_solution)

# def partitions_limited_count_iter(n, min_elems, max_elems, solution, part_solution: List = None):
def partitions_limited_count_iter(n, min_elems, max_elems, solution):
    """
    Direct conversion of `partitions_limited_count_rec` from a recursive function to an iterative one.
    This is done to avoid stack overflow errors though the amount of memory necesary might become too big.

    time complexity O(2^n)
    space complexity O(2^n)
    """

    call_stack = [(n, min_elems, max_elems, solution, None)]

    while call_stack:
        n, min_elems, max_elems, solution, part_solution = call_stack.pop()
        
        if part_solution is None:
            part_solution = []
        
        if max_elems >= 0:
            if n > 0:
                for i in range(1, n + 2 - min_elems):
                    call_stack.append((n - i, min_elems-1, max_elems-1, solution, part_solution + [i]))
            elif n == 0 and min_elems <= 0:
                solution.append(part_solution)

def partitions(n):
    """
    Obtains all the ordered integer partitions of `n`.
    """

    return partitions_limited_count(n, 0, n)

def partitions_fixed_count(n, elem):
    """
    Obtains all the ordered integer partitions of `n` with `elem` elements.
    """

    return partitions_limited_count(n,elem,elem)

def partitions_limited_count(n, min_elems, max_elems):
    """
    Obtains all the ordered integer partitions of `n` with more than `min_elem` elements and less than `max_elems`.
    """

    solution = []
    # partitions_limited_count_rec(n, min_elems, max_elems, solution)
    partitions_limited_count_iter(n, min_elems, max_elems, solution)
    return solution

def zero_pad_partitions(partitions, length):
    """
    Pads the partitions with zeros to the right and left
    """

    solution = []
    for i in partitions:
        if len(i) == length:
            solution.append(i)
        elif len(i) == length - 1:
            solution.append([0] + i)
            solution.append(i + [0])
        elif len(i) == length - 2:
            solution.append([0] + i + [0])
    return solution

def n_line_perms(width, height, row_hints, progress = None, isHoriz = True):
    """
    Calculates the number of valid permutations given a puzzle hint with generating functions.
    """

    # m - s = x1 + x2 + ... + xn; xi := size of gap between block i and i-1
    # only x1 and xn can be 0
    solution = 0
    if progress is None:
        m = width if isHoriz else height
        s = sum(row_hints)
        if len(row_hints) == 1:
            solution = m - s + 1
        else:
            if m > s + len(row_hints) - 1:
                poly1 = np.poly1d([1 for i in range(m-s)])
                poly2 = np.poly1d([1 for i in range(m-s)] + [0])
                solution = ((poly2**(len(row_hints) - 1)) * (poly1**2))[m - s]
            elif m == s + len(row_hints) - 1:
                solution = 1
    else:
        solution = len(list(line_perms(width, height, row_hints, progress, isHoriz)))
    return solution

def line_perms(width, height, line_hints, progress = None, isHoriz = True):
    """
    Generates the valid permutations given a puzzle hint that satisfy the line constraints.
    """

    m = width if isHoriz else height
    s = sum(line_hints)
    hint_length = len(line_hints)

    if progress is None:
        progress = np.full(m, -1)

    line_sols = []
    if m > s + hint_length - 1:
        i = 0

        # gap_lengths is the posible gap positions
        gap_lengths = zero_pad_partitions(partitions_limited_count(m - s, hint_length - 1, hint_length + 1), hint_length + 1)
        for gap_option in gap_lengths:
            aux_line = np.zeros(m)
            cursor = 0
            for i in range(hint_length):
                cursor += gap_option[i]
                aux_line[cursor:cursor+line_hints[i]] = 1
                cursor += line_hints[i]

            if np.all((progress == -1) | (progress == aux_line)):
                yield aux_line


    elif m == s + hint_length - 1:
        row = np.ones(m)
        cursor = line_hints[0]
        for i in line_hints[1:]:
            row[cursor] = 0
            cursor += i + 1
        yield row

    return None

def common_from_perms(permutations: List) -> List:
    """
    Finds the intersection between each of the permutations provided.
    """
    
    perm_arr = np.array(permutations)
    result = perm_arr[0,:]
    result[np.any(result != perm_arr, axis=0)] = -1
    
    return result
