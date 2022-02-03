import numpy as np

def partitions_limited_count_aux(n, min_elems, max_elems, solution, part_solution = []):
    #ordered partitions
    if max_elems >= 0:
        if n > 0:
            #for unordered partitions change 1 with the i of the previous call
            for i in range(1, n + 2 - min_elems):
                partitions_limited_count_aux(n - i, min_elems-1, max_elems-1, solution, part_solution + [i])
        elif n == 0 and min_elems <= 0:
            solution.append(part_solution)

def partitions(n):
    return partitions_limited_count(n, 0, n)

def partitions_fixed_count(n, elem):
    return partitions_limited_count(n,elem,elem)

def partitions_limited_count(n, min_elems, max_elems):
    solution = []
    partitions_limited_count_aux(n, min_elems, max_elems, solution)
    return solution

def zero_pad_partitions(partitions, length):
    solution = []
    #print("partitions:", partitions)
    for i in partitions:
        if len(i) == length:
            solution.append(i)
        elif len(i) == length - 1:
            solution.append([0] + i)
            solution.append(i + [0])
        elif len(i) == length - 2:
            solution.append([0] + i + [0])
    return solution

def n_line_perms(width, height, row_hints, isHoriz = True):
    # m - s = x1 + x2 + ... + xn; xi := size of gap between block i and i-1
    # only x1 and xn can be 0
    # only for non ordered pemutations

    m = width if isHoriz else height
    s = sum(row_hints)
    solution = 0
    if len(row_hints) == 1:
        solution = m - s + 1
    else:
        if m > s + len(row_hints) - 1:
            polys = []
            poly1 = np.poly1d([1]*(m-s))
            poly2 = np.poly1d([1]*(m-s) + [0])
            solution = ((poly2**(len(row_hints) - 1)) * (poly1**2))[m - s]
        elif m == s + len(row_hints) - 1:
            solution = 1

    return solution

def line_perms(width, height, line_hints, progress = None, isHoriz = True):
    m = width if isHoriz else height
    s = sum(line_hints)
    hint_length = len(line_hints)

    if progress is None:
        progress = -np.ones(m)

    #line_sols = np.zeros([m, hint_length])
    line_sols = []
    if m > s + hint_length - 1:
        i = 0
        # gap_lengths is the posible gap positions
        gap_lengths = zero_pad_partitions(partitions_limited_count(m - s, hint_length - 1, hint_length + 1), hint_length + 1)
        #print("gap lengths:", gap_lengths)
        for gap_option in gap_lengths:
            aux_line = np.zeros(m)
            cursor = 0
            for i in range(hint_length):
                cursor += gap_option[i]
                aux_line[cursor:cursor+line_hints[i]] = 1
                cursor += line_hints[i]

            valid = np.all(np.logical_or(progress == -1, np.equal(aux_line, progress)))
            if valid:
                line_sols.append(aux_line)


    elif m == s + hint_length - 1:
        row = np.ones(m)
        cursor = line_hints[0]
        for i in line_hints[1:]:
            row[cursor] = 0
            cursor += i + 1
        line_sols.append(row)

    return line_sols

def common_from_perms(permutations):
    aux = permutations[0]
    l = len(aux)
    finish = False
    i = 0
    while i < len(permutations) and not finish:
        j = 0
        while j < l:
            if aux[j] != -1 and permutations[i][j] != aux[j]:
                aux[j] = -1
            j += 1
        finish = np.count_nonzero(aux == -1) == len(aux)
        i += 1
    return aux
