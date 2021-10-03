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
    for i in partitions:
        if len(i) == length:
            solution.append(i)
        elif len(i) == length - 1:
            solution.append([0] + i)
            solution.append(i + [0])
        elif len(i) == length - 2:
            solution.append([0] + i + [0])
    return solution
