import numpy as np
from functools import reduce
import random

class Puzzle:
    def __init__(self, bitmap):
        self.hints = [[],[]]
        self.solution = bitmap
        self.height = len(bitmap)
        self.width = len(bitmap[0])
        h, v = 0, 0
        for i in range(self.height):
            self.hints[0].append([])
            for j in range(self.width):
                if bitmap[i][j]:
                    h += 1
                elif h > 0:
                    self.hints[0][i].append(h)
                    h = 0
            if h > 0:
                self.hints[0][i].append(h)
                h = 0
            if len(self.hints[0][i]) == 0:
                self.hints[0][i].append(0)

        for i in range(self.width):
            self.hints[1].append([])
            for j in range(self.height):
                if bitmap[j][i]:
                    v += 1
                elif v > 0:
                    self.hints[1][i].append(v)
                    v = 0
            if v > 0:
                self.hints[1][i].append(v)
                v = 0
            if len(self.hints[1][i]) == 0:
                self.hints[1][i].append(0)

    def verify_rows(self, map):
        aux_hints = [[],[]]
        h, v = 0, 0
        for i in range(len(map)):
            aux_hints[0].append([])
            for j in range(len(map[0])):
                if map[i][j]:
                    h += 1
                elif h > 0:
                    aux_hints[0][i].append(h)
                    h = 0
            if h > 0:
                aux_hints[0][i].append(h)
                h = 0
            if len(aux_hints[0][i]) == 0:
                aux_hints[0][i].append(0)
            if aux_hints[0][i] != self.hints[0][i]:
                return False
        return True

    def verify_cols(self, map):
        aux_hints = [[],[]]
        h, v = 0, 0
        for i in range(len(map[0])):
            aux_hints[1].append([])
            for j in range(len(map)):
                if map[j][i]:
                    v += 1
                elif v > 0:
                    aux_hints[1][i].append(v)
                    v = 0
            if v > 0:
                aux_hints[1][i].append(v)
                v = 0
            if len(aux_hints[1][i]) == 0:
                aux_hints[1][i].append(0)
            if aux_hints[1][i] != self.hints[1][i]:
                return False
        return True

    def verify_map(self, map):
        return self.verify_rows(map) and self.verify_cols(map)

    def still_works_vert(self, map):
        #the max number of black squares is lower than the hints maximum
        aux_hints = [[],[]]
        h, v = 0, 0
        for i in range(len(map[0])):
            aux_hints[1].append([])
            for j in range(len(map)):
                if map[j][i]:
                    v += 1
                elif v > 0:
                    aux_hints[1][i].append(v)
                    v = 0
            if v > 0:
                aux_hints[1][i].append(v)
                v = 0
            if len(aux_hints[1][i]) == 0:
                aux_hints[1][i].append(0)

            count = 0
            for j in range(len(aux_hints[1][i]) - 1):
                count += 1
                if count >= len(self.hints[1][i]) or aux_hints[1][i][j] != self.hints[1][i][j]:
                    return False

            if aux_hints[1][i][-1] > self.hints[1][i][count]:
                return False
        return True

    def still_works_vert(self, map):
        #the max number of black squares is lower than the hints maximum
        aux_hints = [[],[]]
        h, v = 0, 0
        for i in range(len(map[0])):
            aux_hints[1].append([])
            for j in range(len(map)):
                if map[j][i]:
                    v += 1
                elif v > 0:
                    aux_hints[1][i].append(v)
                    v = 0
            if v > 0:
                aux_hints[1][i].append(v)
                v = 0
            if len(aux_hints[1][i]) == 0:
                aux_hints[1][i].append(0)

            count = 0
            for j in range(len(aux_hints[1][i]) - 1):
                count += 1
                if count >= len(self.hints[1][i]) or aux_hints[1][i][j] != self.hints[1][i][j]:
                    return False

            if aux_hints[1][i][-1] > self.hints[1][i][count]:
                return False
        return True

    def solve_naive_backtrack(self):
        base_map = [[False]*self.width for i in range(self.height)]
        solved_map = [[None]*self.width for i in range(self.height)]
        self.__naive_backtrack(base_map, solved_map, len(base_map), len(base_map[0]))
        if solved_map[0][0] == None:
            print("no solution")
            solved_map = None

        return solved_map

    def __naive_backtrack(self, bitmap, solved_map, max_y, max_x, y = 0, x = 0):
        if solved_map[0][0] == None:
            if self.verify_map(bitmap):
                for i in range(len(bitmap)):
                    for j in range(self.width):
                        solved_map[i][j] = bitmap[i][j]
            elif y < max_y:
                aux = bitmap[y][x]
                for i in [True, False]:
                    bitmap[y][x] = i
                    next_x = (x + 1) % max_x
                    next_y = y
                    if next_x == 0:
                        next_y += 1
                    self.__naive_backtrack(bitmap, solved_map, max_y, max_x, next_y, next_x)
                bitmap[y][x] = aux


    def solve_perm_backtrack(self):
        base_map = [[False]*self.width for i in range(self.height)]
        solved_map = [[None]*self.width for i in range(self.height)]
        self.__perm_backtrack(base_map, solved_map, len(base_map))
        if solved_map[0][0] == None:
            print("no solution")
            solved_map = None

        return solved_map

    def __perm_backtrack(self, bitmap, solved_map, max_y, y = 0):
        if solved_map[0][0] == None:
            if self.verify_map(bitmap):
                print("nice")
                for i in range(self.height):
                    for j in range(self.width):
                        solved_map[i][j] = bitmap[i][j]
            elif y < max_y and self.still_works_vert(bitmap):
                aux = bitmap[y].copy()
                for i in self.row_perms(self.hints[0][y]):
                    bitmap[y] = i
                    self.__perm_backtrack(bitmap, solved_map, max_y, y+1)
                bitmap[y] = aux

    def solve(self):
        solution = self.solve_perm_backtrack();
        for i in solution:
            for j in i:
                if j:
                    print("■ ",end="")
                else:
                    print("□ ",end="")
            print()

    def display_solution(self):
        for i in self.solution:
            for j in i:
                if j:
                    print("■ ",end="")
                else:
                    print("□ ",end="")
            print()

    def n_row_perms(self, row_hints):
        # m - s = x1 + x2 + ... + xn; xi := size of gap between block i and i-1
        # only x1 and xn can be 0
        # only for non ordered pemutations :/

        m = self.width
        s = sum(row_hints)
        solution = 0

        if m > s + len(row_hints) - 1:
            polys = []
            max_gap_len = m - s
            for i in range(len(row_hints) + 1):
                if(i == len(row_hints) or i == 0):
                    #ordered permutations
                    polys.append(np.poly1d([1/np.math.factorial(i) for i in range(max_gap_len)]))
                    #non ordered permutations
                    #polys.append(np.poly1d([1]*max_gap_len))
                else:
                    #polys.append(np.poly1d([1]*(max_gap_len - 1) + [0]))
            solution = reduce(lambda x, y: x*y, polys)[m - s]
        elif m == s + len(row_hints) - 1:
            solution = 1

        return solution

    def row_perms(self, row_hints):
        m = len(self.solution[0])
        s = sum(row_hints)
        row_sols = []
        if m > s + len(row_hints) - 1:
            i = 0
            #gap_lengths is the posible gap positions
            gap_lengths = zero_pad_partitions(partitions_limited_count(m - s, len(row_hints) - 1, len(row_hints) + 1), len(row_hints) + 1)
            for gap_option in gap_lengths:
                aux_row = []
                for i in range(len(row_hints)):
                    aux_row += [False] * gap_option[i]
                    aux_row += [True] * row_hints[i]
                aux_row += [False] * gap_option[len(row_hints)]
                row_sols.append(aux_row)


        elif m == s + len(row_hints) - 1:
            row_sols.append([])
            row_sols[0] += [True] * row_hints[0]
            for i in row_hints[1:]:
                row_sols[0] += [False]
                row_sols[0] += [True] * i

        return row_sols

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
        else:
            for j in range(length - len(i) + 1):
                solution.append([0]*j + i + [0]*((length - len(i)) - j) )
    return solution

def generate_puzzle(size, p):
    bitmap = []
    for i in range(size):
        bitmap.append([])
        for j in range(size):
            bitmap[i].append(random.random() > p)
    return bitmap


T = True
F = False

bitmapHard = [
    [F, F, T, T, T, F, F, T, F, F, T, T, F, F, F],
    [F, T, T, F, T, F, F, T, T, F, F, T, F, T, F],
    [T, F, F, T, T, T, T, T, F, T, T, T, F, F, T],
    [T, T, T, T, T, T, T, T, T, T, T, T, F, F, T],
    [T, F, T, T, F, F, F, F, T, F, F, F, F, F, F],
    [F, T, F, F, F, T, T, T, F, T, F, F, T, F, T],
    [F, T, T, T, F, F, T, F, T, T, F, T, F, T, T],
    [F, T, F, F, F, F, T, F, T, F, T, T, F, T, T],
    [T, T, T, T, F, F, F, F, F, F, T, T, T, T, T],
    [F, F, T, F, F, T, F, F, F, F, T, F, T, F, T],
    [F, F, F, T, F, T, T, F, T, F, T, F, T, T, T],
    [T, F, T, T, F, F, T, T, F, F, F, T, F, F, F],
    [T, F, T, T, F, F, T, F, F, F, F, F, F, F, T],
    [T, F, T, T, F, F, T, T, F, T, T, F, T, T, T],
    [T, T, T, F, F, T, F, F, T, T, F, F, T, F, T]
]


bitmap0 = [
    [T, F, T, F, T, F, F, T],
    [F, T, F, F, T, F, F, T],
    [F, T, F, T, T, F, F, T],
    [F, T, F, F, T, F, F, T],
    [T, T, F, F, T, F, F, T],
    [F, T, F, F, T, F, F, T],
    [F, T, F, T, T, F, F, T],
    [F, T, F, F, T, F, F, T],
    [F, T, F, T, T, F, F, T],
    [F, T, F, F, T, F, F, T],
    [F, T, F, T, T, F, F, T]
]

bitmap1 = [
    [T, F, T, F, T],
    [F, T, F, F, T],
    [F, T, F, T, T],
    [F, T, F, F, T],
    [T, T, F, F, T],
    [F, T, F, F, T]
]

bitmap2 = [
    [T, F, T, F, T],
    [F, T, F, F, F],
    [F, T, F, F, T]
]

bitmap3 = [
    [T, F, T, F, T],
    [F, T, F, T, F],
    [F, T, F, T, F]
]
