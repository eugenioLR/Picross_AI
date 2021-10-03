import numpy as np
from functools import reduce
from discrete_math import *
import random

class Puzzle:
    def __init__(self, bitmap):
        self.hints = [[],[]]
        self.solution = bitmap
        self.height = len(bitmap)
        self.width = len(bitmap[0])
        self.flipState = False
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
        aux_hints = []
        h = 0
        for i in range(len(map)):
            aux_hints.append([])
            for j in range(len(map[0])):
                if map[i][j]:
                    h += 1
                elif h > 0:
                    aux_hints[i].append(h)
                    h = 0

            if h > 0:
                aux_hints[i].append(h)
                h = 0

            if len(aux_hints[i]) == 0:
                aux_hints[i].append(0)

            if aux_hints[i] != self.hints[0][i]:
                return False

        return True

    def verify_cols(self, map):
        aux_hints = []
        v = 0
        for i in range(len(map[0])):
            aux_hints.append([])
            for j in range(len(map)):
                if map[j][i]:
                    v += 1
                elif v > 0:
                    aux_hints[i].append(v)
                    v = 0

            if v > 0:
                aux_hints[i].append(v)
                v = 0

            if len(aux_hints[i]) == 0:
                aux_hints[i].append(0)

            if aux_hints[i] != self.hints[1][i]:
                return False

        return True

    def verify_map(self, map):
        return self.verify_rows(map) and self.verify_cols(map)

    def still_works_horiz(self, map):
        # The max number of black squares is lower than the hints maximum
        aux_hints = [[],[]]
        h, v = 0, 0
        for i in range(self.height):

            # Count and verify the partial solution with the hints
            j_hint = 0
            aux = 0
            for j in range(self.width):
                if map[i][j]:
                    aux += 1
                else:
                    if aux > 0 or j_hint > len(aux_hints[0]) or aux != aux_hints[0][i]:
                        return False
                    aux = 0
                    j_hint += 1

            """
            aux_hints[0].append([])
            for j in range(self.width):
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

            count = 0
            for j in range(len(aux_hints[0][i]) - 1):
                count += 1
                if count >= len(self.hints[0][i]) or aux_hints[0][i][j] != self.hints[0][i][j]:
                    return False

            if aux_hints[1][i][-1] > self.hints[0][i][count]:
                return False
            """
        return True

    def still_works_vert(self, map):
        #the max number of black squares is lower than the hints maximum
        aux_hints = []
        v = 0
        for i in range(len(map[0])):
            aux_hints.append([])
            for j in range(len(map)):
                if map[j][i]:
                    v += 1
                elif v > 0:
                    aux_hints[i].append(v)
                    v = 0
            if v > 0:
                aux_hints[i].append(v)
                v = 0
            if len(aux_hints[i]) == 0:
                aux_hints[i].append(0)

            count = 0
            for j in range(len(aux_hints[i]) - 1):
                count += 1
                if count >= len(self.hints[1][i]) or aux_hints[i][j] != self.hints[1][i][j]:
                    return False

            if aux_hints[i][-1] > self.hints[1][i][count]:
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


    #idea, fusion, get common squares for all combinations of each row and fill the gaps with square by square backtracking
    def solve_perm_backtrack(self, progress = None):

        solved_map = [[None]*self.width for i in range(self.height)]
        base_map = [[False]*self.width for i in range(self.height)]
        if progress is None:
            progress = [[None]*self.width for i in range(self.height)]

        self.__perm_backtrack(base_map, solved_map, progress, len(base_map))

        if solved_map[0][0] == None:
            print()
            print("No solution found.")
            solved_map = None
        else:
            print()
            print("Solution found.")



        return solved_map

    def __perm_backtrack(self, bitmap, solved_map, progress, max_i, i = 0):
        if solved_map[0][0] is None:
            if self.verify_map(bitmap):
                for j in range(self.height):
                    for k in range(self.width):
                        solved_map[j][k] = bitmap[j][k]

            elif i < max_i and self.still_works_vert(bitmap):
                aux = bitmap[i].copy()
                for j in self.line_perms(self.hints[0][i], progress[i], isHoriz = True):
                    bitmap[i] = j
                    self.__perm_backtrack(bitmap, solved_map, progress, max_i, i+1)
                bitmap[i] = aux

    def solve(self):

        # Preprocess the puzzle to solve it parcialy
        progress = self.preprocess(100)

        if self.flipState:
            self.flip()
            progress = np.transpose(progress)

        # Decide if fliping the puzzle will result in a simpler problem
        horiz_cost = 1.0
        for i in range(len(self.hints[0])):
            horiz_cost *= len(self.line_perms(self.hints[0][i], progress[i], isHoriz = True))

        progress_t = np.transpose(progress)
        vert_cost = 1.0
        for i in range(len(self.hints[1])):
            vert_cost *= len(self.line_perms(self.hints[1][i], progress_t[i], isHoriz = False))

        print("horiz: {:e}".format(horiz_cost), "vert: {:e}".format(vert_cost))

        print("partial solution")
        self.display_solution(progress)

        if(vert_cost < horiz_cost):
            self.flip()
            progress = np.transpose(progress)

        # Solve the puzzle
        solution = self.solve_perm_backtrack(progress);

        # Restore the puzzle
        if(vert_cost < horiz_cost):
            self.flip()
            solution = np.transpose(solution)

        # Display the puzzle
        self.display_solution(solution)

    def preprocess(self, times = 1, progress = None):
        self.flip()

        pre_solution = []

        for i in range(self.height):
            if progress is None:
                pre_solution.append(self.common_from_perms(self.line_perms(self.hints[0][i])))
            else:
                pre_solution.append(self.common_from_perms(self.line_perms(self.hints[0][i], progress[i])))

        if times > 1 and not np.array_equal(progress, pre_solution) and not self.verify_map(pre_solution):
            pre_solution = self.preprocess(times-1, np.transpose(pre_solution))

        return pre_solution


    def flip(self):
        self.flipState = not self.flipState
        self.hints[0], self.hints[1] = self.hints[1], self.hints[0]
        self.height, self.width = self.width, self.height

    def display_solution(self, solution = None):
        if solution is None:
            print("preview of the solution")
            solution = self.solution

        for i in solution:
            for j in i:
                if j is None:
                    print("_ ",end="")
                elif j:
                    print("□ ",end="")
                else:
                    print("■ ",end="")

            print()

    def n_line_perms(self, row_hints, isHoriz = True):
        # m - s = x1 + x2 + ... + xn; xi := size of gap between block i and i-1
        # only x1 and xn can be 0
        # only for non ordered pemutations :/

        m = self.width if isHoriz else self.height
        s = sum(row_hints)
        solution = 0
        if len(row_hints) == 1:
            solution = m - s + 1
        else:
            if m > s + len(row_hints) - 1:
                polys = []
                poly1 = np.poly1d([1]*(m-s))
                poly2 = np.poly1d([1]*(m-s) + [0])
                solution = ((poly2**(len(row_hints) - 1)) * (poly1**2))[m - s]#*np.math.factorial(m-s)
            elif m == s + len(row_hints) - 1:
                solution = 1

        return solution

    def line_perms(self, line_hints, progress = [], isHoriz = True):
        m = self.width if isHoriz else self.height
        s = sum(line_hints)
        hint_length = len(line_hints)

        if len(progress) == 0:
            progress = [None] * m

        line_sols = []
        if m > s + hint_length - 1:
            i = 0
            #gap_lengths is the posible gap positions
            gap_lengths = zero_pad_partitions(partitions_limited_count(m - s, hint_length - 1, hint_length + 1), hint_length + 1)
            for gap_option in gap_lengths:
                aux_line = []
                for i in range(hint_length):
                    aux_line += [False] * gap_option[i]
                    aux_line += [True] * line_hints[i]
                aux_line += [False] * gap_option[hint_length]
                valid = all([y is None or x == y for x, y in zip(aux_line, progress)])
                if valid:
                    line_sols.append(aux_line)


        elif m == s + hint_length - 1:
            line_sols.append([])
            line_sols[0] += [True] * line_hints[0]
            for i in line_hints[1:]:
                line_sols[0].append(False)
                line_sols[0] += [True] * i

        return line_sols

    def common_from_perms(self, permutations):
        aux = permutations[0]
        l = len(aux)
        finish = False
        i = 0
        while i < len(permutations) and not finish:
            j = 0
            while j < l:
                if not aux[j] is None and permutations[i][j] != aux[j]:
                    aux[j] = None
                j += 1
            finish = list(aux).count(None) == len(aux)
            i += 1
        return aux

def generate_puzzle(size, p):
    bitmap = []
    for i in range(size):
        bitmap.append([])
        for j in range(size):
            bitmap[i].append(random.random() < p)
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

if __name__ == '__main__':
    z = Puzzle(generate_puzzle(20, 0.5))
    z.solve();
