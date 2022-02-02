import numpy as np
from functools import reduce
from discrete_math import *
import random

class Puzzle:
    def __init__(self, bitmap):
        self.hints = [[],[]]
        self.solution = bitmap
        self.height = bitmap.shape[0]
        self.width = bitmap.shape[1]
        self.swapState = False
        h, v = 0, 0
        for i in range(self.height):
            self.hints[0].append([])
            for j in range(self.width):
                if bitmap[i, j] == 1:
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
                if bitmap[j, i] == 1:
                    v += 1
                elif v > 0:
                    self.hints[1][i].append(v)
                    v = 0
            if v > 0:
                self.hints[1][i].append(v)
                v = 0
            if len(self.hints[1][i]) == 0:
                self.hints[1][i].append(0)
        #print(bitmap)
        #print(self.hints)

    def verify_rows(self, map):
        verifiable = True
        i = 0

        while i < self.height and verifiable:
            j_aux = 0
            j_hint = 0
            j = 0
            while j < self.width and verifiable:
                if map[i, j] == 1:
                    j_aux += 1
                elif j_aux > 0:
                    verifiable = j_hint < len(self.hints[0][i]) and j_aux == self.hints[0][i][j_hint]
                    j_aux = 0
                    j_hint += 1
                j += 1

            if j_hint == 0:
                verifiable = j_hint < len(self.hints[0][i]) and j_aux == self.hints[0][i][j_hint]
            i += 1
        return verifiable

    def verify_cols(self, map):
        verifiable = True
        i = 0
        while i < self.width and verifiable:
            j_aux = 0
            j_hint = 0
            j = 0
            while j < self.height and verifiable:
                if map[j, i] == 1:
                    j_aux += 1
                elif j_aux > 0:
                    verifiable = j_hint < len(self.hints[1][i]) and j_aux == self.hints[1][i][j_hint]
                    j_aux = 0
                    j_hint += 1
                j += 1
            if j_hint == 0:
                verifiable = j_hint < len(self.hints[1][i]) and j_aux == self.hints[1][i][j_hint]
            i += 1
        return verifiable

    def verify_map(self, map):
        return self.verify_rows(map) and self.verify_cols(map)

    def still_works_horiz(self, map):
        verifiable = True
        i = 0
        while i < self.height and verifiable:
            j_aux = 0
            j_hint = 0
            last_j = 0
            j = 0
            skip_row = False
            while j < self.width and verifiable and not skip_row:
                if map[i, j] == -1:
                    skip_row = True
                    if j_hint == len(self.hints[0][i]):
                        verifiable = last_j == self.hints[0][i][j_hint-1]
                    elif j_hint < len(self.hints[0][i]):
                        verifiable = j_aux <= self.hints[0][i][j_hint]
                    else:
                        verifiable = False
                else:
                    if map[i, j]:
                        j_aux += 1
                    elif j_aux > 0:
                        verifiable = j_hint < len(self.hints[0][i]) and j_aux <= self.hints[0][i][j_hint]
                        last_j = j_aux
                        j_aux = 0
                        j_hint += 1
                j += 1
            i += 1
        return verifiable

    def still_works_vert(self, map):
        verifiable = True
        i = 0
        while i < self.width and verifiable:
            j_aux = 0
            j_hint = 0
            last_j = 0
            j = 0
            skip_row = False
            while j < self.height and verifiable and not skip_row:
                if map[j, i] == -1:
                    skip_row = True
                    if j_hint == len(self.hints[1][i]):
                        verifiable = last_j == self.hints[1][i][j_hint-1]
                    elif j_hint < len(self.hints[1][i]):
                        verifiable = j_aux <= self.hints[1][i][j_hint]
                    else:
                        verifiable = False
                else:
                    if map[j, i]:
                        j_aux += 1
                    elif j_aux > 0:
                        verifiable = j_hint < len(self.hints[1][i]) and j_aux <= self.hints[1][i][j_hint]
                        last_j = j_aux
                        j_aux = 0
                        j_hint += 1
                j += 1
            i += 1
        return verifiable

    def solve_naive_backtrack(self):
        base_map = np.zeros([self.height, self.width], dtype=np.byte)
        solved_map = [-np.ones([self.height, self.width], dtype=np.byte)]
        self.__naive_backtrack(base_map, solved_map, base_map.shape[0], base_map.shape[1])
        if solved_map[0][0, 0] == -1:
            print()
            print("No solution found.")
            solved_map[0] = None
        else:
            print()
            print("Solution found.")

        return solved_map[0]

    def __naive_backtrack(self, bitmap, solved_map, max_y, max_x, y = 0, x = 0):
        if solved_map[0][0, 0] == -1:
            if self.verify_map(bitmap):
                print("we are done", self.verify_map(bitmap))
                print(bitmap)
                solved_map[0] = np.copy(bitmap)
            elif y < max_y:
                aux = bitmap[y, x]
                for i in [True, False]:
                    bitmap[y, x] = i
                    self.__naive_backtrack(bitmap, solved_map, max_y, max_x, y + ((x + 1) // max_x), (x + 1) % max_x)
                bitmap[y, x] = aux


    #idea, fusion, get common squares for all combinations of each row and fill the gaps with square by square backtracking
    def solve_perm_backtrack(self, progress = None):
        base_map = np.zeros([self.height, self.width], dtype=np.byte)
        solved_map = [-np.ones([self.height, self.width], dtype=np.byte)]

        if progress is None:
            progress = -np.ones([self.height, self.width], dtype=np.byte)

        self.__perm_backtrack(base_map, solved_map, progress, len(base_map))

        if solved_map[0][0, 0] == -1:
            print()
            print("No solution found.")
            solved_map[0] = None
        else:
            print()
            print("Solution found.")

        return solved_map[0]

    def __perm_backtrack(self, bitmap, solved_map, progress, max_row, row = 0):
        if solved_map[0][0, 0] == -1:
            if self.verify_map(bitmap):
                solved_map[0] = np.copy(bitmap)
            elif row < max_row and self.still_works_vert(bitmap):
                aux = np.copy(bitmap[row, :])
                for j in self.line_perms(self.hints[0][row], progress[row, :], isHoriz = True):
                    bitmap[row, :] = j
                    self.__perm_backtrack(bitmap, solved_map, progress, max_row, row+1)
                bitmap[row, :] = aux

    def solve(self):
        self.display_solution()

        # Preprocess the puzzle to solve it parcialy
        progress = self.preprocess(100)
        self.display_solution(progress)

        if self.swapState:
            self.swap_axis()
            progress = progress.T

        print("partial solution")
        self.display_solution(progress)



        # Decide if swaping the puzzle will result in a simpler problem
        horiz_cost = 1.0
        for i in range(len(self.hints[0])):
            horiz_cost *= len(self.line_perms(self.hints[0][i], progress[:, i], isHoriz = True))

        progress_t = progress.T
        vert_cost = 1.0
        for i in range(len(self.hints[1])):
            vert_cost *= len(self.line_perms(self.hints[1][i], progress_t[:, i], isHoriz = False))

        print("horiz: {:e}".format(horiz_cost), "vert: {:e}".format(vert_cost))

        if(vert_cost < horiz_cost):
            self.swap_axis()
            progress = progress.T

        # Solve the puzzle
        solution = self.solve_perm_backtrack();
        #solution = self.solve_naive_backtrack();

        # Restore the puzzle
        if(vert_cost < horiz_cost):
            self.swap_axis()
            solution = solution.T

        # Display the puzzle
        self.display_solution(solution)

    def preprocess(self, times = 1, progress = None):
        self.swap_axis()

        #print("prerpocessing nº", times)

        pre_solution = np.zeros([self.width, self.height])

        for i in range(self.height):
            print(i)
            if progress is None:
                #print("permutations", self.line_perms(self.hints[0][i]))
                pre_solution[:,i] = self.common_from_perms(self.line_perms(self.hints[0][i]))
            else:
                #print("permutations", self.line_perms(self.hints[0][i], progress[:,i]))
                pre_solution[:,i] = self.common_from_perms(self.line_perms(self.hints[0][i], progress[:,i]))

        if times > 1 and not np.all(progress == pre_solution) and not self.verify_map(pre_solution.T):
            print("partial solution:")
            if times%2 == 0:
                self.display_solution(pre_solution)
            else:
                self.display_solution(pre_solution.T)
            pre_solution = self.preprocess(times-1, pre_solution.T)

        return pre_solution


    def swap_axis(self):
        self.swapState = not self.swapState
        self.hints[0], self.hints[1] = self.hints[1], self.hints[0]
        self.height, self.width = self.width, self.height

    def display_solution(self, solution = None):
        if solution is None:
            print("preview of the solution")
            solution = self.solution

        display_map = {-1:"_ ", 0:"□ ", 1:"■ "}

        for i in solution:
            for j in i:
                print(display_map[j], end="")
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

    def line_perms(self, line_hints, progress = None, isHoriz = True):
        m = self.width if isHoriz else self.height
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

    def common_from_perms(self, permutations):
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

def generate_bitmap(size, p):
    return (np.random.uniform(0, 1, size) < p).astype(np.byte)

def generate_puzzle(size, p):
    return Puzzle(generate_bitmap(size, p))


if __name__ == '__main__':
    z = generate_puzzle([5, 2], 0.5)
    z.solve();
