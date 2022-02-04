import numpy as np
from functools import reduce
from permutation_utils import *
import random

class Puzzle:
    def __init__(self, hints):
        self.hints = hints
        self.height = len(hints[0])
        self.width = len(hints[1])
        self.swapState = False
        print(self.hints[0])
        print(self.hints[1])
        print(self.height)
        print(self.width)

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
        return np.all(map != -1) and self.verify_rows(map) and self.verify_cols(map)

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
                if j_hint == 0:
                    verifiable = j_hint < len(self.hints[0][i]) and j_aux <= self.hints[0][i][j_hint]
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
                if j_hint == 0:
                    verifiable = j_hint < len(self.hints[1][i]) and j_aux <= self.hints[1][i][j_hint]
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
                for j in line_perms(self.width, self.height, self.hints[0][row], progress[row, :], isHoriz = True):
                    bitmap[row, :] = j
                    self.__perm_backtrack(bitmap, solved_map, progress, max_row, row+1)
                bitmap[row, :] = aux

    def solve_completion_backtrack(self, progress = None):
        base_map = np.zeros([self.height, self.width], dtype=np.byte)
        solved_map = [-np.ones([self.height, self.width], dtype=np.byte)]

        if progress is None:
            progress = -np.ones([self.height, self.width], dtype=np.byte)

        if self.verify_map(progress):
            print("There's no need for further inspection")
            solved_map[0] = progress
        else:
            self.__completion_backtrack(base_map, solved_map, progress, base_map.shape[0], base_map.shape[1])

        if solved_map[0][0, 0] == -1:
            print()
            print("No solution found.")
            solved_map[0] = None
        else:
            print()
            print("Solution found.")

        return solved_map[0]

    def __completion_backtrack(self, bitmap, solved_map, progress, max_y, max_x, y = 0, x = 0):
        if solved_map[0][0, 0] == -1:
            if self.verify_map(bitmap):
                solved_map[0] = np.copy(bitmap)
            elif y < max_y:
                aux = bitmap[y, x]
                if progress[y, x] == -1:
                    for i in [True, False]:
                        bitmap[y, x] = i

                        (new_bitmap, solvable) = self.partial_solution(100, bitmap)

                        if solvable:
                            self.__completion_backtrack(new_bitmap, solved_map, progress, max_y, max_x, y + ((x + 1) // max_x), (x + 1) % max_x)
                        else:
                            self.__completion_backtrack(bitmap, solved_map, progress, max_y, max_x, y + ((x + 1) // max_x), (x + 1) % max_x)
                else:
                    bitmap[y, x] = progress[y, x]
                    self.__completion_backtrack(bitmap, solved_map, progress, max_y, max_x, y + ((x + 1) // max_x), (x + 1) % max_x)

                bitmap[y, x] = aux

    def solve_perm_complete_backtrack(self, progress = None):
        base_map = np.zeros([self.height, self.width], dtype=np.byte)
        solved_map = [-np.ones([self.height, self.width], dtype=np.byte)]

        if progress is None:
            progress = -np.ones([self.height, self.width], dtype=np.byte)


        self.__perm_complete_backtrack(base_map, solved_map, progress, len(base_map))

        if solved_map[0][0, 0] == -1:
            print()
            print("No solution found.")
            solved_map[0] = None
        else:
            print()
            print("Solution found.")

        return solved_map[0]

    def __perm_complete_backtrack(self, bitmap, solved_map, progress, max_row, row = 0):
        if solved_map[0][0, 0] == -1:
            if self.verify_map(bitmap):
                solved_map[0] = np.copy(bitmap)
            elif row < max_row and self.still_works_vert(bitmap):
                aux = np.copy(bitmap[row, :])
                if np.any(progress[row, :] != 1):
                    for j in line_perms(self.width, self.height, self.hints[0][row], progress[row, :], isHoriz = True):
                        bitmap[row, :] = j
                        (new_bitmap, solvable) = self.partial_solution(5, bitmap)
                        if solvable:
                            self.__perm_complete_backtrack(new_bitmap, solved_map, progress, max_row, row+1)
                        else:
                            self.__perm_complete_backtrack(bitmap, solved_map, progress, max_row, row+1)
                else:
                    bitmap[row, :] = progress[row, :]
                    self.__perm_complete_backtrack(bitmap, solved_map, progress, max_row, row+1)
                bitmap[row, :] = aux

    def solve(self):

        # Preprocess the puzzle to solve it parcialy
        progress = self.partial_solution(100)[0]

        print("partial solution")
        self.display_solution(progress)

        # Decide if swaping the puzzle will result in a simpler problem
        horiz_cost = 1.0
        for i in range(len(self.hints[0])):
            horiz_cost *= len(line_perms(self.width, self.height,self.hints[0][i], progress[i, :], isHoriz = True))

        progress_t = progress.T
        vert_cost = 1.0
        for i in range(len(self.hints[1])):
            vert_cost *= len(line_perms(self.width, self.height,self.hints[1][i], progress_t[i, :], isHoriz = False))

        print("horiz: {:}".format(round(horiz_cost)), "vert: {:}".format(round(vert_cost)))

        #solution = self.solve_perm_complete_backtrack();
        #solution = self.solve_completion_backtrack();
        #solution = self.solve_perm_backtrack();
        #solution = self.solve_naive_backtrack();

        if vert_cost < horiz_cost:
            self.swap_axis()
            progress = progress.T

        # Solve the puzzle
        solution = self.solve_perm_complete_backtrack(progress);
        #solution = self.solve_completion_backtrack(progress);
        #solution = self.solve_perm_backtrack(progress);
        #solution = self.solve_naive_backtrack();

        # Restore the puzzle
        if vert_cost < horiz_cost:
            self.swap_axis()
            progress = progress.T
            if solution is not None:
                solution = solution.T

        # Display the puzzle
        if solution is None:
            self.display_solution(progress)
        else:
            self.display_solution(solution)

    def partial_solution(self, times = 1, progress = None, solvable = True):
        pre_solution = np.zeros([self.height, self.width])

        for i in range(self.height):
            if progress is None:
                permutations = line_perms(self.width, self.height, self.hints[0][i])
            else:
                permutations = line_perms(self.width, self.height, self.hints[0][i], progress[i,:])

            solvable = len(permutations) > 0
            if solvable:
                pre_solution[i,:] = common_from_perms(permutations)
            #else:
            #    print("this combination is impossible")

        if solvable and times > 1 and not np.all(progress == pre_solution) and not self.verify_map(pre_solution):
            self.swap_axis()
            pre_solution = self.partial_solution(times-1, pre_solution.T, solvable)[0].T
            self.swap_axis()

        return (pre_solution, solvable)


    def swap_axis(self):
        self.swapState = not self.swapState
        self.hints[0], self.hints[1] = self.hints[1], self.hints[0]
        self.height, self.width = self.width, self.height

    def display_solution(self, sol):
        display_map = {-1:"_ ", 0:"■ ", 1:"□ "}

        for i in sol:
            for j in i:
                print(display_map[j], end="")
            print()


def generate_bitmap(size, p):
    return (np.random.uniform(0, 1, size) < p).astype(np.byte)

def generate_puzzle(size, p):
    return Puzzle(generate_bitmap(size, p))

def get_from_file(file_path):
    result = []
    with open(file_path, "r") as hint_file:
        file_contents = hint_file.read().replace(" ", "")[:-1]
        for row in file_contents.split("\n"):
            result.append([])
            for elem in row.split(";"):
                hint = [int(i) for i in elem.split(",")]
                result[-1].append(hint)
    return result


if __name__ == '__main__':
    #z = generate_puzzle([15, 10], 0.5)
    hints = get_from_file("txtTests/test5.txt")
    z = Puzzle(hints)
    z.solve();
