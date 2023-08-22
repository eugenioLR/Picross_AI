from __future__ import annotations
import numpy as np
from functools import reduce
from .permutation_utils import *
from PIL import Image, ImageOps
import random

class PicrossPuzzle:
    """
    Class that holds the hints of a picross (or nonogram) puzzle.

    It also provides functions to verify solutions.
    """

    def __init__(self, hints: List[List[int]]):
        self.hints = hints
        self.height = len(hints[0])
        self.width = len(hints[1])
        self.transposed = False
    
    def transpose(self):
        """
        Transposes the puzzle, swaps vertical hints and horizontal hints.
        """
        
        self.transposed = not self.transposed
        self.hints[0], self.hints[1] = self.hints[1], self.hints[0]
        self.height, self.width = self.width, self.height

    def verify_rows(self, solution: np.npndarray) -> bool:
        """
        Checks if the row restrictions of the puzzle are satisfied.
        """

        verifiable = True

        row_to_str_map = {-1: "u", 0: " ", 1: "b"}
        
        for i, row in enumerate(solution):
            row_str = "".join([row_to_str_map[j] for j in row])
            row_pieces = row_str.split(" ")
            longs = [len(j) for j in row_pieces if len(j) != 0]
            if longs == []:
                longs = [0]
            verifiable = verifiable and longs == self.hints[0][i]
        
        return verifiable

    def verify_cols(self, solution: np.ndarray) -> bool:
        """
        Checks if the column restrictions of the puzzle are satisfied.
        """

        verifiable = True

        col_to_str_map = {-1: "u", 0: " ", 1: "b"}

        for i, col in enumerate(solution.T):
            col_str = "".join([col_to_str_map[j] for j in col])
            col_pieces = col_str.split(" ")
            longs = [len(j) for j in col_pieces if len(j) != 0]
            if longs == []:
                longs = [0]
            verifiable = verifiable and longs == self.hints[1][i]
        
        return verifiable

    def verify_solution(self, solution: np.ndarray) -> bool:
        """
        Checks if all the restrictions of the puzzle are satisfied.
        """

        return solution is not None \
               and np.all(solution != -1) \
               and self.verify_rows(solution) \
               and self.verify_cols(solution)
    
    def is_solution(self, solution: np.ndarray) -> bool:
        return solution is not None and PicrossPuzzle.from_bitmap(solution).hints == self.hints

    def still_works_horiz(self, solution: np.ndarray) -> bool:
        """
        Checks if the rows could satisfy the restrictions if the empty cells were filled.
        """

        verifiable = True
        i = 0
        while i < self.height and verifiable:
            j_aux = 0
            j_hint = 0
            last_j = 0
            j = 0
            skip_row = False
            while j < self.width and verifiable and not skip_row:
                if solution[i, j] == -1:
                    skip_row = True
                    if j_hint == len(self.hints[0][i]):
                        verifiable = last_j == self.hints[0][i][j_hint-1]
                    elif j_hint < len(self.hints[0][i]):
                        verifiable = j_aux <= self.hints[0][i][j_hint]
                    else:
                        verifiable = False
                else:
                    if solution[i, j]:
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

    def still_works_vert(self, solution: np.ndarray) -> bool:
        """
        Checks if the column could satisfy the restrictions if the empty cells were filled.
        """

        verifiable = True
        i = 0
        while i < self.width and verifiable:
            j_aux = 0
            j_hint = 0
            last_j = 0
            j = 0
            skip_row = False
            while j < self.height and verifiable and not skip_row:
                if solution[j, i] == -1:
                    skip_row = True
                    if j_hint == len(self.hints[1][i]):
                        verifiable = last_j == self.hints[1][i][j_hint-1]
                    elif j_hint < len(self.hints[1][i]):
                        verifiable = j_aux <= self.hints[1][i][j_hint]
                    else:
                        verifiable = False
                else:
                    if solution[j, i]:
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
    
    @staticmethod
    def from_bitmap(bitmap: np.ndarray) -> PicrossPuzzle:
        """
        Creates a puzzle from a bitmap or array of boolean values.
        """

        (height, width) = bitmap.shape
        hints = [[],[]]

        # Construct horizontal hints
        for i in range(height):
            h = 0

            hints[0].append([])
            for j in range(width):
                if bitmap[i, j] == 1:
                    h += 1
                elif h > 0:
                    hints[0][i].append(h)
                    h = 0
            
            if h > 0:
                hints[0][i].append(h)
                
            
            if not hints[0][i]:
                hints[0][i].append(0)
        
        # Construct vertical hints
        for i in range(width):
            v = 0

            hints[1].append([])
            for j in range(height):
                if bitmap[j, i] == 1:
                    v += 1
                elif v > 0:
                    hints[1][i].append(v)
                    v = 0
            
            if v > 0:
                hints[1][i].append(v)
            
            if not hints[1][i]:
                hints[1][i].append(0)

        return PicrossPuzzle(hints)
    
    @staticmethod
    def from_image(path: str) -> PicrossPuzzle:
        """
        Creates a puzzle from a black and white image.
        """

        im = ImageOps.grayscale(Image.open(path))
        bitmap = np.asarray(im) != 0
        puzzle_grid = np.invert(bitmap).astype(int)
        return PicrossPuzzle.from_bitmap(puzzle_grid)
    
    @staticmethod
    def from_txt(file_path: str) -> PicrossPuzzle:
        """
        Creates a puzzle from a text file.

        The format of the file looks like this:
        1,1,1;3;3;3,1;1,2
        2,1;1,2;4;1,1;1,3

        Which represents the following puzzle:
        x _ x _ x 
        x x x _ _ 
        _ _ x x x
        x x x _ x
        _ x _ x x 

        The first line indicates the horizontal hints and the second one indicates the vertical
        hints.

        The hint for each row/column is divided by semicolons (;) and each hint has a collection of
        lengths separated by commas (,).
        """

        hints = []

        with open(file_path, "r") as hint_file:
            file_contents = hint_file.read().replace(" ", "")

            if file_contents[-1] == "\n":
                file_contents = file_contents[:-1]

            for row in file_contents.split("\n"):
                hints.append([])
                for elem in row.split(";"):
                    hint = [int(i) for i in elem.split(",")]
                    hints[-1].append(hint)
        
        return PicrossPuzzle(hints)

display_map = {-1:"_", 0:"◼", 1:"◻"}
# display_map = {-1:"_", 0:"◻", 1:"◼"}
# display_map = {-1:"?", 0:"_", 1:"x"}
# display_map = {-1:"_", 0:"·", 1:"#"}
# display_map = {-1:"?", 0:"·", 1:"#"}

def list_lpad(lst, n, fill_val):
    return ([fill_val]*n + lst)[-n:]

def display_puzzle_solution(puzzle: PicrossPuzzle, solution: np.ndarray):
    """
    Displays a solution to a puzzle along with its hints.

    This won't work for very large puzzles or for hints with values higher than 99.
    """

    h_hints = puzzle.hints[0]
    h_hints_str = [",".join([str(i) for i in h_hint]) for h_hint in h_hints]
    max_str_len = max([len(s) for s in h_hints_str])
    h_hints_str = [s.rjust(2 + max_str_len) for s in h_hints_str]

    # Column hints
    v_hints = puzzle.hints[1]
    max_hint_len = max([len(v_hint) for v_hint in v_hints])
    v_hints_aux = [list_lpad(i, max_hint_len, 0) for i in v_hints]
    
    for i in range(max_hint_len):
        print(" "*(4+max_str_len), end="")
        for v_hint in v_hints_aux:
            if v_hint[i] == 0:
                print("  ", end="")
            else:
                print(str(v_hint[i]).rjust(2), end="")
        print()
    print()
    
    # Row hints and solution matrix
    for r_idx, row in enumerate(solution):
        print(h_hints_str[r_idx], end="   ")
        for cell in row:
            print(display_map[cell], end=" ")
        print()

def display_solution(solution: np.ndarray):
    """
    Displays the matrix representing a nonogram solution.
    """

    for row in solution:
        for cell in row:
            print(display_map[cell], end=" ")
        print()
