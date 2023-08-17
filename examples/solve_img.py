import argparse
import numpy as np
import time
from picross_ai import PicrossPuzzle
from picross_ai import *

def run_solver(puzzle, solver = "partial", boosted = False, transposed = False, boost_depth = 100):
    time_spent = 0

    progress = None
    if boosted:
        progress, solvable = iterated_intersections(puzzle, boost_depth)
        if not solvable:
            print("The puzzle cannot be solved.")
            return

    if transposed:
        puzzle.transpose()
        if progress is not None:
            progress = progress.T
    
    if solver == "backtrack":
        time_start = time.process_time()
        solution = solve_naive_backtrack(puzzle, progress, verbose=True)
        time_end = time.process_time()
        time_spent = time_end - time_start
    
    elif solver == "random_backtrack":
        time_start = time.process_time()
        solution = solve_random_backtrack(puzzle, progress, verbose=True)
        time_end = time.process_time()
        time_spent = time_end - time_start
    
    elif solver == "row_backtrack":
        time_start = time.process_time()
        solution = solve_row_backtrack(puzzle, progress, verbose=True)
        time_end = time.process_time()
        time_spent = time_end - time_start
    
    elif solver == "partial":
        time_start = time.process_time()
        solution, solvable = iterated_intersections(puzzle, 100)
        time_end = time.process_time()
        time_spent = time_end - time_start

        print(f"Solvable: {solvable}")

    if transposed:
        puzzle.transpose()
        solution = solution.T
    
    display_solution(solution)
    
    print(f"Solution verification: {'Ok' if puzzle.verify_solution(solution) else 'Incorrect'}")
    print(f"Hint verification: {'Ok' if puzzle.is_solution(solution) else 'Incorrect'}")
    print(f"Time spent: {time_spent}")

def main():
    solvers = [
        "backtrack",
        "random_backtrack",
        "row_backtrack",
        "partial"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", dest="solv", 
                        help=f"Solver to use [{', '.join(solvers)}]")
    
    parser.add_argument("-b", "--boost", dest="boost", action="store_true",
                        help="Use partial solver before backtracking.")
    
    parser.add_argument("-t", "--transpose", dest="trans", action="store_true",
                        help="Transposes the puzzle before solving it.")
    
    parser.add_argument("-i", "--inp", dest="inp", 
                        help="Image to encode as a puzzle.")

    args = parser.parse_args()

    solver = "backtrack"
    boosted = args.boost
    transposed = args.trans

    if args.solv:
        solver = args.solv
        if solver not in solvers:
            raise RuntimeError(f"Incorrect solver name, use one of the available solvers: [{', '.join(solvers)}]")
        
    if not args.inp:
        raise Exception("Please input an image to use as a puzzle.")
    
    puzzle = PicrossPuzzle.from_image(args.inp)
    
    run_solver(puzzle, solver, boosted, transposed)

if __name__ == "__main__":
    main()


    
