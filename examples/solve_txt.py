import argparse
import numpy as np
import time
from picross_ai import PicrossPuzzle
from picross_ai.solver_functions import *
from picross_ai import display_solution

def run_solver(puzzle, solver = "partial", boosted = False, boost_depth = 100):

    time_spent = 0

    progress = None
    if boosted:
        progress, solvable = partial_solution(puzzle, boost_depth)
    
        if not solvable:
            print("The puzzle cannot be solved.")
            return

    if solver == "backtrack":
        time_start = time.process_time()
        solution = solve_naive_backtrack(puzzle, progress)
        time_end = time.process_time()
        time_spent = time_end - time_start
    
    elif solver == "random_backtrack":
        time_start = time.process_time()
        solution = solve_random_backtrack(puzzle, progress)
        time_end = time.process_time()
        time_spent = time_end - time_start
    
    elif solver == "row_backtrack":
        time_start = time.process_time()
        solution = solve_optimized_backtrack(puzzle, progress)
        time_end = time.process_time()
        time_spent = time_end - time_start
    
    elif solver == "row_backtrack":
        puzzle.transpose()
        if progress is not None:
            progress = progress.T
        
        time_start = time.process_time()
        solution = solve_optimized_backtrack(puzzle, progress)
        time_end = time.process_time()
        time_spent = time_end - time_start
        
        if solution is not None:
            solution = solution.T
        if progress is not None:
            progress = progress.T
        puzzle.transpose()
    
    elif solver == "partial":
        time_start = time.process_time()
        solution, solvable = partial_solution(puzzle, 100)
        time_end = time.process_time()
        time_spent = time_end - time_start

        print(f"Solvable: {solvable}")
    
    elif solver == "best":
        solve(puzzle)
        solution = np.array([[]])

    if solution is not None:
        display_solution(solution)
    
    print(f"Solution verification: {'Ok' if puzzle.verify_solution(solution) else 'Incorrect'}")
    print(f"Hint verification: {'Ok' if puzzle.is_solution(solution) else 'Incorrect'}")
    print(f"Time spent: {time_spent}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", dest="solv", 
                        help="Solver to use [backtrack, backtrack_boosted, row_backtrack, row_backtrack_boosted, partial]")
    
    parser.add_argument("-b", "--boost", dest="boost", action="store_true",
                        help="Use partial solver before backtracking.")
    
    parser.add_argument("-i", "--input", dest="inp",
                        help="Input file that has the hints to a puzzle.")

    args = parser.parse_args()

    solvers = [
        "backtrack",
        "random_backtrack",
        "row_backtrack",
        "col_backtrack",
        "partial",
        "best"
    ]

    solver = "backtrack"
    boosted = args.boost

    if args.solv:
        solver = args.solv
        if solver not in solvers:
            raise RuntimeError(f"Incorrect solver name, use one of the available solvers: [{', '.join(solvers)}]")
        
    if not args.inp:
        raise Exception("Please input an image to use as a puzzle.")
    
    puzzle = PicrossPuzzle.from_txt(args.inp)
    
    run_solver(puzzle, solver, boosted)

if __name__ == "__main__":
    main()


    
