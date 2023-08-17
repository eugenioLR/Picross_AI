# from Picross_from_image import *
# from Picross_solver import *
# from os.path import exists
# import os
# import numpy as np

# def ask_for_puzzle():
#     puzzle = None
#     path = input("Image path:")
#     if os.path.exists(path):
#         extension = path.split(".")[-1]
#         if extension in ["png", "jpg"]:
#             puzzle = Picross_from_image(read_bitmap(path))
#         else:
#             puzzle = Picross_solver(hints_from_file(path))

#         puzzle.solve()
#     else:
#         print("The file doesn't exists")
#     return puzzle



# if __name__ == '__main__':
#     z = ask_for_puzzle()
