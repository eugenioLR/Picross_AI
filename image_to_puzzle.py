from bitmap_image import *
from PIL import Image, ImageOps
import numpy as np

def ask_for_puzzle():
    return get_puzzle(input("Image path:"))

def get_puzzle(path):
    result = None
    im = ImageOps.grayscale(Image.open(path))
    bitmap = (np.asarray(im) == 0).astype(np.byte)
    result = Puzzle(bitmap)
    return result


if __name__ == '__main__':
    #z = get_puzzle("magnemite.png")
    z = ask_for_puzzle()
    z.solve()
