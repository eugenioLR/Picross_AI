from bitmap_image import *
from PIL import Image

def ask_for_puzzle():
    path = ""
    result = None
    try:
        path = input("Image path:")
        im = Image.open(path)
        px = im.load()
        bitmap = []
        for i in range(im.size[1]):
            bitmap.append([])
            for j in range(im.size[0]):
                bitmap[i].append(px[j,i][0] == 0)
        result = Puzzle(bitmap)
    except:
        print("does not exist, sorry")
    return result
