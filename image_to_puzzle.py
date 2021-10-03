from bitmap_image import *
from PIL import Image

def ask_for_puzzle():
    return get_puzzle(input("Image path:"))

def get_puzzle(path):
    result = None
    try:
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


if __name__ == '__main__':
    z = get_puzzle("magnemite.png")
    z.solve()
