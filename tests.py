import unittest
from bitmap_image import *

T = 1
F = 0
N = -1

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
bitmapHard = np.array(bitmapHard)


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
bitmap0 = np.array(bitmap0)


bitmap1 = [
    [T, F, T, F, T],
    [F, T, F, F, T],
    [F, T, F, T, T],
    [F, T, F, F, T],
    [T, T, F, F, T],
    [F, T, F, F, T]
]
bitmap1 = np.array(bitmap1)


bitmap2 = [
    [T, F, T, F, T],
    [F, T, F, F, F],
    [F, T, F, F, T]
]
bitmap2 = np.array(bitmap2)


bitmap3 = [
    [T, F, T, F, T],
    [F, T, F, T, F],
    [F, T, F, T, F]
]
bitmap3 = np.array(bitmap3)


class Test1(unittest.TestCase):
    def test_bitmap_generation_size(self):
        bm1 = generate_bitmap([10, 10], 0.9);
        self.assertTrue(bm1.shape[0] == 10)
        self.assertTrue(bm1.shape[1] == 10)

    def test_bitmap_generation_prob(self):
        bm1 = generate_bitmap([1000, 1000], 0.9);
        porc_true = bm1.sum() / 1000**2
        self.assertTrue(porc_true > 0.8)

    def test_puzzle_generation(self):
        bitmaps = [bitmapHard, bitmap3, bitmap2, bitmap1, bitmap0]

        for bitmap in bitmaps:
            p = Puzzle(bitmap)
            self.assertTrue(bitmap.shape[0] == p.height == len(p.hints[0]))
            self.assertTrue(bitmap.shape[1] == p.width == len(p.hints[1]))

    def test_puzzle_verify(self):
        bm1 = [
            [F, F, F, F],
            [T, T, F, F],
            [T, F, T, F],
            [T, T, F, F],
            [F, F, F, F]
        ]
        bm1 = np.array(bm1)

        bm2 = [
            [F, F, F, F],
            [T, T, F, F],
            [T, F, F, T],
            [T, T, F, F],
            [F, F, F, F]
        ]
        bm2 = np.array(bm2)

        bm3 = [
            [F, F, F, F],
            [T, T, F, F],
            [T, F, T, F],
            [T, F, F, F],
            [F, T, F, F]
        ]
        bm3 = np.array(bm3)

        bm4 = [
            [F, F, F, F],
            [F, F, F, F],
            [F, F, F, F],
            [F, F, F, F],
            [F, F, F, F]
        ]
        bm4 = np.array(bm4)

        p = Puzzle(bm1)
        self.assertTrue(p.verify_rows(bm1))
        self.assertTrue(p.verify_cols(bm1))
        self.assertTrue(p.verify_map(bm1))

        self.assertTrue(p.verify_rows(bm2))
        self.assertTrue(p.verify_cols(bm3))

        self.assertFalse(p.verify_map(bm2))
        self.assertFalse(p.verify_map(bm3))
        print(bm4)
        self.assertFalse(p.verify_map(bm4))

    def test_puzzle_still_valid(self):
        bm1 = [
            [F, F, F, F, T],
            [T, T, F, F, T],
            [T, F, T, F, T],
            [T, T, F, F, F],
            [F, F, F, F, F],
            [T, T, T, T, T]
        ]
        bm1 = np.array(bm1)

        #same row hints
        bm2 = [
            [F, F, F, F, T],
            [F, T, T, F, T],
            [N, F, F, T, T],
            [N, N, T, F, N],
            [N, N, N, F, N],
            [N, N, N, T, N]
        ]
        bm2 = np.array(bm2)

        #same column hints
        bm3 = [
            [F, F, F, F, N],
            [T, N, N, N, N],
            [T, F, T, F, T],
            [T, T, F, N, N],
            [F, N, N, N, N],
            [T, T, N, N, N]
        ]
        bm3 = np.array(bm3)

        bm4 = [
            [F, T, T, F, N],
            [T, N, N, N, N],
            [T, F, T, F, T],
            [T, T, F, N, N],
            [F, N, N, N, N],
            [T, T, N, N, N]
        ]
        bm4 = np.array(bm4)

        bm5 = [
            [F, F, F, F, T],
            [F, T, T, F, T],
            [N, T, T, T, T],
            [N, N, T, F, N],
            [N, N, N, F, N],
            [N, N, N, T, N]
        ]
        bm5 = np.array(bm5)

        p = Puzzle(bm1)
        #print("\nwith itself")
        self.assertTrue(p.still_works_vert(bm1))
        self.assertTrue(p.still_works_horiz(bm1))

        #print("\nstill works")
        self.assertTrue(p.still_works_vert(bm2))
        self.assertTrue(p.still_works_horiz(bm3))

        #print("\ndoesn't work")
        self.assertFalse(p.still_works_horiz(bm4))
        self.assertFalse(p.still_works_vert(bm5))

    def test_still_works_cases(self):
        line1 = [T, T, T, T, T]
        line2 = [T, T, F, ]


if __name__ == '__main__':
    unittest.main()
