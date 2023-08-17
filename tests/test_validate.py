from picross_ai import *
import pytest

T = 1
F = 0
N = -1

def test_puzzle_verify():
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

    p = PicrossPuzzle.from_bitmap(bm1)
    assert p.verify_rows(bm1)
    assert p.verify_cols(bm1)
    assert p.verify_solution(bm1)

    assert p.verify_rows(bm2)
    assert p.verify_cols(bm3)

    assert not p.verify_solution(bm2)
    assert not p.verify_solution(bm3)
    assert not p.verify_solution(bm4)


def test_puzzle_still_valid():
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

    p = PicrossPuzzle.from_bitmap(bm1)

    assert p.still_works_vert(bm1) # True
    assert p.still_works_horiz(bm1) # True

    assert p.still_works_vert(bm2) # True
    assert p.still_works_horiz(bm3) # True

    assert not p.still_works_horiz(bm4) # False
    assert not p.still_works_vert(bm5) # False