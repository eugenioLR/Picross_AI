from picross_ai import *
import pytest

def test_list_replace():
    pass

@pytest.mark.parametrize(
    "n, parts",
    [
        (1, [(1,)]),
        (2, [(2,), (1,1)]),
        (3, [(3,), (2,1), (1,2), (1,1,1)]),
        (4, [(4,), (3,1), (1,3), (2,2), (2,1,1), (1,2,1), (1,1,2), (1,1,1,1)]),
        (5, [(5,), (4,1), (1,4), (3,2), (2,3), (3,1,1), (1,3,1), (1,1,3), (2,2,1),
             (2,1,2), (1,2,2), (2,1,1,1), (1,2,1,1), (1,1,2,1), (1,1,1,2), (1,1,1,1,1)])
    ]
)
def test_partitions(n, parts):
    tup_parts = [tuple(i) for i in partitions(n)]
    assert set(tup_parts) == set(parts)


@pytest.mark.parametrize(
    "n, parts, count",
    [
        (1, [(1,)], 1),
        (2, [(1,1)], 2),
        (4, [(3,1), (1,3), (2,2)], 2),
        (4, [(2,1,1), (1,2,1), (1,1,2)], 3),
        (5, [(4,1), (1,4), (3,2), (2,3)], 2),
        (5, [(3,1,1), (1,3,1), (1,1,3), (2,2,1), (2,1,2), (1,2,2)], 3)
    ]
)
def test_partitions_fixed(n, parts, count):
    tup_parts = [tuple(i) for i in partitions_fixed_count(n, count)]
    assert set(tup_parts) == set(parts)

@pytest.mark.parametrize(
    "n, parts, min_elem, max_elem",
    [
        (1, [(1,)], 0, 1),
        (2, [(2,), (1,1)], 0, 2),
        (3, [(2,1), (1,2), (1,1,1)], 2, 3),
        (4, [(3,1), (1,3), (2,2), (2,1,1), (1,2,1), (1,1,2), (1,1,1,1)], 2, 4),
        (4, [(3,1), (1,3), (2,2), (2,1,1), (1,2,1), (1,1,2)], 2, 3),
        (4, [(4,), (3,1), (1,3), (2,2), (2,1,1), (1,2,1), (1,1,2)], 1, 3),
        (5, [(4,1), (1,4), (3,2), (2,3), (3,1,1), (1,3,1), (1,1,3), (2,2,1),
             (2,1,2), (1,2,2), (2,1,1,1), (1,2,1,1), (1,1,2,1), (1,1,1,2), (1,1,1,1,1)], 2, 5),
        (5, [(4,1), (1,4), (3,2), (2,3), (3,1,1), (1,3,1), (1,1,3), (2,2,1),
             (2,1,2), (1,2,2), (2,1,1,1), (1,2,1,1), (1,1,2,1), (1,1,1,2)], 2, 4),
        (5, [(3,1,1), (1,3,1), (1,1,3), (2,2,1), (2,1,2), (1,2,2), (2,1,1,1), (1,2,1,1),
             (1,1,2,1), (1,1,1,2)], 3, 4)
    ]
)
def test_partitions_limited(n, parts, min_elem, max_elem):
    tup_parts = [tuple(i) for i in partitions_limited_count(n, min_elem, max_elem)]
    assert set(tup_parts) == set(parts)

@pytest.mark.parametrize(
    "n_len, hints, possibilities",
    [
        (1, (1,), [(1,)]),
        (2, (1,), [(0,1), (1,0)]),
        (5, (1,), [(0,0,0,0,1), (0,0,0,1,0), (0,0,1,0,0), (0,1,0,0,0), (1,0,0,0,0)]),
        (5, (1,2), [(1,0,1,1,0), (1,0,0,1,1), (0,1,0,1,1)]),
        (5, (1,1,1), [(1,0,1,0,1)]),
        (5, (5,), [(1,1,1,1,1)]),
        (9, (2,4,1), [(1,1,0,1,1,1,1,0,1)]),
        (9, (2,1,1,1), [(1,1,0,1,0,1,0,1,0), (1,1,0,1,0,1,0,0,1), (1,1,0,1,0,0,1,0,1), (1,1,0,0,1,0,1,0,1), (0,1,1,0,1,0,1,0,1)]),
        (9, (0,), [(0,0,0,0,0,0,0,0,0)])
    ]
)
def test_line_perms(n_len, hints, possibilities):
    tup_parts = [tuple(i) for i in line_perms(n_len, n_len, hints)]
    print(tup_parts)
    assert set(tup_parts) == set(possibilities)
