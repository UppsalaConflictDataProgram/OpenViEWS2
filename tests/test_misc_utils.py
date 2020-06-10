""" Test misc utils that don't fit anywhere else """
from views.utils import misc


def test_lists_disjoint() -> None:
    a = [1, 2]
    b = [3, 4]
    c = [5, 6]
    d = [6, 7]  # 6 shared with c
    assert not misc.lists_disjoint([a, b, c, d])
    assert misc.lists_disjoint([a, b, c])
