""" Misc utils that don't fit anyhwere else """
from typing import Any, List


def lists_disjoint(lists: List[List[Any]]) -> bool:
    """ Do lists share any elements"""
    disjoint = True
    for i, base_list in enumerate(lists):
        lists_to_check = lists[i + 1 :]
        for to_check in lists_to_check:
            if not set(base_list).isdisjoint(to_check):
                disjoint = False
    return disjoint
