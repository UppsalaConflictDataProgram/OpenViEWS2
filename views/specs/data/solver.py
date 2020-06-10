""" Specification solver for transformations

A user can specify a set of transformations as a dictionary.
make_transforms_ordered() returns a dependency ordered list of
the corresponding Transform() instances.
"""
from typing import Any, Dict, List

from views.apps.data import api


def _get_cols_source(transforms: List[api.Transform]) -> List[str]:
    """ Get a list of source columns needed for a list of Tranforms """

    all_names = [transform.name for transform in transforms]
    all_cols = []
    for transform in transforms:
        for col in transform.cols_input:
            all_cols.append(col)

    # Dedup
    all_cols = sorted(list(set(all_cols)))
    cols_source = [col for col in all_cols if col not in all_names]
    cols_source = sorted(cols_source)

    return cols_source


def _order_transforms(transforms: List[api.Transform]) -> List[api.Transform]:
    """ Order transformations so they are done in dependency order """

    def names(tasks):
        return [task.name for task in tasks]

    ordered: List[api.Transform] = list()
    while transforms:
        progress = False
        for task in transforms:
            # if task has deps in the other transforms that haven't
            # been solved themselves wait
            if any(
                [
                    col in names(transforms) and col not in names(ordered)
                    for col in task.cols_input
                ]
            ):
                pass
            else:
                ordered.append(task)
                transforms.remove(task)
                progress = True
        if not progress:
            raise RuntimeError(
                "No progress, transform spec broken."
                f"Ordered (OK): {ordered}"
                f"Remaining: {transforms}"
            )

    return ordered


def make_transforms_ordered(
    specs: Dict[str, Dict[str, Any]]
) -> List[api.Transform]:
    """ Make dependency ordered list of Transform objects """
    transforms = [api.Transform(name, **spec) for name, spec in specs.items()]
    transforms = _order_transforms(transforms)
    return transforms
