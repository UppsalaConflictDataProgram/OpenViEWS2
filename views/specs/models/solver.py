""" Model specification solver """

import os
from typing import Any, Dict, List, Union

from views.utils import io


def solve_formulas(
    spec: Dict[Any, Any]
) -> Dict[str, Dict[str, Union[List[str], str]]]:
    """ Solve the colsets, themes and formulas from spec """

    def solve_theme(
        name_theme: str, colsets: Dict[str, str], themes: Dict[str, str]
    ) -> List[str]:
        """ Get a dictionary of column-populated themes """

        cols_theme = list()
        refs = themes[name_theme]
        for ref in refs:

            # If the reference is to a colset just get the cols
            if ref in colsets.keys():
                for col in colsets[ref]:
                    if col not in cols_theme:  # avoid dups
                        cols_theme.append(col)

            # Recursive lookup for themes
            elif ref in themes.keys():
                for col in solve_theme(ref, colsets, themes):
                    if col not in cols_theme:  # avoid dups
                        cols_theme.append(col)

            else:
                raise RuntimeError(
                    f"{ref} not found in {colsets.keys()} or {themes.keys()}"
                )

        return sorted(cols_theme)

    def solve_themes_and_colsets(spec: Dict[Any, Any]) -> Dict[str, List[str]]:
        """ Solve the themes by looking up names from colsets or themes """

        solved_themes = dict()
        for name_theme in spec["themes"].keys():
            solved_themes[name_theme] = solve_theme(
                name_theme, spec["colsets"], spec["themes"]
            )
        for name_colset, colset in spec["colsets"].items():
            solved_themes[name_colset] = colset
        return solved_themes

    assert list(spec.keys()) == ["colsets", "themes", "formulas"]

    solved_themes: Dict[str, List[str]] = solve_themes_and_colsets(spec)
    solved_formulas = dict()
    for name_formula, formula in spec["formulas"].items():
        solved_formulas[name_formula] = {
            "col_outcome": formula["col_outcome"],
            "cols_features": sorted(solved_themes[formula["cols_features"]]),
        }
    return solved_formulas


def solved_cm() -> Dict[str, Dict[str, Union[List[str], str]]]:
    """ Get solved CM formulas from cm.yaml """
    spec = io.load_yaml(os.path.join(os.path.dirname(__file__), "cm.yaml"))
    formulas = solve_formulas(spec)
    return formulas


def solved_pgm() -> Dict[str, Dict[str, Union[List[str], str]]]:
    """ Get solved CM formulas from cm.yaml """
    spec = io.load_yaml(os.path.join(os.path.dirname(__file__), "pgm.yaml"))
    formulas = solve_formulas(spec)
    return formulas
