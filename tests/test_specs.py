""" Test the specs interface """
import pytest  # type: ignore
import yaml
from views.specs import models

SPEC_TEST = yaml.safe_load(
    """
colsets:
    colset_z:
        - zeus
    colset_a:
        - asda
        - bobo
    colset_b:
        - bertil
        - cesar
    colset_c:
        - cesar
        - david
    colset_steve:
        - steven
        - dave
themes:
    theme_a:
        - colset_a
        - colset_b
    theme_b:
        - colset_b
        - colset_c
    theme_nested:
        - theme_a
        - theme_b
    theme_supernested:
        - colset_steve
        - theme_nested

formulas:
    o:
        col_outcome: asda
        cols_features: theme_supernested
"""
)

SPEC_TEST_BROKEN = yaml.safe_load(
    """
colsets:
    colset_a:
        - asda
        - bobo
    colset_b:
        - bertil
        - cesar
themes:
    theme_a:
        - colset_a
        - colset_b
        - missing_key

formulas:
    o:
        col_outcome: asda
        cols_features: theme_supernested
"""
)


def test_spec_models():
    assert isinstance(models.cm, dict)
    # assert isinstance(models.solver.solved_cm(), dict)


def test_solver():
    """ Test that solver solves properly """
    filled_formulas = models.solver.solve_formulas(SPEC_TEST)

    wanted = ["steven", "dave", "asda", "bobo", "bertil", "cesar", "david"]
    assert filled_formulas["o"]["cols_features"] == sorted(wanted)

    with pytest.raises(RuntimeError) as excinfo:

        _ = models.solver.solve_formulas(SPEC_TEST_BROKEN)
        assert "No match for missing_key in" in str(excinfo.value)
