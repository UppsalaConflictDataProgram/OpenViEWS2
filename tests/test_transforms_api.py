import pandas as pd  # type: ignore
import pytest  # type: ignore
from views.apps.data import api
from views.apps.transforms import lib
from views.utils import mocker


def test_col_cols_ok() -> None:

    # These are ok
    t = api.Transform(name="testname", f="rollmax", cols=["a", "b"])
    t = api.Transform(name="testname", f="rollmax", col="a")


def test_col_col_not_ok() -> None:
    # Not OK, col is list
    with pytest.raises(TypeError) as exc:
        t = api.Transform(name="testname", f="rollmax", col=["a", "b"])
        assert "col should be string" in str(exc.value)


def test_col_cols_not_ok() -> None:
    # Not OK, cols is str
    with pytest.raises(TypeError) as exc:
        t = api.Transform(name="testname", f="rollmax", cols="a")
        assert "col should be string" in str(exc.value)


def test_f_unknown() -> None:
    with pytest.raises(KeyError) as exc:
        t = api.Transform(name="testname", f="unknown", col="a")
        assert "following values of f are recognised:" in str(exc.value)


def test_f_missing() -> None:
    with pytest.raises(KeyError) as exc:
        t = api.Transform(name="testname", col="a")
        assert "Transformer needs a 'f' field" in str(exc.value)


def test_compute() -> None:
    t = api.Transform(name="testname", f="tlag", col="b_a", time=1)
    df = mocker.DfMocker().df

    s_t = t.compute(df)
    s_raw = lib.tlag(s=df["b_a"], time=1)
    pd.testing.assert_series_equal(s_t, s_raw)
