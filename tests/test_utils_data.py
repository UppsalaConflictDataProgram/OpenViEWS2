import pytest  # type: ignore
import pandas as pd  # type: ignore

from views.utils.mocker import DfMocker
from views.utils import data


def test_assign_into_df() -> None:

    df_a = DfMocker(n_t=20).df
    df_b = df_a.copy()
    df_into = df_a.loc[:, []].copy()

    # Test we get the full frame if we give all times
    df_into = data.assign_into_df(df_to=df_into, df_from=df_a.loc[0:9])
    df_into = data.assign_into_df(df_to=df_into, df_from=df_a.loc[10:19])
    pd.testing.assert_frame_equal(df_a, df_into, check_dtype=False)

    # Test we get missing if we don't give all cols
    df_into = df_a.loc[:, []].copy()
    df_into = data.assign_into_df(df_to=df_into, df_from=df_a.loc[0:3])
    df_into = data.assign_into_df(df_to=df_into, df_from=df_a.loc[10:19])
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df_a, df_into, check_dtype=False)
