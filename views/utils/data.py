""" Common data utilities """
from typing import List, Union

import pandas as pd  # type: ignore


def resample(
    df: pd.DataFrame,
    cols: List[str],
    share_positives: float,
    share_negatives: float,
    threshold=0,
):
    """ Resample a dataframe with respect to cols

    Resampling is a technique for changing the positive/negative balance
    of a dataframe. Positives are rows where any of the specified cols
    are greater than the threshold. Useful for highly unbalanced
    datasets where positive outcomes are rare.

    """

    # If both shares are 1 just return the unaltered df
    if share_positives == 1 and share_negatives == 1:
        return df

    # Those rows where at least one of the cols are greater than threshold
    df_positives = df[(df[cols] > threshold).max(axis=1)]
    # it's inverse
    df_negatives = df[~(df[cols] > threshold).max(axis=1)]

    len_positives = len(df_positives)
    len_negatives = len(df_negatives)

    n_positives_wanted = int(share_positives * len_positives)
    n_negatives_wanted = int(share_negatives * len_negatives)

    replacement_pos = share_positives > 1
    replacement_neg = share_negatives > 1
    df = pd.concat(
        [
            df_positives.sample(n=n_positives_wanted, replace=replacement_pos),
            df_negatives.sample(n=n_negatives_wanted, replace=replacement_neg),
        ]
    )

    return df


def check_has_multiindex(data: Union[pd.Series, pd.DataFrame]) -> None:
    """ Raise RuntimeError if Series s doesn't have MultiIndex """
    if not isinstance(data.index, pd.MultiIndex):
        msg = (
            "Data is lacking a multiindex that was expected."
            "Set the index with df.set_index([timevar, groupvar])."
        )
        raise RuntimeError(msg)


def balance_panel_last_t(df: pd.DataFrame) -> pd.DataFrame:
    """ Balance a multiindexed dataframe panel.

    The balanced index has observations for all groups present at the
    last t.
    Assumens df is indexed with timevar as index level 0, and groupvar
    at index level 1.

    Args:
        df: Dataframe with multiindex to balance
    Returns:
        df: A reindexed dataframe
    """
    check_has_multiindex(df)

    # Reset the index to actual values,
    # Needed in case data has been subseted with .loc before
    # If this isn't done, df.index.levels[0].max() gets the
    # pre-subsetting max
    df = df.reset_index().set_index(df.index.names).sort_index()

    return df.reindex(
        pd.MultiIndex.from_product(
            [
                df.index.levels[0].unique(),
                df.loc[df.index.levels[0].max()].index.unique(),
            ],
            names=df.index.names,
        )
    ).sort_index()


def assign_into_df(df_to, df_from):
    for col in df_from:
        df_to[col] = df_from[col]
    return df_to
