""" Missing data filling functionality """
from typing import Any, List
import logging
import multiprocessing as mp

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from sklearn.experimental import enable_iterative_imputer  # type: ignore # noqa: F401, E501 # pylint: disable=unused-import
from sklearn.impute import IterativeImputer  # type: ignore
from sklearn.linear_model import BayesianRidge  # type: ignore

from views.utils import data
from views.utils.log import logtime

log = logging.getLogger(__name__)


def list_totally_missing(df: pd.DataFrame) -> List[str]:
    """ Get a list of columns for which all values are missing """
    log.debug(f"Checking {len(df.columns)} cols for complete missingness.")
    cols = []
    for col in df:
        if df[col].isnull().mean() == 1.0:
            cols.append(col)
            log.debug(f"{col} completely missing")
    return cols


@logtime
def fill_groups_with_time_means(df: pd.DataFrame) -> pd.DataFrame:
    """ Fill completely missing groups with time means """

    log.debug("Filling completely missing groups with time means.")
    data.check_has_multiindex(df)

    # TODO: Handle properly
    if not (df.dtypes == np.float64).all():
        log.warning("Not all cols are float64, this might break.")

    # Only fill numeric cols
    cols = list(df.select_dtypes(include=[np.number]).columns.values)
    for g_i, g_df in df.groupby(level=1):
        # If missing everything from a group
        if g_df.isnull().all().all():
            log.debug(
                f"All missing for groupvar {g_i}, filling with time mean"
            )
            # Get the times for this group
            times_group = g_df.index.get_level_values(0)
            # Fill all columns with the time mean
            df.loc[g_df.index, cols] = (
                df.loc[times_group, cols].groupby(level=0).mean().values
            )
    return df


@logtime
def fill_with_group_and_global_means(df: pd.DataFrame) -> pd.DataFrame:
    """ Impute missing values to group-level or global means. """
    log.debug("Filling missing with group means.")
    for col in df.columns:
        # impute with group level mean
        df[col].fillna(
            df.groupby(level=1)[col].transform("mean"), inplace=True
        )
        # fill remaining NaN with df level mean
        df[col].fillna(df[col].mean(), inplace=True)

    return df


@logtime
def extrapolate(df: pd.DataFrame) -> pd.DataFrame:
    """ Interpolate and extrapolate """
    data.check_has_multiindex(df)
    return (
        df.sort_index()
        .groupby(level=1)
        .apply(lambda group: group.interpolate(limit_direction="both"))
    )


@logtime
def _fill_iterative(
    df: pd.DataFrame,
    seed: int = 1,
    max_iter: int = 10,
    estimator: Any = BayesianRidge(),
):
    """ Gets a single imputation using IterativeImputer from sklearn.

    Uses BayesianRidge() from sklearn.

    Changed default of sample_posterior to True as we're doing
    multiple imputation.

    Clips imputed values to min-max of observed values to avoid
    brokenly large values. When imputation model doesn't converge
    nicely we otherwise end up with extreme values that are out of
    range of the float32 type used by model training, causing crashes.
    Consider this clipping a workaround until a more robust imputation
    strategy is in place.

    """

    log.info(
        "Started imputing "
        f"df with shape {df.shape} "
        f"missing share {df.isnull().mean().mean()}"
        f"with estimator {estimator.__class__.__name__}"
    )
    # Only impute numberic cols
    cols_numeric = list(df.select_dtypes(include=[np.number]).columns.values)
    cols_not_numeric = [col for col in df.columns if col not in cols_numeric]

    log.info(
        f"imputing {len(cols_numeric)} numeric cols, "
        f"ignoring {len(cols_not_numeric)} non-numeric cols"
    )

    for col in cols_numeric:
        log.debug(
            f"Missing share before impute {col} : {df[col].isnull().mean()}"
        )

    # Get bounds so we can clip imputed values to not be outside
    # observed values
    observed_min = df[cols_numeric].min()
    observed_max = df[cols_numeric].max()

    df_imputed = df.loc[:, []].copy()
    for col in df:
        df_imputed[col] = np.nan

    df_imputed[cols_numeric] = IterativeImputer(
        random_state=seed, max_iter=max_iter, estimator=estimator
    ).fit_transform(df[cols_numeric])
    df_imputed[cols_not_numeric] = df[cols_not_numeric]

    # Clip imputed values to observed min-max range
    df_imputed[cols_numeric] = df_imputed[cols_numeric].clip(
        observed_min, observed_max, axis=1
    )

    log.info(
        "Finished _fill_iterative()"
        f"Imputed df mising share numeric cols "
        f"{df[cols_numeric].isnull().mean().mean()}"
    )

    for col in cols_numeric:
        log.debug(
            "Missing share after impute "
            f"{col} : {df_imputed[col].isnull().mean()}"
        )

    return df_imputed


def impute_mice_generator(
    df, n_imp, estimator=None, parallel=False, n_jobs=mp.cpu_count()
):
    """ Impute df with MICE """

    if parallel:
        with mp.Pool(processes=n_jobs, maxtasksperchild=1) as pool:
            results = [
                pool.apply_async(_fill_iterative, (df, imp, 10, estimator,))
                for imp in range(n_imp)
            ]
            for result in results:
                yield result.get()

    else:
        log.info(f"Starting impute_mice_generator() for {n_imp} imputations")
        for imp in range(n_imp):
            log.info(f"Starting imp {imp}")
            yield _fill_iterative(df, seed=imp, estimator=estimator)
