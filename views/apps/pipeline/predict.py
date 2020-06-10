""" Pipeline prediction interface """
import os
import logging
import multiprocessing as mp
from typing import List, Tuple, Optional
import warnings
import tempfile
import uuid

import pandas as pd  # type: ignore

from views.apps.data import api as data_api
from views.apps.model import api
from views import config
from views.utils.data import assign_into_df, rebuild_index
from views.specs.periods import get_periods_by_name
from views.utils import io
from views.utils.log import logtime

log = logging.getLogger(__name__)

DIR_PREDICTIONS = os.path.join(config.DIR_STORAGE, "pipeline", "predictions")


def get_period_pairs(run_id: str) -> List[Tuple[api.Period, api.Period]]:
    """ Get the calib-test period pairs for a run

    Prediction runs will only have B and C periods.
    Some runs will have A, B and C.
    In the first case return a list of a single pair.
    In the seconds case return a list of two pairs, A-B and B-C
    """
    periods_by_name = get_periods_by_name(run_id)

    # Chech we have at least B and C keys
    if not all(["B" in periods_by_name and "C" in periods_by_name]):
        raise RuntimeError(
            f"run_id {run_id} doesn't have B and C keys"
            f"Only has {periods_by_name.keys()}"
        )

    period_pairs: List[Tuple[api.Period, api.Period]] = []
    if "A" in periods_by_name:
        period_pairs.append((periods_by_name["A"], periods_by_name["B"]))
    period_pairs.append((periods_by_name["B"], periods_by_name["C"]))

    return period_pairs


def get_all_times_predict(
    period_pairs: List[Tuple[api.Period, api.Period]]
) -> List[int]:
    """ Get all times to predict for a list of periods pairs """

    times_predict = []
    for period_pair in period_pairs:
        for period in period_pair:
            times_predict += period.times_predict

    # Drop duplicates and return
    return sorted(list(set(times_predict)))


def path_prediction(run_id: str, dataset: data_api.Dataset) -> str:
    """ Get the path to a stored prediction df """
    dir_run = os.path.join(DIR_PREDICTIONS, run_id)
    io.create_directory(dir_run)
    path = os.path.join(
        dir_run, f"predictions_{run_id}_{dataset.name}.parquet"
    )
    return path


@logtime
def store_prediction_on_disk(
    df: pd.DataFrame, run_id: str, dataset: data_api.Dataset
) -> None:
    """ Store df of predictions on disk """
    log.info("Storing predictions")
    io.df_to_parquet(
        df=df, path=path_prediction(run_id=run_id, dataset=dataset)
    )


@logtime
def get_predictions_from_disk(
    run_id: str, dataset: data_api.Dataset, cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """ Read predictions from disk """
    log.info("Reading predictions from disk")
    return io.parquet_to_df(
        path=path_prediction(run_id=run_id, dataset=dataset), cols=cols
    )


@logtime
def predict_ensemble(
    ensemble: api.Ensemble,
    dataset: data_api.Dataset,
    run_id: str,
    tempdir: str,
) -> str:
    """ Predict for a single ensemble """
    cols_needed = ensemble.cols_needed
    cols_data = dataset.list_cols_cached()
    cols_pred = io.list_columns_in_parquet(
        path=path_prediction(run_id, dataset)
    )

    # Check we have all we need
    cols_missing = [
        col not in cols_data + cols_pred for col in ensemble.cols_needed
    ]
    if any(cols_missing):
        raise RuntimeError(
            f"Ensemble {ensemble.name} missing cols {cols_missing}"
        )

    # Get constituent predictions and features (outcomes) needed for ensemble
    df_constituent = io.parquet_to_df(
        path=path_prediction(run_id, dataset),
        cols=[col for col in cols_needed if col in cols_pred],
    )
    df_data = dataset.get_df_cols(
        cols=[col for col in cols_needed if col in cols_data]
    )
    df = df_constituent.join(df_data)

    period_pairs = get_period_pairs(run_id)

    # Empty df to hold predictions
    df_pred = df.loc[get_all_times_predict(period_pairs), []]

    for period_pair in period_pairs:
        df_pred = assign_into_df(
            df_to=df_pred,
            df_from=ensemble.predict(
                df=df, period_calib=period_pair[0], period_test=period_pair[1]
            ),
        )

    log.debug(
        f"Done predicting for ensemble {ensemble.name}, writing results."
    )
    # Generate a random filename in the tempdir
    path = os.path.join(tempdir, f"{uuid.uuid4().hex}.parquet")
    io.df_to_parquet(df=df_pred, path=path)
    return path


@logtime
def predict_ensembles(
    ensembles: List[api.Ensemble], dataset, run_id, n_cores: int
) -> None:
    """ Predict for ensembles """

    # Get empty df of times predict to hold predictions
    df_pred_ensembles = dataset.get_df_cols(cols=[]).loc[
        get_all_times_predict(get_period_pairs(run_id))
    ]

    # Use only half the cores, using all memoryerrors a rackham node.
    with mp.Pool(processes=n_cores, maxtasksperchild=1) as pool:
        with tempfile.TemporaryDirectory() as tempdir:
            results = []
            for ensemble in ensembles:
                results.append(
                    pool.apply_async(
                        predict_ensemble,
                        (ensemble, dataset, run_id, tempdir,),
                    )
                )
            for result in results:
                path = result.get()
                df_pred_ensembles = assign_into_df(
                    df_to=df_pred_ensembles, df_from=io.parquet_to_df(path)
                )
    # Join ensemble and constituent predictions and write them all to disk
    store_prediction_on_disk(
        df=df_pred_ensembles.join(
            get_predictions_from_disk(run_id=run_id, dataset=dataset)
        ),
        run_id=run_id,
        dataset=dataset,
    )


@logtime
def predict_model(
    model: api.Model,
    dataset: data_api.Dataset,
    period_calib: api.Period,
    period_test: api.Period,
    tempdir: str,
) -> str:
    """ Predict for single model """

    log.info(
        f"Started predicting for model {model.name} "
        f"period_calib: {period_calib.name} "
        f"period_test: {period_test.name}."
    )

    # Read in only features needed to predict for this model
    # @TODO: remove fillna(0), make sure input data is missing-free.
    cols_needed = model.cols_features + [model.col_outcome]
    df = dataset.get_df_cols(cols_needed).fillna(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df_calib = model.predict(df=df, period=period_calib)
        df_test = model.predict(df=df, period=period_test)
        df_calibrated = model.predict_calibrated(
            df=df, period_calib=period_calib, period_test=period_test
        )

    all_times_predict = period_calib.times_predict + period_test.times_predict
    df_pred = rebuild_index(df.loc[all_times_predict, []])
    for df_from in [df_calib, df_test, df_calibrated]:
        df_pred = assign_into_df(df_to=df_pred, df_from=df_from)
    log.info(f"Fininshed predicting for {model.name}")

    # Generate a random filename in the tempdir
    path = os.path.join(tempdir, f"{uuid.uuid4().hex}.parquet")
    io.df_to_parquet(df=df_pred, path=path)
    return path


@logtime
def predict_models(
    models: List[api.Model],
    dataset: data_api.Dataset,
    run_id: str,
    n_cores: int,
) -> None:
    """ Predict for models """

    # Get our calib/test period pairs
    period_pairs = get_period_pairs(run_id)

    log.info(
        f"Predicting for {len(models)} models "
        f"for {len(period_pairs)} period pairs."
    )

    # Create predictions df with predict times and no cols
    df_pred = dataset.get_df_cols(cols=[]).loc[
        get_all_times_predict(period_pairs), []
    ]

    # Predict the models in parallel
    with mp.get_context("spawn").Pool(
        processes=n_cores, maxtasksperchild=1
    ) as pool:
        # with mp.Pool(processes=n_cores, maxtasksperchild=1) as pool:
        with tempfile.TemporaryDirectory() as tempdir:
            results = []
            for period_pair in period_pairs:
                for model in models:
                    # period_pair[0] is calib period and [1] the test period
                    results.append(
                        pool.apply_async(
                            predict_model,
                            (
                                model,
                                dataset,
                                period_pair[0],
                                period_pair[1],
                                tempdir,
                            ),
                        )
                    )
            # Collect as results become ready
            for result in results:
                path = result.get()
                log.debug(f"Insert from {path}")
                df_pred = assign_into_df(
                    df_to=df_pred, df_from=io.parquet_to_df(path)
                )

    log.debug("Done collecting.")
    store_prediction_on_disk(df=df_pred, run_id=run_id, dataset=dataset)
