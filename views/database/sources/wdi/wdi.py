""" WDI module """
import os
import logging
import tempfile

import pandas as pd  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore

from views.apps.data import missing
from views.utils import io, db, data
from views.database import common

log = logging.getLogger(__name__)


def fetch_wdi() -> None:
    """ Fetch WDI from world bank website """

    log.info("Started fetching WDI.")
    url = "http://databank.worldbank.org/data/download/WDI_csv.zip"
    common.fetch_source_simply(name="wdi", url=url)
    log.info("Finished fetchign WDI.")


def _flip_wdi(df: pd.DataFrame) -> pd.DataFrame:
    """ Flip WDI data from wide to long """

    log.info("Flipping WDI")

    df = df.rename(columns=lambda x: x.replace(" ", ""))
    df = df.rename(columns=lambda x: x.lower())

    # Headache-magic, tbh I don't remember how it works.
    df = df.drop(["countryname", "indicatorname"], axis=1)
    df = df.set_index(["countrycode", "indicatorcode"])
    df.columns.name = "year"
    df = df.stack().unstack("indicatorcode")
    df = df.reset_index()
    df["year"] = df["year"].astype("int32")
    df = df.set_index(["year", "countrycode"]).sort_index()

    df = df.rename(columns=lambda x: x.replace(".", "_"))
    df = df.rename(columns=lambda x: x.lower())

    log.info("Done flipping WDI")

    return df


def _load_and_stage_wdi() -> pd.DataFrame:

    log.debug("Reading raw fetch.")
    with tempfile.TemporaryDirectory() as tempdir:
        paths = common.get_files_latest_fetch(name="wdi", tempdir=tempdir)
        path_zip = [
            path for path in paths if os.path.basename(path) == "WDI_csv.zip"
        ].pop()
        io.unpack_zipfile(path_zip, destination=tempdir)
        df = io.csv_to_df(path=os.path.join(tempdir, "WDIData.csv"))
        # TODO: Build codebook from this
        _ = io.csv_to_df(path=os.path.join(tempdir, "WDISeries.csv"))

    log.debug("Preparing WDI.")
    df = _flip_wdi(df=df)
    # Get country_id isoab matching
    log.debug("Fetching df_keys")
    df_keys = db.query_to_df(
        query="""
        SELECT id AS country_id, isoab AS countrycode FROM staging.country;
        """
    )

    # Drop duplicates, Soviet Union, Yugoslavia etc
    # Keep those with highest country_id, i.e. latest.
    df_keys = (
        df_keys.sort_values(by="country_id", ascending=False)
        .drop_duplicates(subset=["countrycode"])
        .set_index(["countrycode"])
    )

    # Join in keys
    log.debug("Joining in df_keys")
    df = df.join(df_keys)
    df = (
        df.reset_index()
        .dropna(subset=["country_id"])
        .set_index(["year", "country_id"])
        .add_prefix("wdi_")
        .drop(columns=["wdi_countrycode"])
    )

    # Stage to CY skeleton
    log.debug("Fetching skeleton")
    df_skeleton = db.db_to_df(
        fqtable="skeleton.cy_global", cols=["year", "country_id"]
    ).set_index(["year", "country_id"])
    df = df_skeleton.join(df, how="left")

    # Drop cols that have no values at all
    cols_completely_missing = missing.list_totally_missing(df)
    df = df.drop(columns=cols_completely_missing)
    log.debug(
        f"Dropped cols {cols_completely_missing} because they had no values"
    )

    # order columns and rows
    df = df.sort_index(axis=1).sort_index(axis=0)

    return df


def _fill_and_push_wdi(df: pd.DataFrame, schema: str, n_imp: int) -> None:

    # We have data for 1980-2019, 2020 and beyond are just extrapolated
    df_data = df.loc[1980:2019].copy()
    df_data = data.rebuild_index(df_data)

    log.debug(
        "Shares missing for 1980-2019 before fill: "
        f"{df_data.isnull().mean().mean()}"
    )

    # Some countries are completely missing, probably due to join issues
    # Fill their values with time specific means
    df_data = missing.fill_groups_with_time_means(df_data)
    log.debug(
        f"Shares missing for 1980-2019 after filling missing countries "
        f"with time means: {df_data.isnull().mean().mean()} ."
    )

    # Interpolate and extrapolate data we have
    df_data = missing.extrapolate(df_data)
    log.debug(
        f"Shares missing for 1980-2019 after inter/extra-polating: "
        f"{df_data.isnull().mean().mean()} "
    )

    # Impute the remaining missing data, then extrapolate the future
    log.info("Starting WDI sklearn imputation")
    for i, df_imp in enumerate(
        missing.impute_mice_generator(
            df=df_data,
            n_imp=n_imp,
            estimator=DecisionTreeRegressor(max_features="sqrt"),
            parallel=True,
        )
    ):
        db.df_to_split_db(
            df=missing.extrapolate(df_imp.reindex(df.index)),
            fqtables=[f"{schema}.cy_imp_sklearn_{i}_part_{p}" for p in [1, 2]],
        )


def load_wdi() -> None:
    """ Load WDI to database """

    log.info("Started loading WDI.")
    df = _load_and_stage_wdi()

    schema = "wdi_202005"
    db.drop_schema(schema)
    db.create_schema(schema)

    # Push completely raw but staged data
    log.debug("Done preparing raw WDI, pushing.")
    fqtables = [f"{schema}.cy_unimp_part_{p}" for p in [1, 2]]
    db.df_to_split_db(df=df, fqtables=fqtables)

    _fill_and_push_wdi(df=df, schema=schema, n_imp=5)
    log.info("Finished loading WDI.")
