""" VDEM module """
import os
import tempfile
import logging

import pandas as pd  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore

from views.apps.data import missing
from views.utils import io, db, data
from views.database import common

log = logging.getLogger(__name__)

COLS_META = [
    "country_name",
    "country_text_id",
    "historical_date",
    "project",
    "historical",
    "histname",
    "codingstart",
    "codingend",
    "codingstart_contemp",
    "codingend_contemp",
    "codingstart_hist",
    "codingend_hist",
    "gapstart1",
    "gapstart2",
    "gapstart3",
    "gapend1",
    "gapend2",
    "gapend3",
    "cowcode",
]


def fetch_vdem() -> None:
    """ Fetch VDEM """
    log.info("Fetching VDEM.")

    url_base = "http://v-dem.pol.gu.se/v10"
    urls = [
        f"{url_base}/Country_Year_V-Dem_Full+others_CSV_v10.zip",
        f"{url_base}/v10/Country_Date_V-Dem_CSV_v10.zip",
        f"{url_base}/v10/Coder_Level_V-Dem_CSV_v10.zip",
    ]

    common.fetch_source_simply(name="vdem_v10", urls=urls)
    log.info("Finished fetching VDEM")


def _load_and_stage_vdem() -> pd.DataFrame:
    """ Load and stage VDEM """
    log.debug("Loading raw fetch data for VDEM.")
    with tempfile.TemporaryDirectory() as tempdir:
        _ = common.get_files_latest_fetch(name="vdem_v10", tempdir=tempdir)

        _ = io.unpack_zipfile(
            path_zip=os.path.join(
                tempdir, "Country_Year_V-Dem_Full+others_CSV_v10.zip"
            ),
            destination=tempdir,
        )
        path_df = os.path.join(
            tempdir,
            "Country_Year_V-Dem_Full+others_CSV_v10",
            "V-Dem-CY-Full+Others-v10.csv",
        )
        df = (
            io.csv_to_df(path=path_df)
            .add_prefix("vdem_")
            .rename(columns={"vdem_year": "year"})
            .set_index(["year", "vdem_country_text_id"])
        )

    df_keys = (
        db.query_to_df(
            query="""
            SELECT id AS country_id, isoab AS vdem_country_text_id
            FROM staging.country;
            """
        )
        .sort_values(by="country_id", ascending=False)
        .drop_duplicates(subset=["vdem_country_text_id"])
        .set_index(["vdem_country_text_id"])
    )
    df = df.join(df_keys)

    # Drop where join failed
    df.dropna(subset=["country_id"])
    df = df.reset_index().set_index(["year", "country_id"]).sort_index()
    df.isnull().mean().mean()

    # Stage to CY skeleton
    log.debug("Fetching skeleton")
    df_skeleton = db.db_to_df(
        fqtable="skeleton.cy_global", cols=["year", "country_id"]
    ).set_index(["year", "country_id"])
    df = df_skeleton.join(df, how="left")

    cols_completely_missing = missing.list_totally_missing(df)
    df = df.drop(columns=cols_completely_missing)
    log.debug(
        f"Dropped cols {cols_completely_missing} because they had no values"
    )

    # order columns and rows
    df = df.rename(columns=lambda col: col.lower())

    cols = sorted(list(df.columns))
    cols = [col for col in cols if not col.endswith("_codehigh")]
    cols = [col for col in cols if not col.endswith("_codelow")]
    cols = [col for col in cols if not col.endswith("_ord")]
    cols = [col for col in cols if not col.endswith("_sd")]
    cols = [col for col in cols if not col.endswith("_mean")]
    cols = [col for col in cols if not col.endswith("_nr")]
    cols = [col for col in cols if not col.endswith("_osp")]
    df = df[cols]

    df = df.sort_index(axis=1).sort_index(axis=0)

    return df


def _fill_and_push_vdem(df: pd.DataFrame, schema: str, n_imp: int) -> None:
    """ Fill missing and push """
    df_data = df.loc[1980:2019].copy()
    df_data = data.rebuild_index(df_data)
    log.debug(
        "Shares missing for 1980-2019 before fill: "
        f"{df_data.isnull().mean().mean()}"
    )

    df_data = missing.fill_groups_with_time_means(df_data)
    log.debug(
        "Shares missing for 1980-2019 after filling missing countries "
        f"with time means: {df_data.isnull().mean().mean()} ."
    )

    df_data = missing.extrapolate(df_data)
    log.debug(
        "Shares missing for 1980-2019 after inter/extra-polating: "
        f"{df_data.isnull().mean().mean()} "
    )

    # Impute the remaining missing data, then extrapolate the future
    log.info("Starting sklearn imputation of VDEM.")
    for i, df_imp in enumerate(
        missing.impute_mice_generator(
            df=df_data,
            n_imp=n_imp,
            estimator=DecisionTreeRegressor(max_features="sqrt"),
            parallel=True,
        )
    ):
        db.df_to_db(
            df=missing.extrapolate(df_imp.reindex(df.index)),
            fqtable=f"{schema}.cy_imp_sklearn_{i}",
        )


def load_vdem() -> None:
    """ Load VDEM """
    log.info("Started loading VDEM")
    df = _load_and_stage_vdem()

    schema = "vdem_v10"
    # db.drop_schema(schema)
    # db.create_schema(schema)

    log.debug("Done preparing raw VDEM, pushing.")

    db.df_to_db(df=df, fqtable=f"{schema}.cy_unimp")

    _fill_and_push_vdem(df=df, schema=schema, n_imp=5)
    log.info("Finished loading VDEM.")
