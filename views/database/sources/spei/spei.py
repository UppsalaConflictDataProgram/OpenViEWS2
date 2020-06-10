""" SPEI module """
import os
import tempfile
from typing import Tuple
import multiprocessing as mp
import logging
import math

import pandas as pd  # type: ignore
import xarray as xr  # type: ignore
import numba  # type: ignore


from views.utils import io, db
from views.database import common

log = logging.getLogger(__name__)


def fetch_spei() -> None:
    """ Fetch SPEI """
    urls = [
        f"https://soton.eead.csic.es/spei/10/nc/spei{i:0>2d}.nc"
        for i in range(1, 49)
    ]
    common.fetch_source_simply(name="spei", urls=urls)


def _get_id_dfs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Get dataframes with ids from database """
    db.execute_query(
        query=io.read_file(
            path=os.path.join(os.path.dirname(__file__), "pg_ug.sql")
        )
    )
    df_pg_ug = db.db_to_df(fqtable="spei_v2.pg_ug", ids=["pg_id"])

    df_m = (
        db.db_to_df(
            fqtable="staging.month", cols=["id"], ids=["year_id", "month"]
        )
        .reset_index()
        .rename(columns={"year_id": "year", "id": "month_id"})
        .set_index(["year", "month"])
    )

    df_ug_pgm = (
        db.db_to_df(
            fqtable="staging.priogrid_month",
            cols=["id"],
            ids=["priogrid_gid", "month_id"],
        )
        .reset_index()
        .rename(columns={"id": "priogrid_month_id", "priogrid_gid": "pg_id"})
        .set_index(["pg_id", "month_id"])
        .join(df_pg_ug)
        .reset_index()
        .set_index(["ug_id", "month_id"])[["pg_id", "priogrid_month_id"]]
    )

    return df_pg_ug, df_m, df_ug_pgm


@numba.vectorize([numba.int32(numba.float64, numba.float64)])
def _priogrid_vec(lat, lon):
    """ Get pg_id from latitide and longitude vectorised """

    lat_part = ((int((90 + (math.floor(lat * 2) / 2)) * 2) + 1) - 1) * 720
    lon_part = (180 + (math.floor(lon * 2) / 2)) * 2
    pg_id = lat_part + lon_part + 1
    pg_id = int(pg_id)

    return pg_id


def _spei_num_from_path(path: str) -> int:
    """ Get the SPEI number from the path """
    spei_num = "".join([s for s in os.path.basename(path) if s.isdigit()])
    return int(str(int(spei_num)))  # Get rid of leading zeros


def _load_spei_from_path(
    path: str,
    df_pg_ug: pd.DataFrame,
    df_m: pd.DataFrame,
    df_ug_pgm: pd.DataFrame,
) -> None:
    """ Load a single SPEI from .nc at path """

    spei_num = _spei_num_from_path(path)
    colname = f"spei_{spei_num}"

    # Use xarray to load the .nc, it deals with indexing, time etc
    df = xr.open_dataset(path).to_dataframe().reset_index()
    log.debug(f"Read {len(df)} rows of data from {path}")

    # Vectorised pg_id assignment on lat/lon
    df["pg_id"] = _priogrid_vec(df.lat.to_numpy(), df.lon.to_numpy())
    log.debug("Assigned pg_ids")

    # Get year/month from date for joining
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df.drop(columns=["time", "lat", "lon"], inplace=True)

    # Put the SPEI number in the colname
    df.rename(columns={"spei": colname}, inplace=True)
    # Drop rows with no data
    df.dropna(inplace=True)

    # Join in month_id and drop those with missing
    df = df.set_index(["year", "month"]).join(df_m).dropna()
    df.month_id = df.month_id.astype(int)
    log.debug("Assigned month_id")

    # Keep only data
    df = df.reset_index().set_index(["pg_id", "month_id"])[[colname]]

    # Join in the ug_id
    df = df.join(df_pg_ug).dropna()
    df.ug_id = df.ug_id.astype(int)
    log.debug("Assigned ug_id")

    # Now reindex to ug_id-month_id so we can join bigly
    df = df.reset_index().set_index(["ug_id", "month_id"]).sort_index()
    df = df[[colname]]

    # Now join in pg_id and priogrid_month_id by ug_id
    df = df_ug_pgm.join(df)

    df = df.set_index(["priogrid_month_id"])[[colname]].dropna()

    log.info(f"Started pushing {path}")

    # Push it up and create index
    fqtable = f"spei_v2.spei_{spei_num}"
    db.df_to_db(fqtable=fqtable, df=df)
    db.execute_query(query=f"CREATE INDEX ON {fqtable} (priogrid_month_id);")
    log.info(f"{fqtable} ready")


def _stage_spei():
    """ Stage SPEI """
    log.debug("Started staging SPEI")
    db.execute_query(
        query=io.read_file(
            os.path.join(os.path.dirname(__file__), "stage.sql")
        )
    )
    db.execute_query(
        query=io.read_file(
            os.path.join(os.path.dirname(__file__), "cleanup.sql")
        )
    )
    log.debug("Finished staging SPEI")


def load_spei(parallel=True) -> None:
    """ Load SPEI """
    log.info("Started loading SPEI.")
    db.drop_schema("spei_v2")
    db.create_schema("spei_v2")
    df_pg_ug, df_m, df_ug_pgm = _get_id_dfs()

    with tempfile.TemporaryDirectory() as tempdir:
        paths = common.get_files_latest_fetch(name="spei", tempdir=tempdir)

        if parallel:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = [
                    pool.apply_async(
                        func=_load_spei_from_path,
                        args=(path, df_pg_ug, df_m, df_ug_pgm),
                    )
                    for path in paths
                ]
                _ = [result.get() for result in results]
        else:
            for path in paths:
                _load_spei_from_path(path, df_pg_ug, df_m, df_ug_pgm)

    _stage_spei()
    log.info("Finished loading SPEI.")
