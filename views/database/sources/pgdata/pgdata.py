""" Priogrid data """
import os
import tempfile
import multiprocessing as mp
import logging
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore


from views.database import common
from views.utils import io, db
from views.apps.data import missing

log = logging.getLogger(__name__)


def _inserts_vinfs_data_to_df(
    tempdir: str, df: pd.DataFrame, vinfs: Any, ids: Any, drops: Any
) -> pd.DataFrame:
    """ Insert data from varinfos into df in parallel """

    with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:

        results = [
            pool.apply_async(_vinf_to_s, args=(tempdir, vinf, ids, drops))
            for vinf in vinfs
        ]
        result_series = [res.get() for res in results]

    for s in result_series:
        log.debug(f"Inserting {s.name}")
        df[s.name] = s

    return df.sort_index()


def _vinf_to_s(
    tempdir: str,
    vinf: Dict[Any, Any],
    ids: List[str],
    drops: Optional[List[str]] = None,
):
    """ Get a single series from a varinfo """

    varname = vinf["name"]
    data_dict = io.load_json(os.path.join(tempdir, f"{varname}.json"))
    df = pd.DataFrame.from_dict(data_dict["cells"])
    df = df.rename(columns={"value": varname})
    df = df.set_index(ids)
    if drops:
        df = df.drop(columns=drops)

    s = df[varname]
    s.name = varname

    return s


def _prepare(df: pd.DataFrame, spec: Dict[Any, Any]) -> pd.DataFrame:
    """ Preparations before pushing """

    def ffill(df, spec):
        for col in [col for col in spec["cols_ffill"] if col in df.columns]:
            df[col] = df[col].groupby(level=0).fillna(method="ffill")
        return df

    def fill_nulls_to_zero(df, spec):
        """ Cols with _y and _s have nulls for zeros, fill with 0"""

        for col in spec["nulls_to_zero"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    df = ffill(df, spec)
    df = fill_nulls_to_zero(df, spec)

    return df


def load_initial_pgdata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Load pgdata into three dataframes: static, yearly and core """
    spec = io.load_yaml(os.path.join(os.path.dirname(__file__), "spec.yaml"))
    with tempfile.TemporaryDirectory() as tempdir:
        _ = common.get_files_latest_fetch(name="pgdata", tempdir=tempdir)

        varinfos = io.load_json(os.path.join(tempdir, "varinfos.json"))
        basegrid = io.load_json(os.path.join(tempdir, "basegrid.json"))

        varinfos_static = [vi for vi in varinfos if vi["type"] == "static"]
        varinfos_yearly = [vi for vi in varinfos if vi["type"] == "yearly"]
        varinfos_core = [vi for vi in varinfos if vi["type"] == "core"]
        varinfos_core = list(
            filter(
                lambda x: x["name"] not in spec["excludes_core"], varinfos_core
            )
        )

        # Build the indices for the dfs
        y_start = min([vinf["startYear"] for vinf in varinfos_yearly])
        y_end = max([vinf["endYear"] for vinf in varinfos_yearly])
        years = list(range(y_start, y_end + 1))
        gids = [cell["gid"] for cell in basegrid]

        df_static = _inserts_vinfs_data_to_df(
            tempdir=tempdir,
            df=pd.DataFrame(index=pd.Index(gids, name="gid")),
            vinfs=varinfos_static,
            ids=["gid"],
            drops=["year"],
        )
        df_static = _prepare(df_static, spec)

        df_yearly = _inserts_vinfs_data_to_df(
            tempdir=tempdir,
            df=pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [gids, years], names=["gid", "year"]
                )
            ),
            vinfs=varinfos_yearly,
            ids=["gid", "year"],
            drops=None,
        )
        df_yearly = _prepare(df_yearly, spec)

        df_core = _inserts_vinfs_data_to_df(
            tempdir=tempdir,
            df=pd.DataFrame(index=pd.Index(gids, name="gid")),
            vinfs=varinfos_core,
            ids=["gid"],
            drops=["year"],
        )
        df_core = _prepare(df_core, spec)

    # Set indices
    df_static = (
        df_static.reset_index()
        .rename(columns={"gid": "pg_id"})
        .set_index(["pg_id"])
        .sort_index()
    )
    df_yearly = (
        df_yearly.reset_index()
        .rename(columns={"gid": "pg_id"})
        .set_index(["year", "pg_id"])
        .sort_index()
    )
    df_core = (
        df_core.reset_index()
        .rename(columns={"gid": "pg_id"})
        .set_index(["pg_id"])
        .sort_index()
    )

    return df_static, df_yearly, df_core


def compute_greatests(df: pd.DataFrame) -> pd.DataFrame:
    """ Compute greatest transforms to combine yearly (_y) ans static (_s) """
    df["diamprim"] = df[["diamprim_s", "diamprim_y"]].max(axis=1)
    df["diamsec"] = df[["diamsec_s", "diamsec_y"]].max(axis=1)
    df["gem"] = df[["gem_s", "gem_y"]].max(axis=1)
    df["goldplacer"] = df[["goldplacer_s", "goldplacer_y"]].max(axis=1)
    df["goldsurface"] = df[["goldsurface_s", "goldsurface_y"]].max(axis=1)
    df["goldvein"] = df[["goldvein_s", "goldvein_y"]].max(axis=1)
    df["petroleum"] = df[["petroleum_s", "petroleum_y"]].max(axis=1)
    return df


def finalise_pgdata(
    df_static: pd.DataFrame, df_yearly: pd.DataFrame, df_core: pd.DataFrame,
) -> pd.DataFrame:
    """ Join all pgdata to PGY level """
    spec = io.load_yaml(os.path.join(os.path.dirname(__file__), "spec.yaml"))
    df_skeleton = db.db_to_df(
        fqtable="skeleton.pgy_global",
        ids=["year", "pg_id"],
        cols=["year", "pg_id"],
    )
    df = df_skeleton.join(df_static).join(df_core).join(df_yearly).sort_index()

    # Check that the default inner join doesn't discard any rows
    if not len(df) == len(df_skeleton):
        raise RuntimeError("Join was supposed to have all skeleton rows.")

    # Compute the _s, _y transformations
    df = compute_greatests(df)

    # Subset to wanted columns
    df = df[spec["cols_data"]]

    df = missing.extrapolate(df)

    df = df.add_prefix("pgd_")

    return df


def _impute_and_push_pgdata(df: pd.DataFrame) -> None:
    for i, df_imp in enumerate(
        missing.impute_mice_generator(
            df=df,
            n_imp=5,
            estimator=DecisionTreeRegressor(max_features="sqrt"),
            parallel=True,
            n_jobs=2,
        )
    ):
        fqtable = f"pgdata.pgy_imp_sklearn_{i}"
        db.df_to_db(df=df_imp, fqtable=fqtable)


def load_pgdata() -> None:
    """ Load and impute priogrid data """
    df = finalise_pgdata(*load_initial_pgdata())
    db.df_to_db(df=df, fqtable="pgdata.pgy_unimp")
    _impute_and_push_pgdata(df)
