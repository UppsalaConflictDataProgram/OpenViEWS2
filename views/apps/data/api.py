""" Data abstractions """
from dataclasses import dataclass
import os
import logging
from typing import Any, Dict, Optional, List, Tuple, Union
from typing_extensions import Literal

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from views.apps.transforms import lib
from views.utils import db, io, data as datautils
from views.utils.log import logtime
from views.config import DIR_STORAGE

from . import missing

log = logging.getLogger(__name__)


DIR_CACHE_TABLES = os.path.join(DIR_STORAGE, "data", "tables")
DIR_CACHE_DATASETS = os.path.join(DIR_STORAGE, "data", "datasets")
DIR_CACHE_GEOMS = os.path.join(DIR_STORAGE, "data", "geometries")

TRANSFORMS_SERIES: Dict[str, Any] = {
    "delta": lib.delta,
    "greq": lib.greater_or_equal,
    "smeq": lib.smaller_or_equal,
    "in_range": lib.in_range,
    "tlag": lib.tlag,
    "tlead": lib.tlead,
    "moving_average": lib.moving_average,
    "cweq": lib.cweq,
    "time_since": lib.time_since,
    "decay": lib.decay,
    "mean": lib.mean,
    "ln": lib.ln,
    "demean": lib.demean,
    "rollmax": lib.rollmax,
    "onset_possible": lib.onset_possible,
    "onset": lib.onset,
}
TRANSFORMS_DF: Dict[str, Any] = {
    "sum": lib.summ,
    "product": lib.product,
}
TRANSFORMS_GDF: Dict[str, Any] = {
    "spdist": lib.distance_to_event,
    "stdist": lib.spacetime_distance_to_event,
    "splag": lib.spatial_lag,
}


class Transform:
    """ Transformer wrapper

    Can be initalised from a spec dictionary and exposes the
    .compute(df) method which deals with the input shapes of the
    underlying transformation functions.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, name, **kwargs):
        def _parse_cols(kwargs):
            """ Get 'col' or 'cols' from kwargs to self.cols """

            # Get col or cols from kwargs
            try:
                cols = kwargs.pop("cols")
            except KeyError:
                try:
                    cols = [kwargs.pop("col")]
                except KeyError:
                    raise KeyError(f"Need 'col' or 'cols' field in {kwargs}")

            # Make sure cols is now a list of strings
            if not all(
                isinstance(element, str) for element in cols
            ) or not isinstance(cols, list):
                raise TypeError(
                    f"col should be string or cols should be list of strings. "
                    f"Now parsed cols is '{cols}'"
                )
            return cols

        def _parse_f(kwargs) -> Tuple[str, Any, str]:
            """ Look up the actual function to use from the functions dicts """
            try:
                f_name = kwargs.pop("f")
            except KeyError:
                raise KeyError(
                    "Transformer needs a 'f' field to know which func to use"
                )

            # Look in all transforms dicts
            all_transforms_keys = (
                list(TRANSFORMS_SERIES.keys())
                + list(TRANSFORMS_DF.keys())
                + list(TRANSFORMS_GDF.keys())
            )
            try:
                func = TRANSFORMS_SERIES[f_name]
                input_type = "s"
            except KeyError:
                try:
                    func = TRANSFORMS_DF[f_name]
                    input_type = "df"
                except KeyError:
                    try:
                        func = TRANSFORMS_GDF[f_name]
                        input_type = "gdf"
                    except KeyError:
                        raise KeyError(
                            f"You passed f={f_name}. "
                            "Only the following values of f are recognised: "
                            f"{all_transforms_keys}"
                        )
            return f_name, func, input_type

        self.name = name
        self.f_name, self.func, self.input_type = _parse_f(kwargs)
        self.cols_input = _parse_cols(kwargs)
        self.kwargs = kwargs

    @logtime
    def compute(
        self, data: Union[gpd.GeoDataFrame, pd.DataFrame]
    ) -> pd.Series:
        """ Compute the transformation on data and return a series """

        log.debug(f"Computing transform {self.name}")
        # Match for input_type
        if self.input_type == "s":
            return self.func(s=data[self.cols_input[0]], **self.kwargs)
        elif self.input_type == "df":
            return self.func(df=data[self.cols_input], **self.kwargs)
        elif self.input_type == "gdf":
            return self.func(gdf=data, col=self.cols_input[0], **self.kwargs)
        else:
            raise RuntimeError(
                f"Something broke with Transform, "
                f"couldn't match self.input_type {self.input_type}"
            )


class GeomCountry:
    """ Country geometry wrapper """

    def __init__(self):
        self.name = "country"
        self.fname = "country.geojson"
        self.path = os.path.join(DIR_CACHE_GEOMS, self.fname)
        self.fqtable = "staging.country"

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """ Geodataframe """
        if os.path.isfile(self.path):
            log.info(f"Found gdf on file at {self.path}")
            return (
                io.geojson_to_gdf(self.path)
                .set_index(["country_id"])
                .sort_index()
            )
        else:
            return self.refresh()

    @logtime
    def init_cache_from_geojson(self, path):
        """ Copy the geojson from path to self path in DIR_CACHE """
        io.copy_file(path_from=path, path_to=self.path)

    @logtime
    def refresh(self) -> gpd.GeoDataFrame:
        """ Refersh from db """
        gdf = db.db_to_gdf(
            fqtable="staging.country",
            cols=["id", "geom", "gweyear"],
            geom_col="geom",
        )
        gdf = gdf.rename(columns={"id": "country_id"}).set_index(
            ["country_id"]
        )
        gdf = (
            gdf[gdf["gweyear"] == 2016]
            .reset_index()
            .set_index(["country_id"])
            .sort_index()
        )
        gdf = gdf.loc[:, ["geom"]]
        io.gdf_to_geojson(gdf=gdf, path=self.path)
        return gdf


class GeomPriogrid:
    """ Priogrid geometry wrapper """

    def __init__(self):
        self.name = "priogrid"
        self.fname = "priogrid.geojson"
        self.path = os.path.join(DIR_CACHE_GEOMS, self.fname)
        self.fqtable = "staging.priogrid"

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """ Geodataframe """
        if os.path.isfile(self.path):
            log.info(f"Found gdf on file at {self.path}")
            return (
                io.geojson_to_gdf(self.path).set_index(["pg_id"]).sort_index()
            )
        else:
            return self.refresh()

    @logtime
    def init_cache_from_geojson(self, path):
        """ Copy the geojson from path to self path in DIR_CACHE """
        io.copy_file(path_from=path, path_to=self.path)

    @logtime
    def refresh(self) -> gpd.GeoDataFrame:
        """ Refresh from db """
        gdf = db.db_to_gdf(
            fqtable="staging.priogrid", cols=["gid", "geom"], geom_col="geom"
        )
        gdf = (
            gdf.rename(columns={"gid": "pg_id"})
            .set_index(["pg_id"])
            .sort_index()
        )
        io.gdf_to_geojson(gdf=gdf, path=self.path)
        return gdf


@dataclass
class Table:
    """ Represents a table in the database """

    fqtable: str
    ids: List[str]
    # An in-memory cache of the column list
    _cols: Optional[List[str]] = None

    @property
    def name(self) -> str:
        """ The name is just the fqtable without dots """
        return self.fqtable.replace(".", "_")

    @property
    def path(self) -> str:
        """ Path to parquet cache """
        return os.path.join(DIR_CACHE_TABLES, f"{self.name}.parquet")

    def init_cache_from_csv(self, path: str) -> None:
        """ Create local parquet cache file from csv """
        log.debug(f"Initalising {self.name} from csv at {path}")
        df = io.csv_to_df(path=path)
        df = df.set_index(self.ids).sort_index(axis=0).sort_index(axis=1)
        io.df_to_parquet(df=df, path=self.path)
        log.debug(f"{self.name} now cached in local parquet.")

    def refresh(self) -> pd.DataFrame:
        """ Refetch table from database and update cache """
        log.info(f"Refreshing {self.fqtable}")
        df = db.db_to_df(fqtable=self.fqtable, ids=self.ids)
        io.df_to_parquet(df=df, path=self.path)
        return df

    @property
    def cols(self) -> List[str]:
        """ List columns in table """
        if self._cols:
            cols = self._cols
        else:
            if os.path.isfile(self.path):
                cols = io.list_columns_in_parquet(self.path)
            else:
                cols = db.list_cols(fqtable=self.fqtable)
            self._cols = cols
        return cols

    def get_df_cols(self, cols):
        """ Get a dataframe with a subset of columns

        TODO: Optimise to only read subset of cols from cache
        """
        log.debug(f"Getting {cols} cols from {self.name}")
        if os.path.isfile(self.path):
            df = io.parquet_to_df(path=self.path, cols=cols)
        else:
            df = self.df[cols]
        return df

    @property
    def df(self) -> pd.DataFrame:
        """ Get the dataframe """
        if not os.path.isfile(self.path):
            self.refresh()
        return io.parquet_to_df(self.path)


@dataclass
class Dataset:
    """ Represents a dataset

    Args:
        name: A descriptive name
        ids: Identifier columns, should be 2
        table_skeleton: Table instance of the base table to join into
        tables: List of Tables to join in data from
        loa: Name of level of analysis used to get correct geometry
        transforms: List of Transforms to compute
        balance: Whether to make a balanced index of the dataset
        cols: List of columns to subset tables by

    """

    name: str
    ids: List[str]
    table_skeleton: Table
    tables: List[Table]
    loa: Literal["cm", "pgm", "am"]
    transforms: Optional[List[Transform]] = None
    balance: bool = False
    cols: Optional[List[str]] = None

    @property
    def path(self) -> str:
        """ Path to own parquet cache of df """
        return os.path.join(DIR_CACHE_DATASETS, f"{self.name}.parquet")

    @logtime
    def refresh(self, do_transforms=True) -> pd.DataFrame:
        """ Refetch the dataset from db and compute transformations """
        log.info(f"Refreshing dataset {self.name}")
        # SKeleton df should have many indices to join in all the children
        df = self.table_skeleton.df
        log.debug(f"Dataset {self.name} has index cols {df.index.names}.")

        # if we are subsetting by columns do this per source
        if self.cols:
            for table in self.tables:
                cols_this_table = [
                    col for col in self.cols if col in table.cols
                ]
                # if any cols to get
                if cols_this_table:
                    # Join them in
                    len_pre = len(df)
                    log.debug(
                        f"Joining in {cols_this_table} from table {table.name}"
                    )
                    df = df.join(table.get_df_cols(cols_this_table))
                    if not len_pre == len(df):
                        raise RuntimeError("Joining changed the row count!")
        # Else just get all cols
        else:
            for table in self.tables:
                log.debug(f"Joining in all of {table.name}")
                len_pre = len(df)
                df = df.join(table.df)
                if not len_pre == len(df):
                    raise RuntimeError("Joining changed the row count!")

        # After joining in all the tables set the index to regular panel
        log.debug(f"Reset index {self.name}")
        df.reset_index(inplace=True)
        log.debug(f"Setting index {self.ids} on {self.name}.")
        df.set_index(self.ids, inplace=True)
        log.debug(f"Sorting cols on {self.name}")
        df.sort_index(axis=1, inplace=True)
        log.debug(f"Sorting rows on {self.name}")
        df.sort_index(axis=0, inplace=True)

        # Balance the index if we want
        if self.balance:
            df = datautils.balance_panel_last_t(df)
            # if we extend the index to be balanced we just backfill
            df = missing.extrapolate(df)

        if self.transforms and do_transforms:
            # Transforms might need geometry, so do them on a geodataframe
            log.debug(f"Joining in gdf to {self.name}")
            df = self.just_geometry_gdf.join(df)
            log.debug(
                f"Computing {len(self.transforms)} transforms "
                f"for dataset {self.name}"
            )
            for transform in self.transforms:
                df[transform.name] = np.nan
                df[transform.name] = transform.compute(df)
            # Geodataframes with geometry can't be persisted in parquet
            # So we drop the geometry
            df = df.drop(columns=["geom"])

        # Sort our data again as transforms aren't ordered.
        df = df.sort_index(axis=0).sort_index(axis=1)

        # Store the data
        io.df_to_parquet(df, path=self.path)
        return df

    @property
    def df(self):
        """ Get the datafraem from the dataset """
        if os.path.isfile(self.path):
            df = io.parquet_to_df(self.path)
        else:
            df = self.refresh()

        if self.cols:
            for col in self.cols:
                if col not in df:
                    log.warning(f"Col {col} missing. Not in the sources?")

        return df

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """ Get a GeoDataFrame with geometry and data from self.df"""
        return self.just_geometry_gdf.join(self.df)

    @property
    def just_geometry_gdf(self) -> gpd.GeoDataFrame:
        """ Get a geometry_only gdf """
        if not self.loa:
            raise TypeError(
                f"Dataset {self.name} was instantiated without the loa "
                f"argument. Can't get geometry if  don't know loa"
            )
        geom: Union[GeomCountry, GeomPriogrid]
        if self.loa == "cm":
            geom = GeomCountry()
        elif self.loa == "pgm":
            geom = GeomPriogrid()
        elif self.loa == "am":
            raise NotImplementedError("Actor month geometries not implemented")
        return geom.gdf

    def get_df_cols(self, cols):
        """ Get a subset of cols from cached df """
        if os.path.isfile(self.path):
            log.debug(f"Found df on file for {self.name} at {self.path}")
            return io.parquet_to_df(path=self.path, cols=cols)
        else:
            raise RuntimeError(
                "get_df_cols() called but no cached df, run refresh() first."
            )

    def list_cols_cached(self) -> List[str]:
        """ List cols in cached df """
        if os.path.isfile(self.path):
            return io.list_columns_in_parquet(self.path)
        else:
            raise RuntimeError(
                "list_cols_cached() but no cached df, run refresh() first."
            )
