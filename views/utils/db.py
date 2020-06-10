""" Database utilites """
from io import StringIO
import tempfile
from typing import Any, Dict, List, Tuple, Optional
import csv
import logging
import time

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sqlalchemy  # type: ignore
import geopandas as gpd  # type: ignore
import geoalchemy2 as ga  # type: ignore

from views import config
from views.utils.log import logtime

log = logging.getLogger(__name__)


def _unpack_fqtable(fqtable: str) -> Tuple[str, str]:
    """ returns schema, table from "schema.table" """

    if not fqtable.count(".") == 1:
        raise RuntimeError(f"Bad fqtable {fqtable} should be schema.table")

    schema, table = fqtable.split(".")
    return schema.lower(), table.lower()


def _make_engine(db_name: str = "default"):

    db = config.DATABASES[db_name]
    log.debug(f"Connecting to db: {db}")
    engine = sqlalchemy.create_engine(
        db.connectstring, connect_args=db.connect_args
    )
    return engine


def _db_to_df_fast(fqtable, cols: List[str], db_name: str):
    """ Read fqtable from db using COPY to tempfile for speed

    Args:
        fqtable: fully qualifed tablename like schemaname.tablename
        cols: list of cols to fetch, fetches all if None
        ids: set index using these colums, default index if None
    Returns:
        df:
    """

    def read_sql_tmpfile(query, db_name):

        db_engine = _make_engine(db_name)
        with tempfile.TemporaryFile() as tmpfile:
            copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
                query=query, head="HEADER"
            )
            conn = db_engine.raw_connection()
            cur = conn.cursor()
            cur.copy_expert(copy_sql, tmpfile)
            tmpfile.seek(0)
            log.debug("Fininshed writing tempfile")
            df = pd.read_csv(tmpfile)
            log.debug("Fininshed reading tempfile into df")
        return df

    log.debug(f"Fetching {len(cols)} cols from {fqtable} fast.")

    comma_separated_cols = ", ".join(cols)
    query = f"SELECT {comma_separated_cols} FROM {fqtable}"
    df = read_sql_tmpfile(query, db_name)

    return df


@logtime
def db_to_df(
    fqtable: str,
    cols: Optional[List[str]] = None,
    ids: Optional[List[str]] = None,
    db_name: str = "default",
    fast=True,
) -> pd.DataFrame:
    """ Read a database table to a pandas dataframe """

    t_start = time.time()
    extras = ""
    if isinstance(cols, list):
        extras += f" with {len(cols)} cols"
    elif cols is None:
        cols = list_cols(fqtable, db_name)
    if ids:
        extras += f" with ids {ids}"
    log.info(f"Fetching {fqtable} {extras}.")

    if ids and cols:
        for id_col in ids:
            if id_col not in cols:
                cols.append(id_col)

    schema, table = _unpack_fqtable(fqtable)
    if fast:
        df = _db_to_df_fast(fqtable, cols, db_name)
    else:
        df = pd.read_sql_table(
            table_name=table,
            schema=schema,
            con=_make_engine(db_name),
            columns=cols,
        )
    t_fetch = time.time() - t_start

    if ids:
        df.set_index(ids, inplace=True)
        df.sort_index(inplace=True)

    log.debug(f"Fetched {fqtable} with shape {df.shape} in {t_fetch}s.")

    return df


@logtime
def query_to_df(query: str, db_name: str = "default") -> pd.DataFrame:
    """ Get the results of query as dataframe """
    log.debug(f"Getting results of query to df: {query}")
    return pd.read_sql_query(sql=query, con=_make_engine(db_name))


@logtime
def list_cols(fqtable: str, db_name: str = "default") -> List[str]:
    """ Get list of all columns in a table """
    log.debug(f"Listing cols in {fqtable}")
    return list(
        query_to_df(
            f"SELECT * FROM {fqtable} LIMIT 1;", db_name=db_name
        ).columns
    )


@logtime
def _df_to_db_fast(
    df: pd.DataFrame,
    fqtable: str,
    if_exists: str,
    write_index: bool,
    db_name: str = "default",
) -> None:
    """ Push df to db quickly using postgres COPY and a csv buffer.

    See
    https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#insertion-method

    """

    def psql_insert_copy(table, conn, keys, data_iter):
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        table : pandas.io.sql.SQLTable
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
        keys : list of str
            Column names
        data_iter : Iterable that iterates the values to be inserted
        """
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ", ".join('"{}"'.format(k) for k in keys)
            if table.schema:
                table_name = "{}.{}".format(table.schema, table.name)
            else:
                table_name = table.name

            sql = "COPY {} ({}) FROM STDIN WITH CSV".format(
                table_name, columns
            )
            cur.copy_expert(sql=sql, file=s_buf)

    def get_sqlalchemy_types(df: pd.DataFrame) -> Dict[str, Any]:
        """Matches pandas datatypes to their sqlalchemy equivalents,
        returns the mapping as a dict"""

        dtypedict = {}
        for col, dtype in zip(df.columns, df.dtypes):
            if "object" in str(dtype):
                dtypedict.update({col: sqlalchemy.types.VARCHAR()})

            if "datetime" in str(dtype):
                dtypedict.update({col: sqlalchemy.types.DateTime()})

            # Underflow errors can happen when precision is set
            if "float" in str(dtype):
                dtypedict.update({col: sqlalchemy.types.Float(asdecimal=True)})

            if "int" in str(dtype):
                dtypedict.update({col: sqlalchemy.types.INT()})

        return dtypedict

    log.debug("Using _df_to_db_fast()")
    schema, table = _unpack_fqtable(fqtable)

    df.to_sql(
        name=table,
        con=_make_engine(db_name),
        if_exists=if_exists,
        schema=schema,
        index=write_index,
        dtype=get_sqlalchemy_types(df),
        chunksize=10000,
        method=psql_insert_copy,
    )


@logtime
def _df_to_db_standard(
    df: pd.DataFrame,
    fqtable: str,
    if_exists: str,
    write_index: bool,
    db_name: str = "default",
) -> None:
    log.debug("Using _df_to_db_standard()")
    schema, table = _unpack_fqtable(fqtable)
    df.to_sql(
        name=table,
        con=_make_engine(db_name),
        if_exists=if_exists,
        schema=schema,
        index=write_index,
        chunksize=10000,
    )


def sanitise_cols(df):
    """ Sanitise column names to remove case and dots """
    return df.rename(columns=lambda col: col.lower().replace(".", "_"))


@logtime
def df_to_db(
    df: pd.DataFrame,
    fqtable: str,
    write_index: bool = True,
    fast: bool = True,
    db_name: str = "default",
) -> None:
    """ Write a pandas dataframe to a database table """

    if_exists = "replace"
    log.debug(f"Pushing df with shape {df.shape} to {fqtable} to db {db_name}")
    t_start = time.time()
    df = sanitise_cols(df)
    if fast:
        _df_to_db_fast(df, fqtable, if_exists, write_index, db_name)
    else:
        _df_to_db_standard(df, fqtable, if_exists, write_index, db_name)
    log.debug(f"Pushed df with shape {df.shape} in {time.time() - t_start}s")


@logtime
def db_to_gdf(
    fqtable: str,
    cols: Optional[List[str]] = None,
    ids: Optional[List[str]] = None,
    geom_col: str = "geom",
    db_name: str = "default",
) -> gpd.GeoDataFrame:
    """ Read geopandas dataframe from query and assign groupvar index

    Ex:
        gdf_geom = db_to_gdf(
        query="SELECT pg_id, geom FROM staging.priogrid",
        groupvar="pg_id"
        )

    """

    if not isinstance(cols, list):
        cols = list_cols(fqtable, db_name)

    if not ids:
        ids = []

    if isinstance(ids, list):
        for id_col in ids:
            if id_col not in cols:
                cols.append(id_col)

    if geom_col not in cols:
        cols.append(geom_col)

    cols_str = ", ".join(cols)

    query = f"SELECT {cols_str} FROM {fqtable};"

    # Log nice
    extras = ""
    extras += f" with {len(cols)} cols"
    if ids:
        extras += f" with ids {ids}"
    extras += f" with geom_col {geom_col}"
    log.info(f"Fetching {fqtable} {extras} to geodataframe.")

    gdf = gpd.read_postgis(
        sql=query, con=_make_engine(db_name), geom_col=geom_col
    )

    if ids:
        gdf.set_index(ids, inplace=True)
        gdf.sort_index(inplace=True)

    log.debug("Finished fetching geometry")

    return gdf


def gdf_to_db(
    gdf: pd.DataFrame,
    fqtable: str,
    write_index: bool = True,
    geom_type: str = "POLYGON",
    db_name: str = "default",
) -> None:
    """ Push a Polygon geometry geodataframe to PostGIS DB

    Honestly I don't truly understand how this works but I verified
    the values that get stored in the DB and they're identical, type
    and value wise.

    @TODO: See if we can generalise this and put in df_to_db() instead

    """

    ga2_geom_types = [
        "GEOMETRY",
        "POINT",
        "LINESTRING",
        "POLYGON",
        "MULTIPOINT",
        "MULTILINESTRING",
        "MULTIPOLYGON",
        "GEOMETRYCOLLECTION",
        "CURVE",
    ]

    geom_type = geom_type.upper()

    if geom_type not in ga2_geom_types:
        raise TypeError(
            f"Unknow geom_type {geom_type}, only {ga2_geom_types} should work."
        )

    def to_wkt(geom):
        """ Use with apply to make Well-Known-Text objects of geom """
        return ga.WKTElement(geom.wkt, srid=4326)

    # Make a copy so we don't mess with the original
    this_gdf = gdf.copy()
    # Cast geom to a WKT
    this_gdf["geom"] = this_gdf["geom"].apply(to_wkt)
    dtype = {"geom": ga.Geometry(geom_type, srid=4326)}

    schema, table = _unpack_fqtable(fqtable)

    this_gdf.to_sql(
        name=table,
        con=_make_engine(db_name),
        if_exists="replace",
        schema=schema,
        index=write_index,
        chunksize=10000,
        dtype=dtype,
    )


def df_to_split_db(
    df: pd.DataFrame, fqtables: List[str], db_name="default"
) -> None:
    """ Split df by columns into multiple fqtables """
    colsets = [
        list(subset)
        for subset in np.array_split(
            ary=df.columns, indices_or_sections=len(fqtables)
        )
    ]

    for fqtable, cols in zip(fqtables, colsets):
        df_to_db(df=df[cols], fqtable=fqtable, db_name=db_name)


def create_schema(schema: str, db_name="default") -> None:
    """ Create a new schema if it doesn exist """
    query = f"CREATE SCHEMA IF NOT EXISTS {schema};"
    _ = execute_query(query, db_name)


def drop_schema(schema: str, db_name="default") -> None:
    """ Drop a schema if it exists """
    query = f"DROP SCHEMA IF EXISTS {schema} CASCADE"
    _ = execute_query(query, db_name)


def drop_table(fqtable: str, db_name="default") -> None:
    """ Drop a table """
    query = f"DROP TABLE IF EXISTS {fqtable} CASCADE"
    _ = execute_query(query, db_name=db_name)


def execute_query(query: str, db_name: str = "default") -> Any:
    """ Execute a query """

    log.debug(f"Executing query {query}")

    engine = _make_engine(db_name=db_name)

    query = sqlalchemy.sql.text(query)

    # engine.begin() rolls the entire query execution into a transaction block
    # as opposed to engine.connect()
    # engine.begin() saves us from manually placing COMMIT in our queries.
    # I'm note 100% on the differences here
    with engine.begin() as con:
        response = con.execute(query)

    return response
