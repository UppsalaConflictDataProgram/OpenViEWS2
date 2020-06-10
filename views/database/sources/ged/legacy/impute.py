# flake8: noqa
# pylint: skip-file

import os
import numbers

import logging
import requests
import pandas as pd  # type: ignore
from numpy import log1p  # type: ignore

from sqlalchemy import create_engine  # type: ignore
from sqlalchemy.sql import select, column, func, text as alchemy_text  # type: ignore

from views import config
from views.utils import db, io

log = logging.getLogger(__name__)
CONNECTSTRING = config.DATABASES["default"].connectstring


def _geoi_prepare(month_start: int, month_end: int) -> None:
    """Copies the current attached GED for the lookup to work"""
    log.debug("Starting _geoi_prepare()")
    engine = create_engine(CONNECTSTRING)

    db.drop_table("left_imputation.ged")
    db.drop_table("left_imputation.gedfull")
    db.drop_table("left_imputation.toimp4")
    db.drop_table("left_imputation.toimp6")
    with engine.connect() as con:
        trans = con.begin()
        log.debug("Creating left_imputation.ged")
        con.execute(
            """
            CREATE TABLE left_imputation.ged AS
            SELECT id,
                   priogrid_gid,
                   conflict_new_id,
                   month_id_end,
                   month_id_start,
                   type_of_violence,
                   geom,
                   best
            FROM preflight.ged_attached
            """
        )
        log.debug("Creating left_imputation.gedfull")
        con.execute(
            """
            CREATE TABLE left_imputation.gedfull AS
            SELECT id,
                   priogrid_gid,
                   conflict_new_id,
                   month_id_end,
                   month_id_start,
                   type_of_violence,
                   geom,
                   best
            FROM preflight.ged_attached_full
            """
        )
        con.execute(
            "CREATE INDEX lookup_idx ON left_imputation.ged (conflict_new_id, month_id_end)"
        )
        con.execute(
            "CREATE INDEX lookup2_idx ON left_imputation.ged (priogrid_gid)"
        )
        log.debug("Creating left_imputation.toimp4")
        con.execute(
            alchemy_text(
                """
            CREATE TABLE  left_imputation.toimp4
            AS SELECT * FROM preflight.ged_attached_full
            where where_prec=4 AND month_id_end between :m1 and :m2
            """
            ),
            m1=month_start,
            m2=month_end,
        )
        log.debug("Creating left_imputation.toimp6")
        con.execute(
            alchemy_text(
                """
            CREATE TABLE left_imputation.toimp6 AS
            SELECT * FROM preflight.ged_attached_full
            WHERE where_prec=6
            AND month_id_end between :m1 and :m2
            AND geom IS NOT NULL
            """
            ),
            m1=month_start,
            m2=month_end,
        )
        trans.commit()

    log.debug("Finished _geoi_prepare()")


def _fpoint2pgm(
    point, polygon_schema_name="left_imputation", polygon_table_name="gadm1",
):
    """
    :return: a dataframe containing all the GIDs and the AREAS.

    What this does is that it un-fuzzies the fpoint by associating it with all the PGMs that the point may represent.
    A polygon dataset that represents the extent that each point represents (e.g. an ADM dataset for points representing
    administrative divisions or a country dataset for points representing countries) is needed.
    """

    engine = create_engine(CONNECTSTRING)
    with engine.connect() as con:
        query = select([column("gid"), column("area")]).select_from(
            func.geoi_fpoint2poly2pg(
                point, polygon_schema_name, polygon_table_name
            )
        )
        result = pd.read_sql(query, con)
        if result.empty:
            result[0] = [0, 0]
        return result


def _geteventdensity(
    priogrids_df: pd.DataFrame,
    month: int,
    conflict_id: int,
    schema: str,
    table: str,
):
    engine = create_engine(CONNECTSTRING)
    with engine.connect() as con:
        query = alchemy_text(
            f"""
            SELECT priogrid_gid,
            count(*)*100 as density
            FROM {schema}.{table}
            WHERE conflict_new_id = :conflict
            AND month_id_end=:date GROUP BY priogrid_gid
            """
        )
        query = query.bindparams(conflict=conflict_id, date=month)
        density_df = pd.read_sql(query, con)

    df_merge = priogrids_df.merge(
        density_df, left_on="gid", right_on="priogrid_gid", how="left"
    )
    df_merge = df_merge.drop("priogrid_gid", axis=1)
    df_merge = df_merge.fillna(1)
    return df_merge


def _geom_imputation(
    lookup_schema,
    lookup_table,
    adm_table,
    number_imputations,
    point_id,
    point_geom,
    month_start,
    conflict_id,
):
    density = _fpoint2pgm(
        point_geom,
        polygon_schema_name=lookup_schema,
        polygon_table_name=adm_table,
    )
    # print ("!*!*!*!*!*!*!*!*!*!*!*!*!*!*")
    # print (density)
    for past_months in range(0, 13):
        decay = 2 ** ((past_months * -1.0) / 12.0)
        # print(past_months, decay)
        density = _geteventdensity(
            density,
            month=month_start - past_months,
            conflict_id=conflict_id,
            schema=lookup_schema,
            table=lookup_table,
        )
        # print (density)
        if "density_old" in density:
            density["density_old"] = (
                density["density_old"] + density["density"] * decay
            )
        else:
            density["density_old"] = density["density"]
        density = density.drop("density", axis=1)

    sampled_point = density.sample(
        n=number_imputations, replace=True, axis=0, weights="density_old"
    )
    point_id = [point_id]
    point_id.extend(sampled_point["gid"].tolist())
    return pd.DataFrame([point_id])


def _geoi_run(adm1: bool) -> None:

    log.debug("Started _geoi_run()")
    count = 15
    lookup_table = "ged"
    lookup_schema = "left_imputation"

    engine = create_engine(CONNECTSTRING)
    out_df = pd.DataFrame()

    if adm1:
        poly_table = "gadm1"
        table_id = "4"
    else:
        poly_table = "country"
        table_id = "6"

    with engine.connect() as con:
        query = alchemy_text(
            "SELECT count(*) FROM " + lookup_schema + ".toimp" + table_id
        )
        row_count = con.execute(query).fetchone()
        query = alchemy_text(
            "SELECT id,geom_wkt,month_id_end as month_id,conflict_new_id as conflict_id "
            "FROM " + lookup_schema + ".toimp" + table_id
        )
        for row in con.execute(query):
            out_df = out_df.append(
                _geom_imputation(
                    lookup_schema, lookup_table, poly_table, count, *row
                )
            )
        out_df = out_df.rename(columns={0: "id"})
        out_df.to_sql(
            con=con,
            name="geoi_out_" + table_id,
            schema=lookup_schema,
            if_exists="replace",
            index=False,
        )

    log.debug("Finished _geoi_run()")


def _geoi_assemble(month_start: int, month_end: int):

    log.debug("Started _geoi_assemble")

    lookup_schema = "left_imputation"
    lookup_table = "ged"

    engine = create_engine(CONNECTSTRING)
    db.drop_table(fqtable=f"{lookup_schema}.geoi_out_all")

    query = alchemy_text(
        f"""
        CREATE TABLE {lookup_schema}.geoi_out_all AS
        SELECT *
        FROM {lookup_schema}.geoi_out_4
        WHERE "1"<>1
        UNION
        SELECT *
        FROM {lookup_schema}.geoi_out_6
        WHERE "1"<>1
        UNION
        SELECT id,
               priogrid_gid AS "1",
               priogrid_gid AS "2",
               priogrid_gid AS "3",
               priogrid_gid AS "4",
               priogrid_gid AS "5",
               priogrid_gid AS "6",
               priogrid_gid AS "7",
               priogrid_gid AS "8",
               priogrid_gid AS "9",
               priogrid_gid AS "10",
               priogrid_gid AS "11",
               priogrid_gid AS "12",
               priogrid_gid AS "13",
               priogrid_gid AS "14",
               priogrid_gid AS "15"
        FROM {lookup_schema}.{lookup_table}
        WHERE month_id_end BETWEEN :m1 AND :m2;
        """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    query = alchemy_text(
        f"""
        ALTER TABLE {lookup_schema}.geoi_out_all ADD COLUMN month_start BIGINT;
        ALTER TABLE {lookup_schema}.geoi_out_all ADD COLUMN month_end BIGINT;
        ALTER TABLE {lookup_schema}.geoi_out_all ADD COLUMN type_of_violence BIGINT;
        """
    )
    with engine.connect() as con:
        con.execute(query)

    query = alchemy_text(
        f"""
        UPDATE {lookup_schema}.geoi_out_all SET
        month_start=c.month_id_start,
        month_end=c.month_id_end,
        type_of_violence=c.type_of_violence
        FROM {lookup_schema}.gedfull c
        WHERE {lookup_schema}.geoi_out_all.id = c.id;
        """
    )
    with engine.connect() as con:
        con.execute(query)
    log.debug("Finished _geoi_assemble")


def _geoi_pgm_dummy_update(
    imputation_id: int, month_start: int, month_end: int,
):
    log.debug("Started _geoi_pgm_dummy_update().")
    if not isinstance(imputation_id, numbers.Integral):
        raise RuntimeError(f"{imputation_id} isn't a numbers.Integral")

    lookup_schema = "left_imputation"

    # imputation_id = str(imputation_id)
    engine = create_engine(CONNECTSTRING)
    with engine.connect() as con:
        trans = con.begin()
        query = alchemy_text(
            f"""
            UPDATE {lookup_schema}.pgm
            SET
            ged_sb_dummy_{imputation_id} = 0,
            ged_ns_dummy_{imputation_id} = 0,
            ged_os_dummy_{imputation_id} = 0
            WHERE month_id BETWEEN :m1 AND :m2
            """
        )
        con.execute(query, m1=month_start, m2=month_end)

        trans.commit()
        trans = con.begin()
        query = alchemy_text(
            f"""
            WITH a AS
            (
                SELECT
                "{imputation_id}" as priogrid_gid,
                type_of_violence,
                random_series_int(month_start :: INT, month_end :: INT + 1) AS month_id
                FROM  {lookup_schema}.geoi_out_all
            )
            UPDATE {lookup_schema}.pgm as i SET ged_ns_dummy_{imputation_id}=1
            FROM a
            WHERE a.type_of_violence=2 AND a.priogrid_gid=i.priogrid_gid AND a.month_id=i.month_id
            """
        )
        con.execute(query)

        query = alchemy_text(
            f"""
            WITH a AS
            (
                SELECT
                "{imputation_id}" as priogrid_gid,
                type_of_violence,
                random_series_int(month_start :: INT, month_end :: INT + 1) AS month_id
                FROM  {lookup_schema}.geoi_out_all
            )
            UPDATE {lookup_schema}.pgm as i SET ged_os_dummy_{imputation_id}=1
            FROM a
            WHERE a.type_of_violence=3 AND a.priogrid_gid=i.priogrid_gid AND a.month_id=i.month_id
            """
        )
        con.execute(query)

        query = alchemy_text(
            f"""
            WITH a AS
            (
                SELECT
                "{imputation_id}" as priogrid_gid,
                type_of_violence,
                random_series_int(month_start :: INT, month_end :: INT + 1) AS month_id
                FROM  {lookup_schema}.geoi_out_all
            )
            UPDATE {lookup_schema}.pgm as i SET ged_sb_dummy_{imputation_id}=1
            FROM a
            WHERE a.type_of_violence=1 AND a.priogrid_gid=i.priogrid_gid AND a.month_id=i.month_id
            """
        )
        con.execute(query)
        trans.commit()

    log.debug("Finished _geoi_pgm_dummy_update().")


def _geoi_build_dummies(n_imp: int, month_start: int, month_end: int):
    for i in range(1, n_imp + 1):
        _geoi_pgm_dummy_update(
            imputation_id=i, month_start=month_start, month_end=month_end,
        )
