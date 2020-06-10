# flake8: noqa
# pylint: skip-file

import os
import logging
import requests
import pandas as pd  # type: ignore
from sqlalchemy import create_engine  # type: ignore
from sqlalchemy.sql import select, column, func, text as alchemy_text  # type: ignore

from views import config
from views.utils import db, io

from .impute import (
    _geoi_prepare,
    _geoi_run,
    _geoi_assemble,
    _geoi_build_dummies,
)

log = logging.getLogger(__name__)
CONNECTSTRING = config.DATABASES["default"].connectstring


def _get_ged_slice(next_page_url: str):
    log.debug(f"Fetching GED slice from {next_page_url}")
    r = requests.get(next_page_url)
    output = r.json()
    next_page_url = (
        output["NextPageUrl"] if output["NextPageUrl"] != "" else None
    )
    ged = pd.DataFrame(output["Result"])
    page_count = output["TotalPages"]
    return next_page_url, ged, page_count


def _check_ged(month_start: int, api_version: str) -> None:
    """ Check that GED has data for month_start """

    # Imputation needs 12 months, check that we have that in the API?
    month_check = month_start - 11
    log.debug(f"Checking GED has data for {month_check}")

    engine = create_engine(CONNECTSTRING)

    with engine.connect() as con:
        query = alchemy_text(
            "SELECT month, year_id FROM staging.month WHERE id=:sd"
        )
        result = con.execute(query, sd=month_check).fetchone()
        iso_start_check = "{1:04d}-{0:02d}-01".format(*result)
        iso_end_check = "{1:04d}-{0:02d}-25".format(*result)
        next_page_url = (
            "http://ucdpapi.pcr.uu.se/api/gedevents/"
            + api_version
            + "?pagesize=1&StartDate="
            + iso_start_check
            + "&EndDate="
            + iso_end_check
        )
        next_page_url, ged_slice, total_pages = _get_ged_slice(
            next_page_url=next_page_url
        )

        if total_pages > 0:
            log.debug(f"GED data exists for {month_check}")
        else:
            raise RuntimeError(
                f"No data in api version {api_version} for month {month_check}"
            )


def _get_ged(api_version: str) -> None:
    """ Get GED from API into dataprep.ged """

    fqtable = "dataprep.ged"

    log.debug(
        "Started fetching GED from "
        f"API version {api_version} into {fqtable}"
    )

    cur_page = 1
    next_page_url = (
        "http://ucdpapi.pcr.uu.se/api/gedevents/"
        + api_version
        + "?pagesize=1000"
    )
    df = pd.DataFrame()
    while next_page_url:
        next_page_url, ged_slice, total_pages = _get_ged_slice(
            next_page_url=next_page_url
        )
        df = df.append(ged_slice, ignore_index=True)
        log.debug(f"{cur_page} of {total_pages} pages loaded.")
        cur_page += 1

    log.debug(f"Got all pages. Inserting into db at {fqtable}")
    db.df_to_db(df=df, fqtable=fqtable)
    log.debug(f"Done loading {len(df)} rows into {fqtable}.")


def _prepare_ged() -> None:
    """ Recreates preflight.ged_attached and preflight.ged_attached_full """

    # Moved into .sql file in this dir.
    log.debug(f"Preparing preflight.ged_attached(_full)")
    query = io.read_file(
        path=os.path.join(os.path.dirname(__file__), "prepare_ged.sql")
    )
    db.execute_query(query)
    log.debug(f"Done preflight.ged_attached(_full)")


def _stage_ged_2_pgm(month_start: int, month_end: int) -> None:
    """ Update staging.priogrid_month """
    log.debug(
        "Starting _stage_ged_2_pgm with "
        f"month_start: {month_start} month_end: {month_end}"
    )
    engine = create_engine(CONNECTSTRING)

    log.debug("Finding limits ")
    with engine.connect() as con:
        limits = con.execute(
            "SELECT min(month_id_end) AS int, max(month_id_end) AS int FROM preflight.ged_attached"
        ).fetchone()
        if limits[0] > month_start:
            month_start = limits[0]
        if limits[1] <= month_end:
            month_end = limits[1]
    log.debug(f"Limits of preflight.ged_attached are {limits}")

    log.debug(f"Updating staging.priogrid_month sums.")
    query = alchemy_text(
        """
        UPDATE staging.priogrid_month
        SET ged_best_sb        = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 0, 1),
            ged_best_ns        = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 0, 2),
            ged_best_os        = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 0, 3),
            ged_count_sb       = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 0, 1),
            ged_count_ns       = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 0, 2),
            ged_count_os       = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 0, 3),
            ged_best_sb_start  = aggregate_deaths_on_date_start(priogrid_gid, month_id, FALSE, FALSE, 0, 1),
            ged_best_ns_start  = aggregate_deaths_on_date_start(priogrid_gid, month_id, FALSE, FALSE, 0, 2),
            ged_best_os_start  = aggregate_deaths_on_date_start(priogrid_gid, month_id, FALSE, FALSE, 0, 3),
            ged_count_sb_start = aggregate_deaths_on_date_start(priogrid_gid, month_id, TRUE, FALSE, 0, 1),
            ged_count_ns_start = aggregate_deaths_on_date_start(priogrid_gid, month_id, TRUE, FALSE, 0, 2),
            ged_count_os_start = aggregate_deaths_on_date_start(priogrid_gid, month_id, TRUE, FALSE, 0, 3),
            ged_best_sb_lag1   = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 1, 1),
            ged_best_ns_lag1   = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 1, 2),
            ged_best_os_lag1   = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 1, 3),
            ged_count_sb_lag1  = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 1, 1),
            ged_count_ns_lag1  = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 1, 2),
            ged_count_os_lag1  = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 1, 3),
            ged_best_sb_lag2   = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 2, 1),
            ged_best_ns_lag2   = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 2, 2),
            ged_best_os_lag2   = aggregate_deaths_on_date_end(priogrid_gid, month_id, FALSE, FALSE, 2, 3),
            ged_count_sb_lag2  = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 2, 1),
            ged_count_ns_lag2  = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 2, 2),
            ged_count_os_lag2  = aggregate_deaths_on_date_end(priogrid_gid, month_id, TRUE, FALSE, 2, 3)
        WHERE month_id >= :m1
          AND month_id <= :m2
        """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Creating temporal lags.")
    query = alchemy_text(
        """
        SELECT public.make_priogrid_month_temporal_lags(1,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(2,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(3,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(4,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(5,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(6,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(7,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(8,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(9,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(10,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(11,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_temporal_lags(12,FALSE,:m1,:m2);
    """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Updating count variables.")
    query = alchemy_text(
        """
        UPDATE staging.priogrid_month SET
        ged_months_since_last_sb = public.months_since_last_event('ged_count_sb', priogrid_gid, month_id),
        ged_months_since_last_ns = public.months_since_last_event('ged_count_ns', priogrid_gid, month_id),
        ged_months_since_last_os = public.months_since_last_event('ged_count_os', priogrid_gid, month_id),
        ged_months_since_last_sb_lag1 = public.months_since_last_event('ged_count_sb_lag1', priogrid_gid, month_id),
        ged_months_since_last_ns_lag1 = public.months_since_last_event('ged_count_ns_lag1', priogrid_gid, month_id),
        ged_months_since_last_os_lag1 = public.months_since_last_event('ged_count_os_lag1', priogrid_gid, month_id),
        ged_months_since_last_sb_lag2 = public.months_since_last_event('ged_count_sb_lag2', priogrid_gid, month_id),
        ged_months_since_last_ns_lag2 = public.months_since_last_event('ged_count_ns_lag2', priogrid_gid, month_id),
        ged_months_since_last_os_lag2 = public.months_since_last_event('ged_count_os_lag2', priogrid_gid, month_id)
        WHERE month_id BETWEEN :m1 AND :m2
        """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Updating monthly counts for thresholds.")
    query = alchemy_text(
        """
        UPDATE staging.priogrid_month SET
        ged_months_since_last_sb_tx5 = public.months_since_last_event_threshold('ged_best_sb', priogrid_gid, month_id,5),
        ged_months_since_last_ns_tx5 = public.months_since_last_event_threshold('ged_best_ns', priogrid_gid, month_id,5),
        ged_months_since_last_os_tx5 = public.months_since_last_event_threshold('ged_best_os', priogrid_gid, month_id,5),
        ged_months_since_last_sb_lag1_tx5 = public.months_since_last_event_threshold('ged_best_sb_lag1', priogrid_gid, month_id,5),
        ged_months_since_last_ns_lag1_tx5 = public.months_since_last_event_threshold('ged_best_ns_lag1', priogrid_gid, month_id,5),
        ged_months_since_last_os_lag1_tx5 = public.months_since_last_event_threshold('ged_best_os_lag1', priogrid_gid, month_id,5),
        ged_months_since_last_sb_tx25 = public.months_since_last_event_threshold('ged_best_sb', priogrid_gid, month_id,25),
        ged_months_since_last_ns_tx25 = public.months_since_last_event_threshold('ged_best_ns', priogrid_gid, month_id,25),
        ged_months_since_last_os_tx25 = public.months_since_last_event_threshold('ged_best_os', priogrid_gid, month_id,25),
        ged_months_since_last_sb_lag1_tx25 = public.months_since_last_event_threshold('ged_best_sb_lag1', priogrid_gid, month_id,25),
        ged_months_since_last_ns_lag1_tx25 = public.months_since_last_event_threshold('ged_best_ns_lag1', priogrid_gid, month_id,25),
        ged_months_since_last_os_lag1_tx25 = public.months_since_last_event_threshold('ged_best_os_lag1', priogrid_gid, month_id,25),
        ged_months_since_last_sb_tx100 = public.months_since_last_event_threshold('ged_best_sb', priogrid_gid, month_id,100),
        ged_months_since_last_ns_tx100 = public.months_since_last_event_threshold('ged_best_ns', priogrid_gid, month_id,100),
        ged_months_since_last_os_tx100 = public.months_since_last_event_threshold('ged_best_os', priogrid_gid, month_id,100),
        ged_months_since_last_sb_lag1_tx100 = public.months_since_last_event_threshold('ged_best_sb_lag1', priogrid_gid, month_id,100),
        ged_months_since_last_ns_lag1_tx100 = public.months_since_last_event_threshold('ged_best_ns_lag1', priogrid_gid, month_id,100),
        ged_months_since_last_os_lag1_tx100 = public.months_since_last_event_threshold('ged_best_os_lag1', priogrid_gid, month_id,100),
        ged_months_since_last_sb_tx500 = public.months_since_last_event_threshold('ged_best_sb', priogrid_gid, month_id,500),
        ged_months_since_last_ns_tx500 = public.months_since_last_event_threshold('ged_best_ns', priogrid_gid, month_id,500),
        ged_months_since_last_os_tx500 = public.months_since_last_event_threshold('ged_best_os', priogrid_gid, month_id,500),
        ged_months_since_last_sb_lag1_tx500 = public.months_since_last_event_threshold('ged_best_sb_lag1', priogrid_gid, month_id,500),
        ged_months_since_last_ns_lag1_tx500 = public.months_since_last_event_threshold('ged_best_ns_lag1', priogrid_gid, month_id,500),
        ged_months_since_last_os_lag1_tx500 = public.months_since_last_event_threshold('ged_best_os_lag1', priogrid_gid, month_id,500)
        WHERE month_id BETWEEN :m1 AND :m2
        """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Compute spatial distances.")
    query = alchemy_text(
        """
        UPDATE staging.priogrid_month SET
        dist_ged_sb_event=public.distance_to_nearest_ged('preflight','ged_attached',priogrid_gid,month_id,1),
        dist_ged_ns_event=public.distance_to_nearest_ged('preflight','ged_attached',priogrid_gid,month_id,2),
        dist_ged_os_event=public.distance_to_nearest_ged('preflight','ged_attached',priogrid_gid,month_id,3)
        WHERE month_id BETWEEN :m1 AND :m2
        """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Updating base onset variables for PGM.")
    query = alchemy_text(
        """
        WITH a AS (SELECT * FROM onset_months_table('staging', 'priogrid_month', 'ged_best_sb'))
        UPDATE staging.priogrid_month
        SET onset_months_sb = a.onset_distance
        FROM a
        WHERE a.id = staging.priogrid_month.id;

        WITH a AS (SELECT * FROM onset_months_table('staging', 'priogrid_month', 'ged_best_ns'))
        UPDATE staging.priogrid_month
        SET onset_months_ns = a.onset_distance
        FROM a
        WHERE a.id = staging.priogrid_month.id;

        WITH a AS (SELECT * FROM onset_months_table('staging', 'priogrid_month', 'ged_best_os'))
        UPDATE staging.priogrid_month
        SET onset_months_os = a.onset_distance
        FROM a
        WHERE a.id = staging.priogrid_month.id;
        """
    )
    with engine.connect() as con:
        trans = con.begin()
        con.execute(query)
        trans.commit()

    log.debug("Updating onsets based on spatial lags at PGM.")
    query = alchemy_text(
        """
        UPDATE staging.priogrid_month
        SET onset_month_sb_lag1 =
                onset_lags(
                        priogrid := priogrid_gid,
                        month_id := month_id,
                        lags := 1,
                        schema_name := 'staging'::varchar,
                        table_name := 'priogrid_month'::varchar,
                        column_name := 'ged_best_sb'::varchar),

            onset_month_sb_lag2 =
                onset_lags(
                        priogrid := priogrid_gid,
                        month_id := month_id,
                        lags := 2,
                        schema_name := 'staging'::varchar,
                        table_name := 'priogrid_month'::varchar,
                        column_name := 'ged_best_sb'::varchar)
        WHERE onset_months_sb > 0
          AND month_id BETWEEN :m1 AND :m2;


        UPDATE staging.priogrid_month
        SET onset_month_ns_lag1 =
                onset_lags(
                        priogrid := priogrid_gid,
                        month_id := month_id,
                        lags := 1,
                        schema_name := 'staging'::varchar,
                        table_name := 'priogrid_month'::varchar,
                        column_name := 'ged_best_ns'::varchar),

            onset_month_ns_lag2 =
                onset_lags(
                        priogrid := priogrid_gid,
                        month_id := month_id,
                        lags := 2,
                        schema_name := 'staging'::varchar,
                        table_name := 'priogrid_month'::varchar,
                        column_name := 'ged_best_ns'::varchar)
        WHERE onset_months_ns > 0
          AND month_id BETWEEN :m1 AND :m2;

        UPDATE staging.priogrid_month
        SET onset_month_os_lag1 =
                onset_lags(
                        priogrid := priogrid_gid,
                        month_id := month_id,
                        lags := 1,
                        schema_name := 'staging'::varchar,
                        table_name := 'priogrid_month'::varchar,
                        column_name := 'ged_best_os'::varchar),

            onset_month_os_lag2 =
                onset_lags(
                        priogrid := priogrid_gid,
                        month_id := month_id,
                        lags := 2,
                        schema_name := 'staging'::varchar,
                        table_name := 'priogrid_month'::varchar,
                        column_name := 'ged_best_os'::varchar)
        WHERE onset_months_os > 0
          AND month_id BETWEEN :m1 AND :m2;
    """
    )
    with engine.connect() as con:
        trans = con.begin()
        con.execute(query, m1=month_start, m2=month_end)
        trans.commit()

    log.debug("Updating windowed onsets.")
    query = alchemy_text(
        """
    WITH a AS
             (
                 SELECT id,
                        max(ged_best_sb)
                        OVER (PARTITION BY priogrid_gid ORDER BY month_id ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS sb_x1,
                        max(ged_best_os)
                        OVER (PARTITION BY priogrid_gid ORDER BY month_id ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS os_x1,
                        max(ged_best_ns)
                        OVER (PARTITION BY priogrid_gid ORDER BY month_id ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS ns_x1
                 FROM staging.priogrid_month
                 WHERE month_id BETWEEN :m1 - 48 AND :m2
             )
    UPDATE staging.priogrid_month
    SET max24_best_sb = a.sb_x1,
        max24_best_ns = a.ns_x1,
        max24_best_os = a.os_x1
    FROM a
    WHERE a.id = staging.priogrid_month.id
      AND month_id BETWEEN :m1 AND :m2
    """
    )
    with engine.connect() as con:
        trans = con.begin()
        con.execute(query, m1=month_start, m2=month_end)
        trans.commit()

    log.info("Finished _stage_ged_2_pgm()")


def _stage_ged_2_cm(month_start: int, month_end: int) -> None:
    """ Update staging.country_month """

    log.debug(
        "Starting _stage_ged_2_cm with "
        f"month_start: {month_start} month_end: {month_end}"
    )
    engine = create_engine(CONNECTSTRING)

    log.debug("Finding limits ")
    with engine.connect() as con:
        limits = con.execute(
            "SELECT min(month_id_end) AS int, max(month_id_end) AS int FROM preflight.ged_attached"
        ).fetchone()
        if limits[0] > month_start:
            month_start = limits[0]
        if limits[1] <= month_end:
            month_end = limits[1]
    log.debug(f"Limits of preflight.ged_attached are {limits}")

    log.debug("Staging CM with base GED variables and spatial lags.")
    query = alchemy_text(
        """
        UPDATE staging.country_month SET
        ged_best_sb = public.aggregate_cm_deaths_on_date_end(id,FALSE,FALSE,0,1),
        ged_best_ns = public.aggregate_cm_deaths_on_date_end(id,FALSE,FALSE,0,2),
        ged_best_os = public.aggregate_cm_deaths_on_date_end(id,FALSE,FALSE,0,3),
        ged_count_sb = public.aggregate_cm_deaths_on_date_end(id,TRUE,FALSE,0,1),
        ged_count_ns = public.aggregate_cm_deaths_on_date_end(id,TRUE,FALSE,0,2),
        ged_count_os = public.aggregate_cm_deaths_on_date_end(id,TRUE,FALSE,0,3),
        ged_best_sb_lag1 = public.aggregate_cm_deaths_on_date_end(id,FALSE,FALSE,1,1),
        ged_best_ns_lag1 = public.aggregate_cm_deaths_on_date_end(id,FALSE,FALSE,1,2),
        ged_best_os_lag1 = public.aggregate_cm_deaths_on_date_end(id,FALSE,FALSE,1,3),
        ged_count_sb_lag1 = public.aggregate_cm_deaths_on_date_end(id,TRUE,FALSE,1,1),
        ged_count_ns_lag1 = public.aggregate_cm_deaths_on_date_end(id,TRUE,FALSE,1,2),
        ged_count_os_lag1 = public.aggregate_cm_deaths_on_date_end(id,TRUE,FALSE,1,3)
        WHERE month_id BETWEEN :m1 AND :m2
        """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Staging Country-Month t-lags.")
    query = alchemy_text(
        """
        SELECT public.make_country_month_temporal_lags(1,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(2,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(3,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(4,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(5,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(6,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(7,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(8,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(9,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(10,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(11,FALSE,109,:m2);
        SELECT public.make_country_month_temporal_lags(12,FALSE,109,:m2);
        """
    )
    with engine.connect() as con:
        con.execute(query, m2=month_end)

    log.debug("Staging Country-Month distances to nearest event...")
    query = alchemy_text(
        """
        UPDATE staging.country_month SET
        ged_months_since_last_sb = public.cm_months_since_last_event('ged_count_sb', country_id, month_id),
        ged_months_since_last_ns = public.cm_months_since_last_event('ged_count_ns', country_id, month_id),
        ged_months_since_last_os = public.cm_months_since_last_event('ged_count_os', country_id, month_id),
        ged_months_since_last_sb_lag1 = public.cm_months_since_last_event('ged_count_sb_lag1', country_id, month_id),
        ged_months_since_last_ns_lag1 = public.cm_months_since_last_event('ged_count_ns_lag1', country_id, month_id),
        ged_months_since_last_os_lag1 = public.cm_months_since_last_event('ged_count_os_lag1', country_id, month_id)
        WHERE month_id BETWEEN :m1 AND :m2
        """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Computing months_since_last")
    query = alchemy_text(
        """
        UPDATE staging.country_month SET
        ged_months_since_last_sb_tx5 = public.cm_months_since_last_event_threshold('ged_best_sb', country_id, month_id,5),
        ged_months_since_last_ns_tx5 = public.cm_months_since_last_event_threshold('ged_best_ns', country_id, month_id,5),
        ged_months_since_last_os_tx5 = public.cm_months_since_last_event_threshold('ged_best_os', country_id, month_id,5),
        ged_months_since_last_sb_lag1_tx5 = public.cm_months_since_last_event_threshold('ged_best_sb_lag1', country_id, month_id,5),
        ged_months_since_last_ns_lag1_tx5 = public.cm_months_since_last_event_threshold('ged_best_ns_lag1', country_id, month_id,5),
        ged_months_since_last_os_lag1_tx5 = public.cm_months_since_last_event_threshold('ged_best_os_lag1', country_id, month_id,5),
        ged_months_since_last_sb_tx25 = public.cm_months_since_last_event_threshold('ged_best_sb', country_id, month_id,25),
        ged_months_since_last_ns_tx25 = public.cm_months_since_last_event_threshold('ged_best_ns', country_id, month_id,25),
        ged_months_since_last_os_tx25 = public.cm_months_since_last_event_threshold('ged_best_os', country_id, month_id,25),
        ged_months_since_last_sb_lag1_tx25 = public.cm_months_since_last_event_threshold('ged_best_sb_lag1', country_id, month_id,25),
        ged_months_since_last_ns_lag1_tx25 = public.cm_months_since_last_event_threshold('ged_best_ns_lag1', country_id, month_id,25),
        ged_months_since_last_os_lag1_tx25 = public.cm_months_since_last_event_threshold('ged_best_os_lag1', country_id, month_id,25),
        ged_months_since_last_sb_tx100 = public.cm_months_since_last_event_threshold('ged_best_sb', country_id, month_id,100),
        ged_months_since_last_ns_tx100 = public.cm_months_since_last_event_threshold('ged_best_ns', country_id, month_id,100),
        ged_months_since_last_os_tx100 = public.cm_months_since_last_event_threshold('ged_best_os', country_id, month_id,100),
        ged_months_since_last_sb_lag1_tx100 = public.cm_months_since_last_event_threshold('ged_best_sb_lag1', country_id, month_id,100),
        ged_months_since_last_ns_lag1_tx100 = public.cm_months_since_last_event_threshold('ged_best_ns_lag1', country_id, month_id,100),
        ged_months_since_last_os_lag1_tx100 = public.cm_months_since_last_event_threshold('ged_best_os_lag1', country_id, month_id,100),
        ged_months_since_last_sb_tx500 = public.cm_months_since_last_event_threshold('ged_best_sb', country_id, month_id,500),
        ged_months_since_last_ns_tx500 = public.cm_months_since_last_event_threshold('ged_best_ns', country_id, month_id,500),
        ged_months_since_last_os_tx500 = public.cm_months_since_last_event_threshold('ged_best_os', country_id, month_id,500),
        ged_months_since_last_sb_lag1_tx500 = public.cm_months_since_last_event_threshold('ged_best_sb_lag1', country_id, month_id,500),
        ged_months_since_last_ns_lag1_tx500 = public.cm_months_since_last_event_threshold('ged_best_ns_lag1', country_id, month_id,500),
        ged_months_since_last_os_lag1_tx500 = public.cm_months_since_last_event_threshold('ged_best_os_lag1', country_id, month_id,500)
        WHERE month_id BETWEEN :m1 AND :m2
        """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Updating base onset variables for CM.")
    query = alchemy_text(
        """
        WITH a AS (SELECT * FROM onset_months_table('staging', 'country_month', 'ged_best_sb', 'country_id'))
        UPDATE staging.country_month
        SET onset_months_sb = a.onset_distance
        FROM a
        WHERE a.id = staging.country_month.id;

        WITH a AS (SELECT * FROM onset_months_table('staging', 'country_month', 'ged_best_ns', 'country_id'))
        UPDATE staging.country_month
        SET onset_months_ns = a.onset_distance
        FROM a
        WHERE a.id = staging.country_month.id;

        WITH a AS (SELECT * FROM onset_months_table('staging', 'country_month', 'ged_best_os', 'country_id'))
        UPDATE staging.country_month
        SET onset_months_os = a.onset_distance
        FROM a
        WHERE a.id = staging.country_month.id;
        """
    )
    with engine.connect() as con:
        trans = con.begin()
        con.execute(query)
        trans.commit()

    log.debug("Updating windowed onsets at cm level.")
    query = alchemy_text(
        """
        WITH a AS
                 (
                     SELECT id,
                            max(ged_best_sb)
                            OVER (PARTITION BY country_id ORDER BY month_id ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS sb_x1,
                            max(ged_best_os)
                            OVER (PARTITION BY country_id ORDER BY month_id ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS os_x1,
                            max(ged_best_ns)
                            OVER (PARTITION BY country_id ORDER BY month_id ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS ns_x1
                     FROM staging.country_month
                     WHERE month_id BETWEEN :m1 - 48 AND :m2
                 )
        UPDATE staging.country_month
        SET max24_best_sb = a.sb_x1,
            max24_best_ns = a.ns_x1,
            max24_best_os = a.os_x1
        FROM a
        WHERE a.id = staging.country_month.id
          AND month_id BETWEEN :m1 AND :m2
        """
    )
    with engine.connect() as con:
        trans = con.begin()
        con.execute(query, m1=month_start, m2=month_end)
        trans.commit()


def load_ged(api_version: str, from_month_id: int, to_month_id: int) -> None:
    """ Old load GED implementation

    As close to original repo implementation as possible but runnable
    by any user.
    """

    log.info("Started legacy GED load")

    month_start = from_month_id
    month_end = to_month_id

    _check_ged(month_start, api_version)
    _get_ged(api_version=api_version)
    _prepare_ged()
    _stage_ged_2_pgm(month_start=month_start, month_end=month_end)
    _stage_ged_2_cm(month_start=month_start, month_end=month_end)
    log.info("Started legacy GED imputation.")
    _geoi_prepare(month_start=month_start, month_end=month_end)
    _geoi_run(adm1=True)
    _geoi_run(adm1=False)
    _geoi_assemble(month_start=month_start, month_end=month_end)
    _geoi_build_dummies(n_imp=5, month_start=month_start, month_end=month_end)
    log.info("Finished legacy GED load")
