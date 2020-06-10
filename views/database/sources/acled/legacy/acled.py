""" ACLED loader from original ViEWS repo

Identical in functionality to original loader.
Adapted to use connectstring from configfile.
"""
# flake8: noqa
# pylint: skip-file

import json
import logging
import time
import os
from typing import Dict, Union

import pandas as pd  # type: ignore
import requests  # type: ignore
from sqlalchemy import create_engine  # type: ignore
from sqlalchemy.sql import text as alchemy_text  # type: ignore
from sqlalchemy.sql import select, column, func  # type: ignore
from datetime import datetime, timedelta, date  # type: ignore

from views import config
from views.utils import db, io

log = logging.getLogger(__name__)
CONNECTSTRING = config.DATABASES["default"].connectstring


def load_acled(from_date: str, from_month_id: int, to_month_id: int) -> None:
    _get_acled(from_date=from_date)
    _prepare_acled()
    _stage_acled(month_start=from_month_id, month_end=to_month_id)


def _get_acled(from_date: str):
    """ Fetch ACLED from API to dataprep.acled """

    def _get_acled_slice(date_start: str, page_count: int) -> pd.DataFrame:

        url = "http://api.acleddata.com/acled/read"
        # since ACLED only knows > and <, and we want >=, calculate yesterday's date
        event_date: str = datetime.strftime(
            datetime.strptime(date_start, "%Y-%m-%d") - timedelta(1),
            "%Y-%m-%d",
        )
        payload: Dict[str, Union[str, int]] = {
            "terms": "accept",
            "event_date": event_date,
            "event_date_where": ">",
            "page": page_count,
        }
        r = requests.get(url=url, params=payload)

        output = r.json()
        if output["count"] > 0:
            df = pd.DataFrame(output["data"])
            count = output["count"]
        else:
            df = None
            count = output["count"]
        return count, df

    log.debug("Getting ACLED")
    df = pd.DataFrame()
    cur_page = 1
    while True:
        count, acled_slice = _get_acled_slice(from_date, cur_page)
        if count > 0:
            df = df.append(acled_slice, ignore_index=True)
            cur_page += 1
        else:
            break
    db.df_to_db(df=df, fqtable="dataprep.acled")
    log.debug("Finished getting ACLED")


def _prepare_acled():

    log.debug("Started _prepare_acled()")
    # This was pure sql, not even a parametrised query.
    db.execute_query(
        query=io.read_file(
            path=os.path.join(os.path.dirname(__file__), "prepare_acled.sql")
        )
    )
    log.debug("Finished _prepare_acled()")


def _stage_acled(
    month_start: int, month_end: int,
):
    log.debug("Started _stage_acled")
    engine = create_engine(CONNECTSTRING)

    with engine.connect() as con:

        # Check we don't want to update some column that doesn't exist in the downloaded ACLED
        # Last month in ACLED is always incomplete, thus use month-1 for end limit.

        limits = con.execute(
            "SELECT min(month_id) AS int, max(month_id)-1 AS int FROM preflight.acled"
        ).fetchone()
        if limits[0] > month_start:
            month_start = limits[0]
        if limits[1] <= month_end:
            month_end = limits[1]

    # This will compute event counts for priogrid-month observations as well as for 1st and 2nd order lags

    query = alchemy_text(
        """
        UPDATE staging.priogrid_month SET
        acled_count_sb = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,0,1),
        acled_count_ns = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,0,2),
        acled_count_os = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,0,3),
        acled_count_pr = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,0,4),
        acled_count_prp= public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,0,'p'),
        acled_count_prr= public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,0,'r'),
        acled_count_prx= public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,0,'x'),
        acled_count_pry= public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,0,'y'),
        acled_fat_sb = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,0,1),
        acled_fat_ns = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,0,2),
        acled_fat_os = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,0,3),
        acled_fat_pr = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,0,4),
        acled_fat_prp= public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,0,'p'),
        acled_fat_prr= public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,0,'r'),
        acled_fat_prx= public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,0,'x'),
        acled_fat_pry= public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,0,'y'),
        acled_count_sb_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,1,1),
        acled_count_ns_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,1,2),
        acled_count_os_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,1,3),
        acled_count_pr_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,1,4),
        acled_count_prp_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,1,'p'),
        acled_count_prr_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,1,'r'),
        acled_count_prx_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,1,'x'),
        acled_count_pry_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,1,'y'),
        acled_fat_sb_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,1,1),
        acled_fat_ns_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,1,2),
        acled_fat_os_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,1,3),
        acled_fat_pr_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,1,4),
        acled_fat_prp_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,1,'p'),
        acled_fat_prr_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,1,'r'),
        acled_fat_prx_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,1,'x'),
        acled_fat_pry_lag1 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,1,'y'),
        acled_count_sb_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,2,1),
        acled_count_ns_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,2,2),
        acled_count_os_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,2,3),
        acled_count_pr_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,2,4),
        acled_count_prp_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,2,'p'),
        acled_count_prr_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,2,'r'),
        acled_count_prx_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,2,'x'),
        acled_count_pry_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,TRUE,2,'y'),
        acled_fat_sb_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,2,1),
        acled_fat_ns_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,2,2),
        acled_fat_os_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,2,3),
        acled_fat_pr_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,2,4),
        acled_fat_prp_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,2,'p'),
        acled_fat_prr_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,2,'r'),
        acled_fat_prx_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,2,'x'),
        acled_fat_pry_lag2 = public.aggregate_acled_pgm (priogrid_gid,month_id,FALSE,2,'y')
        WHERE month_id BETWEEN :m1 AND :m2
        AND
        priogrid_gid IN (SELECT gid FROM staging.priogrid WHERE in_africa)
  """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Creating temporal lags...")

    # This will compute temporal lags.
    # This must be run AFTER the above query has commited

    query = alchemy_text(
        """
        SELECT public.make_priogrid_month_acled_temporal_lags(1,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_acled_temporal_lags(2,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_acled_temporal_lags(3,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_acled_temporal_lags(4,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_acled_temporal_lags(5,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_acled_temporal_lags(6,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_acled_temporal_lags(7,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_acled_temporal_lags(8,FALSE,:m1,:m2);
        SELECT public.make_priogrid_month_acled_temporal_lags(9,FALSE,:m1,:m2);
    """
    )
    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Computing time since last event...")

    # This will compute time since last event.
    # This must be run AFTER the above query has commited

    query = alchemy_text(
        """
        UPDATE staging.priogrid_month SET
        acled_months_since_last_sb = public.months_since_last_event('acled_count_sb', priogrid_gid, month_id),
        acled_months_since_last_ns = public.months_since_last_event('acled_count_ns', priogrid_gid, month_id),
        acled_months_since_last_os = public.months_since_last_event('acled_count_os', priogrid_gid, month_id),
        acled_months_since_last_pr = public.months_since_last_event('acled_count_pr', priogrid_gid, month_id),
        acled_months_since_last_prp = public.months_since_last_event('acled_count_prp', priogrid_gid, month_id),
        acled_months_since_last_prr = public.months_since_last_event('acled_count_prr', priogrid_gid, month_id),
        acled_months_since_last_prx = public.months_since_last_event('acled_count_prx', priogrid_gid, month_id),
        acled_months_since_last_pry = public.months_since_last_event('acled_count_pry', priogrid_gid, month_id),
        acled_months_since_last_sb_lag1 = public.months_since_last_event('acled_count_sb_lag1', priogrid_gid, month_id),
        acled_months_since_last_ns_lag1 = public.months_since_last_event('acled_count_ns_lag1', priogrid_gid, month_id),
        acled_months_since_last_os_lag1 = public.months_since_last_event('acled_count_os_lag1', priogrid_gid, month_id),
        acled_months_since_last_pr_lag1 = public.months_since_last_event('acled_count_pr_lag1', priogrid_gid, month_id),
        acled_months_since_last_prp_lag1 = public.months_since_last_event('acled_count_prp_lag1', priogrid_gid, month_id),
        acled_months_since_last_prr_lag1 = public.months_since_last_event('acled_count_prr_lag1', priogrid_gid, month_id),
        acled_months_since_last_prx_lag1 = public.months_since_last_event('acled_count_prx_lag1', priogrid_gid, month_id),
        acled_months_since_last_pry_lag1 = public.months_since_last_event('acled_count_pry_lag1', priogrid_gid, month_id),
        acled_months_since_last_sb_lag2 = public.months_since_last_event('acled_count_sb_lag2', priogrid_gid, month_id),
        acled_months_since_last_ns_lag2 = public.months_since_last_event('acled_count_ns_lag2', priogrid_gid, month_id),
        acled_months_since_last_os_lag2 = public.months_since_last_event('acled_count_os_lag2', priogrid_gid, month_id),
        acled_months_since_last_pr_lag2 = public.months_since_last_event('acled_count_pr_lag2', priogrid_gid, month_id),
        acled_months_since_last_prp_lag2 = public.months_since_last_event('acled_count_prp_lag2', priogrid_gid, month_id),
        acled_months_since_last_prr_lag2 = public.months_since_last_event('acled_count_prr_lag2', priogrid_gid, month_id),
        acled_months_since_last_prx_lag2 = public.months_since_last_event('acled_count_prx_lag2', priogrid_gid, month_id),
        acled_months_since_last_pry_lag2 = public.months_since_last_event('acled_count_pry_lag2', priogrid_gid, month_id)
        WHERE month_id BETWEEN :m1 AND :m2;
    """
    )

    with engine.connect() as con:
        con.execute(query, m1=month_start, m2=month_end)

    log.debug("Preparing ACLED for CM...")

    with engine.connect() as con:

        trans = con.begin()
        query = "ALTER TABLE preflight.acled_full ADD COLUMN gwno INT"
        con.execute(query)
        trans.commit()

        trans = con.begin()
        query = (
            "UPDATE preflight.acled_full SET gwno=isonum_to_gwcode(iso::int)"
        )
        con.execute(query)
        trans.commit()

        trans = con.begin()
        try:
            con.execute(
                "ALTER TABLE preflight.acled_full ADD COLUMN country_month_id INT"
            )
        except:
            pass
        trans.commit()

        trans = con.begin()
        query = alchemy_text(
            """
                with a as
        (SELECT cm.*, c.gwcode FROM staging.country_month cm left join
              staging.country c on (cm.country_id=c.id))
        UPDATE preflight.acled_full SET country_month_id=a.id
        FROM a
        WHERE (a.gwcode::int = acled_full.gwno::int AND a.month_id = acled_full.month_id);
        """
        )
        con.execute(query)
        con.execute(
            "CREATE INDEX acled_full_cm_idx ON preflight.acled_full(country_month_id, type_of_violence)"
        )
        trans.commit()

        log.debug("Updating CM aggregates for ACLED...")

        trans = con.begin()
        query = alchemy_text(
            """
        UPDATE staging.country_month SET
        acled_count_sb = public.aggregate_cm_acled(id,TRUE,0,1),
        acled_count_ns = public.aggregate_cm_acled(id,TRUE,0,2),
        acled_count_os = public.aggregate_cm_acled(id,TRUE,0,3),
        acled_count_pr = public.aggregate_cm_acled(id,TRUE,0,4),
        acled_count_sb_lag1 = public.aggregate_cm_acled(id,TRUE,1,1),
        acled_count_ns_lag1 = public.aggregate_cm_acled(id,TRUE,1,2),
        acled_count_os_lag1 = public.aggregate_cm_acled(id,TRUE,1,3),
        acled_count_pr_lag1 = public.aggregate_cm_acled(id,TRUE,1,4)
        WHERE month_id BETWEEN :m1 AND :m2
        """
        )
        con.execute(query, m1=month_start, m2=month_end)
        trans.commit()
        log.debug("Updating CM months since...")
        trans = con.begin()
        query = alchemy_text(
            """
            UPDATE staging.country_month SET
            acled_months_since_last_sb = public.cm_months_since_last_event('acled_count_sb', country_id, month_id),
            acled_months_since_last_ns = public.cm_months_since_last_event('acled_count_ns', country_id, month_id),
            acled_months_since_last_os = public.cm_months_since_last_event('acled_count_os', country_id, month_id),
            acled_months_since_last_pr = public.cm_months_since_last_event('acled_count_pr', country_id, month_id),
            acled_months_since_last_sb_lag1 = public.cm_months_since_last_event('acled_count_sb_lag1', country_id, month_id),
            acled_months_since_last_ns_lag1 = public.cm_months_since_last_event('acled_count_ns_lag1', country_id, month_id),
            acled_months_since_last_os_lag1 = public.cm_months_since_last_event('acled_count_os_lag1', country_id, month_id),
            acled_months_since_last_pr_lag1 = public.cm_months_since_last_event('acled_count_pr_lag1', country_id, month_id)
            WHERE month_id BETWEEN :m1 AND :m2
            """
        )
        con.execute(query, m1=month_start - 12, m2=month_end + 12)
        trans.commit()
        log.debug("Updating CM temporal lags...")
        trans = con.begin()
        query = alchemy_text(
            """
          SELECT make_country_month_acled_temporal_lags(1, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(2, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(3, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(4, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(5, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(6, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(7, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(8, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(9, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(10, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(11, FALSE, :m1, :m2);
          SELECT make_country_month_acled_temporal_lags(12, FALSE, :m1, :m2);
          """
        )
        con.execute(query, m1=month_start, m2=month_end)
        trans.commit()
    log.debug("Finished _stage_acled!")
