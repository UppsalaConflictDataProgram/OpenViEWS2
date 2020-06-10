""" Reign """
import os
import tempfile
import logging
from typing import Any, Dict
import requests
import pandas as pd  # type: ignore
import bs4  # type: ignore

from views.apps.data import missing
from views.database import common
from views.utils import io, db

log = logging.getLogger(__name__)


def fetch_reign() -> None:
    """ Fetch REIGN data """

    def get_latest_data_url(url_report) -> str:
        html_doc = requests.get(url_report).content
        soup = bs4.BeautifulSoup(html_doc, "html5lib")
        container = soup.find("div", {"class": "post-container"})
        url_data = container.find("a", href=True)["href"]
        log.debug(f"url_data: {url_data}")

        if not url_data.endswith(".csv"):
            raise RuntimeError(f"Reign link doesn't look like .csv {url_data}")

        return url_data

    log.debug("Started fetching reign")
    url_base = "https://oefdatascience.github.io/REIGN.github.io"
    url_report = f"{url_base}/menu/reign_current.html"
    url = get_latest_data_url(url_report=url_report)
    common.fetch_source_simply(name="reign", url=url)
    log.debug("Finished fetching reign")


def fix_ccodes(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """ Fix country codes as defined by spec ccode_replaces """
    log.debug("Fixing ccodes")

    fixes = spec["ccode_replaces"]
    for fix_name, values in fixes.items():
        old = values["old"]
        new = values["new"]
        df.loc[df.ccode == old, "ccode"] = new
        log.debug(f"Replaced ccode {old} with {new} for {fix_name}")

    log.debug("Dropping duplicate country-months for leadership changes.")
    dropdup_cols = ["ccode", "year", "month"]
    # Some messages are too big even for debug...
    # msg = df[df.duplicated(subset=dropdup_cols, keep=False)].to_string()
    # log.debug(msg)
    df = df.sort_values("tenure_months")
    len_df_predrop = len(df)
    df = df.drop_duplicates(subset=dropdup_cols, keep="first")
    len_df_postdrop = len(df)

    log.debug(f"Dropped {len_df_predrop - len_df_postdrop} duplicate obs")

    return df


def encode_govt_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """ Encode government dummies """
    log.debug("Encoding reign government dummies")

    def cleanup_govtype_name(name):
        """ Remove " ", "-", "/" from government type strings """
        name = name.lower()
        name = name.replace(" ", "_").replace("-", "_").replace("/", "_")
        name = name.replace("__", "_").replace("__", "_")
        return name

    df["government"] = df["government"].apply(cleanup_govtype_name)
    df_gov = pd.get_dummies(df["government"], prefix="gov")
    log.debug(f"Adding dummy cols {list(df_gov.columns)}")
    df = df.join(df_gov)
    return df


def load_reign() -> None:
    """ Load reign """
    log.info("Started loading reign.")

    spec = io.load_yaml(os.path.join(os.path.dirname(__file__), "spec.yaml"))
    with tempfile.TemporaryDirectory() as tempdir:
        paths = common.get_files_latest_fetch(name="reign", tempdir=tempdir)
        path_csv = [path for path in paths if path.endswith(".csv")].pop()
        df = io.csv_to_df(path=path_csv)

    df = fix_ccodes(df, spec)
    df = encode_govt_dummies(df)

    df = df.set_index(["year", "month", "ccode"])
    df = df.join(
        db.query_to_df(
            query="""
                SELECT id AS country_id, gwcode AS ccode
                FROM staging.country WHERE gweyear=2016;
                """
        ).set_index(["ccode"])
    )
    df = df.join(
        db.query_to_df(
            query="""
            SELECT id AS month_id, year_id AS year, month FROM staging.month;
            """
        ).set_index(["year", "month"])
    )
    df = df.reset_index().set_index(["month_id", "country_id"])
    df = df.drop(
        columns=["year", "month", "ccode", "country", "government", "leader"]
    )

    df_skeleton = db.db_to_df(
        fqtable="skeleton.cm_global",
        cols=["month_id", "country_id"],
        ids=["month_id", "country_id"],
    )
    len_skel = len(df_skeleton)
    df = df_skeleton.join(df, how="left")
    if not len(df) == len_skel:
        raise RuntimeError(f"Join not correct, {len_skel} != {len(df)}")

    df = df.add_prefix("reign_")

    db.drop_schema("reign_v2")
    db.create_schema("reign_v2")
    db.df_to_db(df=df, fqtable="reign_v2.cm_unimp")

    db.df_to_db(
        df=missing.fill_groups_with_time_means(missing.extrapolate(df)),
        fqtable="reign_v2.cm_extrapolated",
    )

    log.info("Finished loading reign.")
