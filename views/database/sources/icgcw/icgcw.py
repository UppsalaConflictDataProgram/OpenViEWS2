""" International Crisis Group - Crisis Watch loader """
from typing import List, Dict, Any
import logging
import re
import tempfile
import os

import joblib  # type: ignore
import pandas as pd  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

from views.utils import db, io
from views.database import common
from views.apps.data import missing

log = logging.getLogger(__name__)


def parse_page(path: str) -> List[Dict[Any, Any]]:
    """ CrisisWatch parser using bs4. Appends to dataframe and returns df """
    # pylint: disable=too-many-locals

    soup = BeautifulSoup(io.read_file(path), "html.parser")

    # loop over blocks
    search = {
        "class": "c-crisiswatch-entry [ o-container o-container--m u-pr ]"
    }
    entries = []
    for block in soup.find_all("div", search):
        # remove whitespace titles
        countryname = block.find("h3").text
        # remove unnecessary spacing

        # countryname = re.sub("^\s+|\s+$", "", countryname, flags=re.UNICODE)
        countryname = re.sub(r"^s+|s+$", "", countryname, flags=re.UNICODE)
        countryname = countryname.strip()
        entrydate = block.find("time").text
        # entries may have no text, so adding a try here
        try:
            cls_tag = {"class": "o-crisis-states__detail [ u-ptserif u-fs18 ]"}
            entrytext = block.find("div", cls_tag).text
            entrytext = entrytext.replace("\n\t", "")
        except AttributeError:
            entrytext = ""
        # prepare dummies using list
        tblock = block.find("h3")
        updates = list(tblock.find_all("use"))
        deteriorated = 1 if "#deteriorated" in str(updates) else 0
        improved = 1 if "#improved" in str(updates) else 0
        alert = 1 if "#risk-alert" in str(updates) else 0
        resolution = 1 if "#resolution" in str(updates) else 0
        unobserved = 0
        entry_data = {
            "date": entrydate,
            "name": countryname,
            "alerts": alert,
            "opportunities": resolution,
            "deteriorated": deteriorated,
            "improved": improved,
            "unobserved": unobserved,
            "text": entrytext,
        }
        entries.append(entry_data)

    log.debug(f"Read {len(entries)} entries from {path}")
    return entries


def load_and_parse_entries(parallel: bool = False) -> List[Dict[Any, Any]]:
    """ Parse all entries in path_tar """

    with tempfile.TemporaryDirectory() as tempdir:
        common.get_files_latest_fetch(name="icgcw", tempdir=tempdir)

        paths = []
        for root, _, files in os.walk(tempdir):
            for fname in files:
                path = os.path.join(root, fname)
                if fname.endswith(".html"):
                    paths.append(path)
                else:
                    msg = (
                        f"Found a file that wasn't .html. "
                        f"something broke for: {path}"
                    )
                    raise RuntimeError(msg)

        log.info(f"Found {len(paths)} files to parse")

        if parallel:
            log.info(f"Parsing {len(paths)} files in parallel")
            parsed_pages = joblib.Parallel(n_jobs=-1, verbose=10)(
                joblib.delayed(parse_page)(path) for path in paths
            )
        else:
            log.info(f"Parsing {len(paths)} files in sequence")
            parsed_pages = list(map(parse_page, sorted(paths)))

    parsed_entries = []
    for parsed_page in parsed_pages:
        for entry in parsed_page:
            parsed_entries.append(entry)

    return parsed_entries


def do_renames(entries, spec):
    """ Do renames in spec["cname_fixes"] """

    for rename in spec["cname_fixes"]:
        old = rename["old"]
        new = rename["new"]
        count = 0
        for entry in entries:
            if entry["name"] == old:
                entry["name"] = new
                count = count + 1
        log.debug(f"Renamed {count} {old} to {new}.")

    return entries


def drop_unnamed_entries(entries):
    """ Drop entries that have empty name """

    def nonempty_name(entry):
        """ True if name not empty """
        if not entry["name"] == "":
            has_name = True
        else:
            has_name = False
            msg = f"Empty name in {entry}"
            log.debug(msg)
        return has_name

    len_predrop = len(entries)
    entries = list(filter(nonempty_name, entries))
    len_postdrop = len(entries)
    msg = f"Dropped {len_predrop - len_postdrop} entries for empty name"
    log.debug(msg)

    return entries


def drop_drops(
    entries: List[Dict[Any, Any]], drops: List[str]
) -> List[Dict[Any, Any]]:
    """ Drop entries as by spec['drops'] """

    entries_wo_drops = [
        entry for entry in entries if not entry["name"] in drops
    ]

    log.debug(f"Dropped {len(entries) - len(entries_wo_drops)} entries")

    return entries_wo_drops


def split_multi_country_entries(
    entries: List[Dict[Any, Any]]
) -> List[Dict[Any, Any]]:
    """ Split entries by / """

    entries_w_splits = []
    done_splits = []
    for entry in entries:
        if "/" in entry["name"]:
            done_splits.append(entry["name"])
            for country_name in entry["name"].split("/"):
                new_entry = entry.copy()
                new_entry["name"] = country_name
                entries_w_splits.append(new_entry)
        else:
            entries_w_splits.append(entry.copy())

    done_splits = sorted(list(set(done_splits)))
    for split in done_splits:
        log.debug(f"Split {split}")
    msg = f"Splitting Created {len(entries_w_splits) - len(entries)}"
    log.debug(msg)

    return entries_w_splits


def debug_log_unique_names(entries: List[Dict[Any, Any]]) -> None:
    """ Put unique entry['name']'s in debug log """

    names = []
    for entry in entries:
        if entry["name"] not in names:
            names.append(entry["name"])

    log.debug(f"Found {len(names)} unique names:")
    for name in sorted(names):
        log.debug(f"{name}")


def drop_duplicate_country_months(df: pd.DataFrame) -> pd.DataFrame:
    """ Drop duplicated country months """

    # @TODO: replace with max() of indicators during month perhaps?
    len_predrop = len(df)
    df = df.sort_values(by=["year", "month", "name"])
    df = df.drop_duplicates(subset=["year", "month", "name"], keep="first")
    len_postdrop = len(df)
    log.info(f"Dropped {(len_predrop - len_postdrop)} duplicates")

    return df


def set_dates(df: pd.DataFrame) -> pd.DataFrame:
    """ Set year and month columns in df from date """

    df["year"] = pd.DatetimeIndex(df.date).year  # pylint: disable=no-member
    df["month"] = pd.DatetimeIndex(df.date).month  # pylint: disable=no-member

    return df


def load_icgcw() -> None:
    """ Load ICGCW """
    log.info("Starting ICGCW import")

    spec = io.load_yaml(
        path=os.path.join(os.path.dirname(__file__), "spec.yaml")
    )
    # Get all the entries as list of dicts
    entries = load_and_parse_entries(parallel=True)
    entries = drop_unnamed_entries(entries)

    # Some renames depend on splits so we do splits twice
    entries = split_multi_country_entries(entries)
    entries = do_renames(entries, spec)
    entries = split_multi_country_entries(entries)

    entries = drop_drops(entries, drops=spec["drops"])

    debug_log_unique_names(entries)

    df = pd.DataFrame(entries)
    df = set_dates(df)
    df = drop_duplicate_country_months(df)
    df = df.set_index(["year", "month", "name"])
    cols = [
        "alerts",
        "opportunities",
        "deteriorated",
        "improved",
        "unobserved",
    ]
    df = df[cols]

    df_c = (
        db.query_to_df("SELECT id AS country_id, name FROM staging.country;")
        .sort_values(by=["country_id"], ascending=False)
        .drop_duplicates(subset=["name"])
        .set_index(["name"])
    )
    df_m = (
        db.query_to_df(
            """
            SELECT id AS month_id, year_id AS year, month  FROM staging.month;
            """
        )
        .set_index(["year", "month"])
        .sort_index()
    )
    df_skeleton = db.db_to_df(
        fqtable="skeleton.cm_global",
        cols=["month_id", "country_id"],
        ids=["month_id", "country_id"],
    )

    df = (
        df.join(df_c)
        .join(df_m)
        .dropna(subset=["country_id"])
        .set_index(["month_id", "country_id"])
        .sort_index()
    )
    df = df_skeleton.join(df, how="left")

    df = missing.extrapolate(df)
    df["unobserved"] = df["unobserved"].fillna(1)
    df = df.fillna(0)

    # orer cols and rows
    df = df.sort_index(axis=1).sort_index()
    df = df.add_prefix("icgcw_")

    # @TODO: Change this to icgcw without v2 once we drop flat
    schema = "icgcw_v2"
    db.drop_schema(schema)
    db.create_schema(schema)
    db.df_to_db(fqtable=f"{schema}.cm", df=df)
