""" Ged loader, depends on original DB implementation

# TODO: Rewrite to hold all loading logic
"""
import os
import logging
from views.utils import db, io
from .legacy import load_ged as load_legacy_ged

log = logging.getLogger(__name__)


def fetch_ged() -> None:
    """ Do nothing, GED still fetched by old code """


def load_ged() -> None:
    """ Collect imputed and unimputed GED """

    log.info("Started loading GED.")

    load_legacy_ged("20.9.4", 484, 484)  # 2020-04

    db.drop_schema("ged")
    db.create_schema("ged")
    db.execute_query(
        query=io.read_file(
            path=os.path.join(os.path.dirname(__file__), "ged.sql")
        )
    )
    log.info("Finished loading GED.")
