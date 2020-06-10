""" ACLED data loader, depends on original DB implementation

# TODO: Rewrite to hold all ACLED loading logic.
"""
import os
import logging
from views.utils import db, io
from .legacy import load_acled as load_legacy_acled


log = logging.getLogger(__name__)


def fetch_acled() -> None:
    """ Do nothing, ACLED still fetched by old code """


def load_acled() -> None:
    """ Code that brings acled to staging yet to be merged """

    log.info("Started loading ACLED.")

    load_legacy_acled(
        from_date="2020-01-01", from_month_id=483, to_month_id=484
    )

    db.drop_schema("acled")
    db.create_schema("acled")

    db.execute_query(
        query=io.read_file(
            path=os.path.join(os.path.dirname(__file__), "acled.sql")
        )
    )
    log.info("Finished loading ACLED.")
