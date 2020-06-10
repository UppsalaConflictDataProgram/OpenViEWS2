""" Skeleton building code """
import os
import logging
from views.utils import db

log = logging.getLogger(__name__)


def build_skeleton() -> None:
    """ Build skeleton schema by executing create_skeleton.sql """
    log.info("Started rebuilding skeleton schema.")
    path_query = os.path.join(os.path.dirname(__file__), "create_skeleton.sql")
    with open(path_query, "r") as f:
        query = f.read()
    db.execute_query(query)
    log.info("Finished rebuilding skeleton schema.")
