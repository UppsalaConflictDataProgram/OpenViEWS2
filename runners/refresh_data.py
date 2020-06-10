""" Refresh all datasets that are defined by the specs """

from typing import Tuple
import argparse
import logging

import views

logging.basicConfig(
    level=logging.DEBUG,
    format=views.config.LOGFMT,
    handlers=[
        logging.FileHandler(views.utils.log.get_log_path(__file__)),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def parse_args() -> Tuple[bool, bool, bool]:

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="refresh all")
    parser.add_argument(
        "--geom", action="store_true", help="refresh geometries"
    )
    parser.add_argument("--tables", action="store_true", help="refresh tables")
    parser.add_argument(
        "--datasets", action="store_true", help="refresh datasets"
    )
    args = parser.parse_args()

    do_geom = args.geom
    do_tables = args.tables
    do_datasets = args.datasets
    if args.all:
        do_geom, do_tables, do_datasets = True, True, True

    if not any([do_geom, do_tables, do_datasets]):
        log.info("Nothing to do, see python refresh_data.py --help for args.")

    return do_geom, do_tables, do_datasets


def refresh_geometries() -> None:
    log.info(f"Refreshing all Geometries")
    for geometry in views.GEOMETRIES.values():
        geometry.refresh()
    log.info("Finished refreshing all Geometries")


def refresh_tables() -> None:
    log.info(f"Refreshing all Tables")
    for table in views.TABLES.values():
        table.refresh()
    log.info("Finished refreshing all Tables")


def refresh_datasets() -> None:
    log.info(f"Refreshing all Datasets")
    for dataset in views.DATASETS.values():
        dataset.refresh()
    log.info("Finished refreshing all Datasets")


def refresh_all():
    do_geom, do_tables, do_datasets = parse_args()
    if do_geom:
        refresh_geometries()
    if do_tables:
        refresh_tables()
    if do_datasets:
        refresh_datasets()


if __name__ == "__main__":
    refresh_all()
