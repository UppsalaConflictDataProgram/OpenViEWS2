""" Import data to local cache """

import argparse
import logging
from typing import Optional, Tuple
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


def parse_args() -> Tuple[Optional[str], bool, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_zip", type=str, help="Path to zip to import from",
    )
    parser.add_argument(
        "--fetch", action="store_true", help="Fetch from website."
    )
    parser.add_argument(
        "--datasets", action="store_true", help="Refresh datasets."
    )
    args = parser.parse_args()

    if args.path_zip and args.fetch:
        raise RuntimeError("Pass in --path_zip or --fetch, not both.")

    return args.path_zip, args.fetch, args.datasets


def run_import_tables_and_geoms(path_zip) -> None:
    views.apps.data.public.import_tables_and_geoms(
        tables=views.TABLES, geometries=views.GEOMETRIES, path_zip=path_zip,
    )


def refresh_datasets() -> None:

    log.info("Started refreshing all datasets.")

    datasets_to_update = [
        "cm_global_imp_0",
        "cm_africa_imp_0",
        "pgm_africa_imp_0",
    ]
    for dataset_name in datasets_to_update:
        log.info(f"Started refreshing dataset {dataset_name}")
        views.DATASETS[dataset_name].refresh(do_transforms=False)

    log.info("Finished refreshing all imp_0 datasets.")


def main() -> None:

    path_zip, do_fetch, do_datasets = parse_args()

    if do_fetch:
        path_zip = views.apps.data.public.fetch_latest_zip_from_website(
            path_dir_destination=views.DIR_SCRATCH
        )

    run_import_tables_and_geoms(path_zip)
    refresh_datasets()


if __name__ == "__main__":
    main()
