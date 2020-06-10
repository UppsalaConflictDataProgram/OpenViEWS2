""" Refresh all datasets that are defined by the specs """

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


def run_export_tables_and_geoms() -> None:
    views.apps.data.public.export_tables_and_geoms(
        tables=views.TABLES,
        geometries=views.GEOMETRIES,
        dir_output=views.DIR_SCRATCH,
    )


def main():
    run_export_tables_and_geoms()


if __name__ == "__main__":
    main()
