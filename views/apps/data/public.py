""" Data publication interface

Data is publichsed as .csv and .geojson files in a .zip archive.
These formats were chosen because they are the most common and can be
read by all.
Functions in this module take a zip file and cache the data in the
views structure as parquet and geojson files.

"""
from typing import Dict, Union, List
import datetime
import os
import tempfile
import logging

from views.utils import io
from .api import Table, GeomCountry, GeomPriogrid

log = logging.getLogger(__name__)


def _date_now() -> str:
    """ Get current UTC time """
    return datetime.datetime.utcnow().strftime("%Y%m%d")


def export_tables_and_geoms(
    tables: Dict[str, Table],
    geometries: Dict[str, Union[GeomCountry, GeomPriogrid]],
    dir_output: str,
) -> str:
    """ Export tables and geometries to timestamped zip in dir_output """
    path_zip = os.path.join(
        dir_output, f"views_tables_and_geoms_{_date_now()}.zip"
    )
    log.info(f"Started exporting tables and geoms to {path_zip}")
    with tempfile.TemporaryDirectory() as tempdir:
        paths: List[str] = []
        for table in tables.values():
            path = os.path.join(tempdir, f"{table.name}.csv")
            io.df_to_csv(df=table.df, path=path)
            paths.append(path)

        for geom in geometries.values():
            # Make sure we have gdf locally
            _ = geom.gdf
            paths.append(geom.path)

        # Add the README to the zip.
        paths.append(
            os.path.join(
                os.path.dirname(__file__), "export_readme", "README.md"
            )
        )

        io.make_zipfile(
            path_zip=path_zip, paths_members=paths,
        )
    log.info(f"Finished exporting tables and geoms to {path_zip}")
    return path_zip


def import_tables_and_geoms(
    tables: Dict[str, Table],
    geometries: Dict[str, Union[GeomCountry, GeomPriogrid]],
    path_zip: str,
) -> None:
    """ Import tables and geometries to local cache structure from zip"""

    log.info(f"Started initalising cache from zip at {path_zip}")
    with tempfile.TemporaryDirectory() as tempdir:
        io.unpack_zipfile(path_zip=path_zip, destination=tempdir)

        for geom in geometries.values():
            path = os.path.join(tempdir, geom.fname)
            if os.path.isfile(path):
                geom.init_cache_from_geojson(path=path)
            else:
                log.debug(f"No matching .geojson for {geom.name}")

        for table in tables.values():
            path = os.path.join(tempdir, f"{table.name}.csv")
            if os.path.isfile(path):
                table.init_cache_from_csv(path)
            else:
                raise RuntimeError(f"No matching .csv for {table.name}")
    log.info(f"Fininshed initalising cache from zip at {path_zip}")


def fetch_latest_zip_from_website(path_dir_destination: str) -> str:
    """ Feth the latest zip from the website """

    # Update this
    url_base = "https://views.pcr.uu.se/download/datasets"
    fnames = [
        fname
        for fname in io.list_files_in_webdir(url=url_base)
        if fname.startswith("views_tables_and_geoms_")
    ]
    log.debug(f"Found {fnames} that look like views_tables_and_geoms_")
    fname_latest = sorted(fnames).pop()
    log.debug(f"Latest file looks like: {fname_latest}")
    url = f"{url_base}/{fname_latest}"
    path = os.path.join(path_dir_destination, fname_latest)
    io.fetch_url_to_file(url, path)
    return path
