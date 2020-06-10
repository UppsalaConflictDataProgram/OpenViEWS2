""" Get the prediction competition data from the ViEWS website """
from typing import List, Optional, Dict
import tempfile
import logging
import os
from datetime import date

from views.apps.data import api
from views.utils import io
from views.config import DIR_STORAGE
from views.specs.data import DATASETS

log = logging.getLogger()

DIR_UPLOAD = os.path.join(DIR_STORAGE, "upload")
io.create_directory(DIR_UPLOAD)


def fetch_prediction_competition_data(
    fnames_want: Optional[List[str]] = None,
) -> Dict[str, str]:
    """ Fetch and unpack the prediction competition data"""

    fname_zip = "views_pred_comp_data_20200427.zip"
    url = f"https://views.pcr.uu.se/download/datasets/{fname_zip}"

    if not fnames_want:
        fnames_want = ["cm.csv", "pgm.csv"]

    dir_destination = os.path.join(DIR_STORAGE, "prediction_competition")
    paths_want = [
        os.path.join(dir_destination, fname) for fname in fnames_want
    ]

    io.create_directory(dir_destination)

    if all([os.path.isfile(path) for path in paths_want]):
        log.info("Files already where we need them")
    else:
        log.info(f"Fetching {fnames_want} from {url}")
        with tempfile.TemporaryDirectory() as tempdir:
            path_zip = os.path.join(tempdir, fname_zip)
            io.fetch_url_to_file(url=url, path=path_zip)
            paths_unzipped = io.unpack_zipfile(path_zip, destination=tempdir)
            paths_destination: List[str] = []
            for path in paths_unzipped:
                fname = os.path.basename(path)
                if fname in fnames_want:
                    path_destination = os.path.join(dir_destination, fname)
                    io.move_file(path_from=path, path_to=path_destination)
                    paths_destination.append(path_destination)

        paths_missing = [
            path for path in paths_want if path not in paths_destination
        ]
        if paths_missing:
            raise RuntimeError(f"Missing paths {paths_missing}")

    data = {os.path.basename(path): path for path in paths_want}

    return data


def extract_and_package_data():
    """ Get raw data from database, dump to files and zip it up """
    with tempfile.TemporaryDirectory() as tempdir:
        paths = []
        # Dump tables to csv
        for name, dataset in DATASETS.items():
            fname = f"{name}.csv"
            path = os.path.join(tempdir, fname)
            dataset.export_raw_to_csv(path=path)
            paths.append(path)

        geom_c = api.GeomCountry()
        geom_c.refresh()
        paths.append(geom_c.path)
        geom_pg = api.GeomPriogrid()
        geom_pg.refresh()
        paths.append(geom_pg.path)

        today = date.today().strftime("%Y%m%d")
        fname_zip = f"data_export_{today}.zip"
        io.make_zipfile(
            path_zip=os.path.join(DIR_UPLOAD, fname_zip), paths_members=paths
        )
    log.info(f"Wrote zip to {os.path.join(DIR_UPLOAD, fname_zip)}")
    log.info("Now go ahead and upload it to the webserver manually.")


def refresh_datasets_from_website(fname_zip="data_export_20200513.zip"):
    """ Initialise local data cache from website public data """

    url = f"https://views.pcr.uu.se/download/datasets/{fname_zip}"
    log.info(f"Fetching from {url}")
    with tempfile.TemporaryDirectory() as tempdir:
        path_zip = os.path.join(tempdir, fname_zip)
        io.fetch_url_to_file(url=url, path=path_zip)
        log.info("Done fetching. Unpacking zipfile.")
        _ = io.unpack_zipfile(path_zip, destination=tempdir)

        log.info("Initalising local geometries")
        geom_c = api.GeomCountry()
        geom_pg = api.GeomPriogrid()
        path_geom_c = os.path.join(tempdir, os.path.basename(geom_c.path))
        path_geom_pg = os.path.join(tempdir, os.path.basename(geom_pg.path))
        geom_c.init_cache_from_geojson(path_geom_c)
        geom_pg.init_cache_from_geojson(path_geom_pg)

        log.info("Initalising datasets.")
        for name, dataset in DATASETS.items():
            fname = f"{name}.csv"
            path_csv = os.path.join(tempdir, fname)
            dataset.init_cache_from_csv(path_csv)
    log.info("Done initalising data, you can now use views.DATASETS")
