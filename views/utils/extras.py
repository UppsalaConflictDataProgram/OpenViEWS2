""" Get the prediction competition data from the ViEWS website """
from typing import List, Optional, Dict
import tempfile
import logging
import os


from views.utils import io
from views.config import DIR_STORAGE

log = logging.getLogger()


def fetch_prediction_competition_data(
    fnames_want: Optional[List[str]] = None,
) -> Dict[str, str]:
    """ Fetch and unpack the prediction competition data"""

    fname_zip = "views_pred_comp_data_20200324.zip"
    url = f"https://views.pcr.uu.se/download/datasets/{fname_zip}"

    if not fnames_want:
        fnames_want = ["cm.csv", "pgm.csv"]

    dir_destination = os.path.join(DIR_STORAGE, "prediction_competition")
    paths_want = [
        os.path.join(dir_destination, fname) for fname in fnames_want
    ]

    io.create_directory(dir_destination)

    if all([os.path.isfile(path) for path in paths_want]):
        log.info(f"Files already where we need them")
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
