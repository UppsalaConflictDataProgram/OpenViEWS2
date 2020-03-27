""" File IO utilities """
import logging
from typing import Any, Optional, List
import shutil
import zipfile
import os

import requests
import yaml
import pandas as pd  # type: ignore

log = logging.getLogger(__name__)


def read_file(path: str) -> str:
    """ Read a files contents """
    log.debug(f"Reading file at {path}")
    with open(path, "r") as f:
        return f.read()


def write_file(content: str, path: str) -> None:
    """ Write contents to file """
    log.debug(f"Writing file at {path}")
    with open(path, "w") as f:
        f.write(content)


def df_to_parquet(df: pd.DataFrame, path: str) -> None:
    """ Write a dataframe to parquet file """
    log.debug(f"Writing parquet at {path}")
    df.to_parquet(path)


def parquet_to_df(path: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """ Read a parquet file into a dataframe """
    log.debug(f"Reading parquet at {path}")
    if cols:
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_parquet(path)
    return df


def csv_to_df(path: str) -> pd.DataFrame:
    """ Read a .csv file into a dataframe """
    log.debug(f"Loading df from csv at {path}")
    return pd.read_csv(path)


def load_yaml(path: str) -> Any:
    """ Read the contents of a yaml file safely """
    log.debug(f"Loading YAML from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_url_to_file(url: str, path: str) -> None:
    """ Fetch the file at url to path """

    log.debug(f"Fetching file from {url} to {path}")
    with requests.get(url, stream=True) as response:
        # total_length = response.headers.get('content-length')
        with open(path, "wb") as f:
            shutil.copyfileobj(response.raw, f)


def create_directory(path: str) -> None:
    """ Create a directory """
    if not os.path.isdir(path):
        log.debug(f"Creating directory at {path}")
        os.makedirs(path)


def unpack_zipfile(path_zip: str, destination: str) -> List[str]:
    """ Unpack a zipfile """
    log.debug(f"Unpacking {path_zip} to {destination}")
    with zipfile.ZipFile(path_zip, "r") as f:
        msg = f"Extracting {f.namelist()} from {path_zip} to {destination}"
        log.debug(msg)
        f.extractall(path=destination)
        names: List[str] = f.namelist()
    destination_paths: List[str] = [
        os.path.join(destination, name) for name in names
    ]
    return destination_paths


def move_file(path_from: str, path_to: str) -> None:
    """ Move a file """

    log.debug(f"Moving {path_from} to {path_to}")
    shutil.move(path_from, path_to)
