""" File IO utilities """
from typing import Any, Dict, Optional, List, Union
import logging
import os
import shutil
import tarfile
import zipfile
import json

import bs4  # type: ignore
import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import requests
import yaml


log = logging.getLogger(__name__)


def list_files_in_webdir(url: str) -> List[str]:
    """ List files in a web directory """
    log.debug(f"Listing files in {url}.")
    soup = bs4.BeautifulSoup(requests.get(url).text, "html.parser")
    hrefs = [node.get("href") for node in soup.find_all("a")]
    log.debug(f"Found {len(hrefs)} files in {url}.")
    return hrefs


def dict_to_json(data: Union[Dict[Any, Any], List[Any]], path: str) -> None:
    """ Write data to .json """
    log.debug(f"Writing dict to json at {path}")
    with open(path, "w") as f:
        json.dump(obj=data, fp=f, indent=2)


def load_json(path: str) -> Any:
    """ Load JSON """
    with open(path, "r") as f:
        return json.load(f)


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
    log.debug(f"Reading parquet at {path} with cols {cols}")
    # is not None instead of just if cols
    # because we want to get index only dfs sometimes.
    if cols is not None:
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_parquet(path)
    log.debug(f"Finished reading parquet from {path}.")
    return df


def list_files_in_dir(path_dir: str) -> List[str]:
    """ Lists files in directory """
    return [
        os.path.join(path_dir, fname)
        for fname in os.listdir(path_dir)
        if os.path.isfile(os.path.join(path_dir, fname))
    ]


def list_columns_in_parquet(path: str) -> List[str]:
    """ List the columns in .parquet file at path """
    file = pq.ParquetFile(path)
    colnames = [col.name for col in file.schema]
    return colnames


def csv_to_df(path: str) -> pd.DataFrame:
    """ Read a .csv file into a dataframe """
    log.debug(f"Loading df from csv at {path}")
    return pd.read_csv(path)


def df_to_csv(df: pd.DataFrame, path: str) -> None:
    """ Write a df to a csv """
    log.debug(f"Writing df with shape {df.shape} to {path}")
    df.to_csv(path_or_buf=path)
    log.debug(f"Wrote df with shape {df.shape} to {path}")


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


def make_zipfile(path_zip: str, paths_members: List[str]) -> None:
    """ Compress files at paths_members into path_zip """
    log.debug(f"Compressing to {path_zip} files: {paths_members}")
    with zipfile.ZipFile(
        path_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for path_member in paths_members:
            log.debug(f"Compressing {path_member}")
            zf.write(
                filename=path_member, arcname=os.path.basename(path_member)
            )


def make_tarfile(path_tar: str, paths_members: List[str]) -> None:
    """ Compress files at paths_members into path_tar """

    with tarfile.open(path_tar, mode="w:xz") as f:
        for path in paths_members:
            fname = os.path.basename(path)
            log.debug(f"Compressing {path} to {path_tar} as {fname}")
            f.add(path, arcname=fname)
    log.debug(
        f"Fininshed compressing {len(paths_members)} files to {path_tar}"
    )


def unpack_tarfile(path_tar: str, dir_destination: str) -> List[str]:
    """ Unpack tarfile """
    with tarfile.open(path_tar, "r") as f:
        members = f.getmembers()
        log.debug(
            f"unpacking tarfile at {path_tar} "
            f"with {len(members)} members to {dir_destination}"
        )
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, path=dir_destination)
    return [os.path.join(dir_destination, member.name) for member in members]


def move_file(path_from: str, path_to: str) -> None:
    """ Move a file """
    log.debug(f"Moving {path_from} to {path_to}")
    shutil.move(path_from, path_to)


def delete_file(path: str) -> None:
    """ Delete file """
    if os.path.isfile(path):
        log.debug(f"Deleting existing file at {path}")
        os.remove(path)


def gdf_to_geojson(gdf: gpd.GeoDataFrame, path: str) -> None:
    """ Write GeoDataFrame to geojson file """

    if not path.endswith(".geojson"):
        # Geopanas induces formats by extension
        raise TypeError(
            f"gdf_to_geojson needs a path ending in .geojson, not {path}"
        )

    # Complains if file exists so delete it first.
    delete_file(path)
    # Doesn't write index by default so reset it for it to be included
    gdf.reset_index().to_file(filename=path, driver="GeoJSON")


def geojson_to_gdf(
    path: str, groupvar: Optional[str] = None, geom_col="geom",
) -> gpd.GeoDataFrame:
    """ Read a geojson to a geodataframe """

    gdf = gpd.read_file(path, driver="GeoJSON")
    gdf = gdf.rename(columns={"geometry": "geom"})
    if groupvar:
        gdf = gdf.set_index([groupvar]).sort_index()
    gdf = gdf.set_geometry(geom_col)

    return gdf


def gdf_to_shp(gdf: gpd.GeoDataFrame, path: str) -> None:
    """ Write geometry and index cols of gdf to path as .shp file """

    log.debug(f"Writing shp {path}")

    if not path.endswith(".shp"):
        raise RuntimeError(f"path:{path} should end in .shp")
    gdf[["geom"]].reset_index().to_file(path)
    log.debug(f"Wrote shp {path}")


def copy_file(path_from: str, path_to: str) -> None:
    """ Copy a file using shutil.copy2, the most high level copy func """
    log.debug(f"Copying file {path_from} to {path_to}")
    shutil.copy2(src=path_from, dst=path_to)
