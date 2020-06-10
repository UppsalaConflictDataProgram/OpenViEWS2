""" Fetch priogrid data from their API """

from typing import Any, Dict, List
import os
import tempfile
import json
import logging
import time
import random
import multiprocessing as mp

import requests

from views.utils import io
from views.database import common

log = logging.getLogger(__name__)

URL_BASE = "https://grid.prio.org/api"


def fetch_variable(
    varinfo: Dict[Any, Any], dir_destination: str, try_number: int = 1
) -> str:
    """ Fetch a single variable from API """

    url = varinfo["url"]
    params = varinfo["payload"]
    log.debug(f"Fetching {url} with params {params} try_number {try_number}")

    try:
        data = requests.get(url=url, params=params).json()
    except json.decoder.JSONDecodeError:
        time.sleep(2 ** try_number + random.random() * 0.01)
        data = fetch_variable(
            varinfo, dir_destination, try_number=try_number + 1
        )

    path = os.path.join(dir_destination, f"{varinfo['name']}.json")
    io.dict_to_json(data, path)
    return path


def fetch_data(
    varinfos: List[Dict[Any, Any]], dir_destination: str
) -> List[str]:
    """ Fetch all the data to dir_destination"""

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = []
        for varinfo in varinfos:
            results.append(
                pool.apply_async(fetch_variable, (varinfo, dir_destination,))
            )
        paths = [result.get() for result in results]

        return paths


def fetch_varinfos() -> List[Dict[Any, Any]]:
    """ Update varinfo dictionaries with API endpoint URLs """

    varinfos = requests.get(f"{URL_BASE}/variables").json()
    varinfos = varinfos.copy()
    for varinfo in varinfos:
        url = f"{URL_BASE}/data/{varinfo['id']}"
        if varinfo["type"] == "yearly":
            payload = {k: varinfo[k] for k in ("startYear", "endYear")}
        elif varinfo["type"] == "static":
            payload = {}

        varinfo.update({"url": url, "payload": payload})

    return varinfos


def fetch_pgdata() -> None:
    """ Fetch priogrid data from API """

    path_tar = common.get_path_tar(name="pgdata")

    log.info("Started fetching pgdata")

    grid = requests.get(f"{URL_BASE}/data/basegrid").json()
    varinfos = fetch_varinfos()

    with tempfile.TemporaryDirectory() as tempdir:

        path_grid = os.path.join(tempdir, "basegrid.json")
        path_varinfos = os.path.join(tempdir, "varinfos.json")
        io.dict_to_json(data=grid, path=path_grid)
        io.dict_to_json(data=varinfos, path=path_varinfos)
        paths_data = fetch_data(varinfos=varinfos, dir_destination=tempdir)

        paths_all = paths_data + [path_varinfos] + [path_grid]

        io.make_tarfile(path_tar=path_tar, paths_members=paths_all)

    log.info("Finished fetching pgdata")
