""" Common utils for database data management """
import logging
from typing import List, Optional
import tempfile
import os
from datetime import date

from views.utils import io
from ..config import DIR_STORAGE

DIR_FETCHES = os.path.join(DIR_STORAGE, "data", "raw")
log = logging.getLogger(__name__)


def get_path_tar(name: str) -> str:
    """ Get a path to a tarfile timestamped for today """
    io.create_directory(DIR_FETCHES)
    today = date.today().strftime("%Y%m%d")
    return os.path.join(DIR_FETCHES, f"{name}_{today}.tar.xz")


def fetch_source_simply(
    name: str, url: Optional[str] = None, urls: Optional[List[str]] = None
) -> None:
    """ Download file at url (or urls) and store in tarfile by name """

    def _get_urls(url: Optional[str], urls: Optional[List[str]]) -> List[str]:
        """ If url return list of one, else pass through urls """
        if url and urls:
            raise TypeError("Use url or urls, not both.")
        if url:
            # pylint: disable=redefined-argument-from-local
            urls = [url]
        assert isinstance(urls, list)

        return urls

    urls = _get_urls(url, urls)
    with tempfile.TemporaryDirectory() as tempdir:
        paths = []
        for url in urls:  # pylint: disable=redefined-argument-from-local
            fname = url.split("/")[-1]
            path_source = os.path.join(tempdir, fname)
            io.fetch_url_to_file(url, path=path_source)
            paths.append(path_source)
        io.make_tarfile(path_tar=get_path_tar(name), paths_members=paths)


def get_files_latest_fetch(name, tempdir) -> List[str]:
    """ Get files from latest fetch

    Unpack the tarfile for the latest fetch for source name into tempdir
    and return paths.
    """
    log.debug(f"Getting files for latest fetch for {name}")
    paths_fetches = io.list_files_in_dir(path_dir=DIR_FETCHES)
    try:
        path_tar = [
            path
            for path in sorted(paths_fetches)
            if os.path.basename(path).startswith(name)
        ].pop(0)
        log.debug(f"Got {path_tar} as latest {name} of {paths_fetches}")
    except IndexError:
        log.exception(f"Couldn't find a latest fetch for {name}.")
        raise

    paths = io.unpack_tarfile(path_tar=path_tar, dir_destination=tempdir)
    return paths
