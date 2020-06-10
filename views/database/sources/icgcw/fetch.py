"""Scrapes all ICG CrisisWatch to file """

# pylint: disable=too-many-arguments

import os
import tempfile
import logging
import datetime

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

from views.utils import io
from views.database import common

log = logging.getLogger(__name__)


def check_if_more_pages(path_html):
    """ True if more pages to fetch indicated in path_html """

    log.debug(f"Checking if more pages in {path_html}")

    with open(path_html, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        search = {
            "class": "c-crisiswatch-entry [ o-container o-container--m u-pr ]"
        }
        matches = soup.find_all("div", search)

        # matches is a list, if len is zero it evaluates to False
        if matches:
            more_pages = True
            log.debug("Found more pages.")
        else:
            more_pages = False
            log.debug("No more pages.")

    return more_pages


def fetch_page_content(url, page, from_year, from_month, to_year, to_month):
    """ Fetch page contents  """
    params = {
        "date_range": "custom",
        "page": page,
        "from_year": from_year,
        "from_month": from_month,
        "to_year": to_year,
        "to_month": to_month,
    }
    headers = {"User-Agent": "Mozilla/5.0"}  # Header because 504 otherwise
    req = requests.get(url=url, params=params, timeout=60, headers=headers)
    log.debug(f"GET {req.url}")
    log.debug(f"Status code: {req.status_code}")
    content = req.content

    return content


def fetch_page_to_file(url, path_dir, page, y_start, m_start, y_end, m_end):
    """ Fetch page at url with time params to file in path_dir """

    # Pad with some zeros
    m_start = str(m_start).zfill(2)
    m_end = str(m_end).zfill(2)

    content = fetch_page_content(url, page, y_start, m_start, y_end, m_end)

    fname = f"{y_start}.{m_start}_{y_end}.{m_end}_p{str(page).zfill(4)}.html"
    path = os.path.join(path_dir, fname)
    with open(path, "wb") as f:
        f.write(content)
    log.info(f"Wrote {path}")

    return path


def fetch_pages(url, path_dir, y_start=2004, m_start=1):
    """ Fetch pages from y_start-m_start until today to path_dir """
    y_end = datetime.date.today().year
    m_end = datetime.date.today().month

    paths = []
    more_pages = True
    page = 0
    while more_pages:
        log.debug(f"Page: {page}")
        path = fetch_page_to_file(
            url, path_dir, page, y_start, m_start, y_end, m_end
        )
        paths.append(path)
        more_pages = check_if_more_pages(path_html=path)
        page = page + 1

    return paths


def fetch_icgcw():
    """ Fetch icgcw to fetch library """
    with tempfile.TemporaryDirectory() as tempdir:
        paths = fetch_pages(
            url="https://www.crisisgroup.org/crisiswatch/database",
            path_dir=tempdir,
        )
        io.make_tarfile(
            paths_members=paths, path_tar=common.get_path_tar(name="icgcw")
        )


if __name__ == "__main__":
    fetch_icgcw()
