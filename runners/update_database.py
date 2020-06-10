""" Update a source in the database """
from typing import Tuple
import argparse
import logging
import views
from views.database.sources import (
    acled,
    cdum,
    fvp,
    ged,
    icgcw,
    pgdata,
    reign,
    spei,
    vdem,
    wdi,
)

logging.basicConfig(
    level=logging.DEBUG,
    format=views.config.LOGFMT,
    handlers=[
        logging.FileHandler(views.utils.log.get_log_path(__file__)),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


def parse_args() -> Tuple[
    bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool
]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nofetch", action="store_true", help="No fetch, only load."
    )
    parser.add_argument("--wdi", action="store_true", help="Update WDI")
    parser.add_argument("--vdem", action="store_true", help="Update VDEM")
    parser.add_argument("--acled", action="store_true", help="Update ACLED")
    parser.add_argument("--ged", action="store_true", help="Update GED")
    parser.add_argument("--icgcw", action="store_true", help="Update ICGCW")
    parser.add_argument(
        "--pgdata", action="store_true", help="Update Priogrid"
    )
    parser.add_argument("--spei", action="store_true", help="Update SPEI")
    parser.add_argument("--fvp", action="store_true", help="Update FVP")
    parser.add_argument(
        "--cdum", action="store_true", help="Update country dummies"
    )
    parser.add_argument("--reign", action="store_true", help="Upate reign")

    args = parser.parse_args()

    return (
        args.nofetch,
        args.wdi,
        args.vdem,
        args.acled,
        args.ged,
        args.icgcw,
        args.pgdata,
        args.spei,
        args.fvp,
        args.cdum,
        args.reign,
    )


def main():

    (
        nofetch,
        do_wdi,
        do_vdem,
        do_acled,
        do_ged,
        do_icgcw,
        do_pgdata,
        do_spei,
        do_fvp,
        do_cdum,
        do_reign,
    ) = parse_args()

    if do_wdi:
        if not nofetch:
            wdi.fetch_wdi()
        wdi.load_wdi()

    if do_vdem:
        if not nofetch:
            vdem.fetch_vdem()
        vdem.load_vdem()

    if do_acled:
        if not nofetch:
            acled.fetch_acled()
        acled.load_acled()

    if do_ged:
        if not nofetch:
            ged.fetch_ged()
        ged.load_ged()

    if do_icgcw:
        if not nofetch:
            icgcw.fetch_icgcw()
        icgcw.load_icgcw()

    if do_pgdata:
        if not nofetch:
            pgdata.fetch_pgdata()
        pgdata.load_pgdata()

    if do_spei:
        if not nofetch:
            spei.fetch_spei()
        spei.load_spei()

    if do_fvp:
        if not nofetch:
            fvp.fetch_fvp()
        fvp.load_fvp()

    if do_cdum:
        if not nofetch:
            cdum.fetch_cdum()
        cdum.load_cdum()

    if do_reign:
        if not nofetch:
            reign.fetch_reign()
        reign.load_reign()


if __name__ == "__main__":
    try:
        main()
    except:
        log.exception(f"Something went wrong in update_database.py")
