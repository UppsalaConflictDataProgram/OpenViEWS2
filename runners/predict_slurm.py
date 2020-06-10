""" Train all models via slurm """
import argparse
import os
import sys
import logging
from typing import Tuple

from views.apps.slurm.slurm import run_command
from views.config import LOGFMT
from views.utils.log import get_log_path

logging.basicConfig(
    level=logging.DEBUG,
    format=LOGFMT,
    handlers=[
        logging.FileHandler(get_log_path(__file__)),
        logging.StreamHandler(),
    ],
)


def parse_args() -> Tuple[str, bool, bool, int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgm", action="store_true", help="Predict PGM?")
    parser.add_argument("--cm", action="store_true", help="Predict CM?")
    parser.add_argument(
        "--run_id", type=str, help="Run ID to predict for", required=True
    )
    parser.add_argument("--n_cores", type=int, choices=range(0, 40), default=4)
    parser.add_argument("--hours", type=int, choices=range(0, 128), default=24)
    args = parser.parse_args()

    return args.run_id, args.pgm, args.cm, args.n_cores, args.hours


def _build_command(loa: str, run_id: str, n_cores: int) -> str:
    path_runner = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "predict.py"
    )
    path_exec = sys.executable
    return f"{path_exec} {path_runner} --{loa} --run_id {run_id} --model --ensemble --n_cores {n_cores}"


def main() -> None:

    run_id, do_pgm, do_cm, n_cores, hours = parse_args()

    if do_cm:
        cmd = _build_command(loa="cm", run_id=run_id, n_cores=n_cores)
        run_command(cmd, hours=hours)

    if do_pgm:
        cmd = _build_command(loa="pgm", run_id=run_id, n_cores=n_cores)
        run_command(cmd, hours=hours)


if __name__ == "__main__":
    main()
