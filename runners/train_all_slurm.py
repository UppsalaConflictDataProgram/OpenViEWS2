""" Train all models via slurm """

import argparse
import os
import sys
import logging
from typing import Tuple

from views.apps.model import api
from views.apps.pipeline import models_cm, models_pgm
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

log = logging.getLogger(__name__)


def parse_args() -> Tuple[bool, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgm", action="store_true", help="Train PGM models?")
    parser.add_argument("--cm", action="store_true", help="Train CM models?")
    args = parser.parse_args()

    return args.pgm, args.cm


def _build_cmd_train_model(model: api.Model, dataset: str, loa: str) -> str:
    path_runner = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train_model.py"
    )
    path_exec = sys.executable
    cmd = (
        f"{path_exec} {path_runner} "
        f"--model {model.name} "
        f"--dataset {dataset} "
        f"--loa {loa} "
    )
    return cmd


def main() -> None:

    train_pgm, train_cm = parse_args()

    # CM
    if train_cm:
        log.info(f"--cm was passed, training all CM models.")
        for model in models_cm.all_cm_models:
            if "train_africa" in model.tags:
                cmd = _build_cmd_train_model(
                    model, dataset="flat_cm_africa_1", loa="cm"
                )
            elif "train_global" in model.tags:
                cmd = _build_cmd_train_model(
                    model, dataset="flat_cm_global_1", loa="cm"
                )
            run_command(cmd)

    # PGM
    if train_pgm:
        log.info(f"--pgm was passed, training all pgm models.")
        for model in models_pgm.all_pgm_models:
            cmd = _build_cmd_train_model(
                model, dataset="flat_pgm_africa_1", loa="pgm"
            )
            run_command(cmd, hours=48)


if __name__ == "__main__":
    main()
