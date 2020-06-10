""" Train on slurm """

import argparse
import os
import sys
import logging
from typing import Tuple, List

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


def parse_args() -> Tuple[bool, bool, bool, List[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgm", action="store_true", help="Train PGM models?")
    parser.add_argument("--cm", action="store_true", help="Train CM models?")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models for selected LOAs?",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Train a particular model. Pass multiple times for multiple models",
    )
    args = parser.parse_args()

    if args.all and args.models:
        raise RuntimeError(f"Can't have --all and --model")

    # We don't know which LOA to train for
    if args.model and args.cm and args.pgm:
        raise RuntimeError(f"Can't have --model, --cm and --pgm")

    return args.pgm, args.cm, args.all, args.model


def _build_cmd_train_model(modelname: str, dataset: str, loa: str) -> str:
    path_runner = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train_model.py"
    )
    path_exec = sys.executable
    cmd = (
        f"{path_exec} {path_runner} "
        f"--model {modelname} "
        f"--dataset {dataset} "
        f"--loa {loa} "
    )
    return cmd


def main() -> None:

    pgm, cm, train_all, modelnames = parse_args()

    if modelnames:
        for modelname in modelnames:
            if pgm:
                if not modelname in models_pgm.all_pgm_models_by_name:
                    raise RuntimeError(f"Couldn't find model name {modelname}")
                cmd = _build_cmd_train_model(
                    modelname, dataset="flat_pgm_africa_1", loa="pgm",
                )
            elif cm:
                # Check we have model
                if not modelname in models_cm.all_cm_models_by_name:
                    raise RuntimeError(f"Couldn't find model name {modelname}")

                model = models_cm.all_cm_models_by_name[modelname]
                if "train_africa" in model.tags:
                    cmd = _build_cmd_train_model(
                        model.name, dataset="flat_cm_africa_1", loa="cm"
                    )
                elif "train_global" in model.tags:
                    cmd = _build_cmd_train_model(
                        model.name, dataset="flat_cm_global_1", loa="cm"
                    )
            run_command(cmd)

    # CM
    if cm and train_all:
        log.info(f"--cm and --all was passed, training all CM models.")
        for model in models_cm.all_cm_models:
            if "train_africa" in model.tags:
                cmd = _build_cmd_train_model(
                    model.name, dataset="flat_cm_africa_1", loa="cm"
                )
            elif "train_global" in model.tags:
                cmd = _build_cmd_train_model(
                    model.name, dataset="flat_cm_global_1", loa="cm"
                )
            run_command(cmd)

    # PGM
    if pgm and train_all:
        log.info(f"--pgm and --all was passed, training all pgm models.")
        for model in models_pgm.all_pgm_models:
            cmd = _build_cmd_train_model(
                model.name, dataset="flat_pgm_africa_1", loa="pgm"
            )
            run_command(cmd, hours=48)


if __name__ == "__main__":
    main()
