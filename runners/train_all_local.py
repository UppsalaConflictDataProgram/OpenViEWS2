""" Train all models locally """

import argparse
import os
import sys
import logging
from typing import Tuple

from views import DATASETS
from views.apps.model import api
from views.apps.pipeline import models_cm, models_pgm
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


def main() -> None:
    do_pgm, do_cm = parse_args()

    if do_pgm:
        for model in models_pgm.all_pgm_models:
            df = DATASETS["flat_pgm_africa_1"].df
            model.fit_estimators(df)
            model.save()

    if do_cm:
        for model in models_cm.all_cm_models:
            if "train_africa" in model.tags:
                df = DATASETS["flat_cm_africa_1"].df
            elif "train_global" in model.tags:
                df = DATASETS["flat_cm_global_1"].df
            model.fit_estimators(df)
            model.save()


if __name__ == "__main__":
    try:
        main()
    except:
        log.exception(f"Training failed for some reason.")
