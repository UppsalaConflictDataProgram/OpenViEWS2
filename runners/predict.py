import argparse
import os
import sys
import logging
from typing import Tuple, List

from views import DATASETS
from views.apps.model import api
from views.apps.pipeline import (
    predict,
    models_cm,
    models_pgm,
    ensembles_cm,
    ensembles_pgm,
)
from views.config import LOGFMT
from views.utils.log import get_log_path
from views.utils.data import assign_into_df


logging.basicConfig(
    level=logging.DEBUG,
    format=LOGFMT,
    handlers=[
        logging.FileHandler(get_log_path(__file__)),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


def predict_cm_models(run_id: str, n_cores: int) -> None:
    """ Predict with all CM models """
    dataset = DATASETS["cm_africa_imp_0"]
    models = models_cm.all_cm_models
    predict.predict_models(
        models=models, dataset=dataset, run_id=run_id, n_cores=n_cores
    )


def predict_pgm_models(run_id: str, n_cores: int) -> None:
    """ Predict with all PGM models """
    dataset = DATASETS["pgm_africa_imp_0"]
    models = models_pgm.all_pgm_models
    predict.predict_models(
        models=models, dataset=dataset, run_id=run_id, n_cores=n_cores
    )


def predict_cm_ensembles(run_id: str, n_cores: int) -> None:
    """ Predict with all CM ensembles """
    ensembles = ensembles_cm.all_cm_ensembles
    dataset = DATASETS["cm_africa_imp_0"]
    predict.predict_ensembles(ensembles, dataset, run_id, n_cores=n_cores)


def predict_pgm_ensembles(run_id: str, n_cores: int) -> None:
    """ Predict with all PGM ensembles """
    ensembles = ensembles_pgm.all_pgm_ensembles
    dataset = DATASETS["pgm_africa_imp_0"]
    predict.predict_ensembles(ensembles, dataset, run_id, n_cores=n_cores)


def predict_pgm_ensembles_and_constituent(run_id: str, n_cores: int) -> None:
    """ Predict all PGM ensembles and their constituent models """
    log.info(f"Predicting PGM ensembles and their constituent models")
    ensembles = ensembles_pgm.all_pgm_ensembles
    models: List[api.Model] = []
    for ensemble in ensembles:
        for model in ensemble.models:
            if not any([m for m in models if m.name == model.name]):
                models.append(model)

    dataset = DATASETS["pgm_africa_imp_0"]
    predict.predict_models(
        models=models, dataset=dataset, run_id=run_id, n_cores=n_cores
    )
    predict.predict_ensembles(
        ensembles=ensembles, dataset=dataset, run_id=run_id, n_cores=n_cores
    )


def predict_cm_ensembles_and_constituent(run_id: str, n_cores: int) -> None:
    """ Predict all cm ensembles and their constituent models """
    log.info(f"Predicting CM ensembles and their constituent models")
    ensembles = ensembles_cm.all_cm_ensembles
    models: List[api.Model] = []
    for ensemble in ensembles:
        for model in ensemble.models:
            if not any([m for m in models if m.name == model.name]):
                models.append(model)

    dataset = DATASETS["cm_africa_imp_0"]
    predict.predict_models(
        models=models, dataset=dataset, run_id=run_id, n_cores=n_cores
    )
    predict.predict_ensembles(
        ensembles=ensembles, dataset=dataset, run_id=run_id, n_cores=n_cores
    )


def parse_args() -> Tuple[str, bool, bool, bool, bool, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pgm", action="store_true", help="Predict PGM models?"
    )
    parser.add_argument("--cm", action="store_true", help="Predict CM models?")
    parser.add_argument(
        "--run_id", type=str, help="Run ID to predict for", required=True
    )
    parser.add_argument(
        "--model", action="store_true", help="Make model predictions"
    )
    parser.add_argument(
        "--ensemble", action="store_true", help="Make ensemble predictions"
    )
    parser.add_argument("--n_cores", type=int, choices=range(0, 40), default=4)
    args = parser.parse_args()

    return (
        args.run_id,
        args.pgm,
        args.cm,
        args.model,
        args.ensemble,
        args.n_cores,
    )


def main():
    run_id, do_pgm, do_cm, do_model, do_ensemble, n_cores = parse_args()
    log.info(
        f"predict running with flags "
        f"run_id {run_id} do_pgm {do_pgm} do_cm {do_cm} do_model "
        f"{do_model} do_ensemble {do_ensemble}"
    )

    if do_model and do_ensemble:
        if do_cm:
            predict_cm_ensembles_and_constituent(run_id, n_cores)
        if do_pgm:
            predict_pgm_ensembles_and_constituent(run_id, n_cores)
    elif do_model:
        if do_cm:
            predict_cm_models(run_id, n_cores)
        if do_pgm:
            predict_pgm_models(run_id, n_cores)
    elif do_ensemble:
        if do_cm:
            predict_cm_ensembles(run_id, n_cores)
        if do_pgm:
            predict_pgm_ensembles(run_id, n_cores)
    else:
        log.info(f"Nothing to do! Run predict.py --help to show args.")


if __name__ == "__main__":
    try:
        main()
    except:
        log.exception(f"Something broke")
        raise
