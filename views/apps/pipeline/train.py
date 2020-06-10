""" This module defines the training of all models used in ViEWS

After it is run, all required models should be persisted on disk and
ready for prediction.
"""
import logging

from typing_extensions import Literal

from views.specs.data import DATASETS
from . import models_cm, models_pgm

log = logging.getLogger(__name__)


def train_and_store_model_by_name(
    loa: Literal["am", "cm", "pgm"], model: str, dataset: str
) -> None:
    """ Lookup a model by name and fit, evaluate and store it """

    if loa == "cm":
        model_object = models_cm.all_cm_models_by_name[model]
    elif loa == "pgm":
        model_object = models_pgm.all_pgm_models_by_name[model]
    else:
        raise NotImplementedError(f"cm and pgm models only yet, not {loa}")

    df = DATASETS[dataset].df
    model_object.fit_estimators(df)
    model_object.save()
