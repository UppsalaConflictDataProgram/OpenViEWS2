""" The views package """

__all__ = [
    "apps",
    "database",
    "specs",
    "utils",
    "ROOTDIR",
    "DATASETS",
    "DIR_STORAGE",
    "DIR_SCRATCH",
    "Model",
    "Ensemble",
    "Period",
    "Downsampling",
    "Transform",
]

import os

from . import apps, config, database, specs, utils
from .apps.model.api import Model, Ensemble, Period, Downsampling
from .apps.data.api import Transform

ROOTDIR = os.path.dirname(__file__)
DIR_STORAGE = config.DIR_STORAGE
DIR_SCRATCH = config.DIR_SCRATCH
GEOMETRIES = specs.data.GEOMETRIES
TABLES = specs.data.TABLES
DATASETS = specs.data.DATASETS


def _setup_dirstructure() -> None:
    """ Setup storage directory structure """
    dirs = [
        DIR_STORAGE,
        os.path.join(DIR_STORAGE, "data", "datasets"),
        os.path.join(DIR_STORAGE, "data", "geometries"),
        os.path.join(DIR_STORAGE, "data", "raw"),
        os.path.join(DIR_STORAGE, "data", "tables"),
        os.path.join(DIR_STORAGE, "logs"),
        os.path.join(DIR_STORAGE, "logs"),
        os.path.join(DIR_STORAGE, "models"),
        os.path.join(DIR_STORAGE, "pipeline", "predictions"),
        os.path.join(DIR_STORAGE, "scratch"),
    ]
    for path_dir in dirs:
        utils.io.create_directory(path_dir)


_setup_dirstructure()
