""" Specification of datasets and transformations """
from typing import Any, Dict, Union

from views.apps.data import api

from . import parsed_datasets


def build_geometries() -> Dict[str, Any]:
    """ Just expose our custom geometries as dict to be consistent """
    geometries = {
        "GeomPriogrid": api.GeomPriogrid(),
        "GeomCountry": api.GeomCountry(),
    }
    return geometries


GEOMETRIES: Dict[
    str, Union[api.GeomPriogrid, api.GeomCountry]
] = build_geometries()
TABLES: Dict[str, api.Table] = parsed_datasets.build_tables()
DATASETS: Dict[str, api.Dataset] = parsed_datasets.build_datasets()
