""" Parsers for Table and Dataset Dicts from spec.yaml """
from typing import Dict
import os
from views.utils import io
from views.apps.data.api import Dataset, Table
from . import solver


def build_tables() -> Dict[str, Table]:
    """ Build Table objects from spec.yaml in this dir """
    specs = io.load_yaml(os.path.join(os.path.dirname(__file__), "spec.yaml"))
    # Build tables dict
    tables: Dict[str, Table] = dict()
    for fqtable, spec in specs["tables"].items():
        tables[fqtable] = Table(fqtable=fqtable, ids=spec["ids"])

    return tables


def build_datasets() -> Dict[str, Dataset]:
    """ Build Datasets from spec.yaml in this dir """
    specs = io.load_yaml(os.path.join(os.path.dirname(__file__), "spec.yaml"))
    tables: Dict[str, Table] = build_tables()

    # Build transformsets dict
    datasets: Dict[str, Dataset] = dict()
    for name, spec in specs["datasets"].items():
        dataset = Dataset(
            name=name,
            ids=spec["ids"],
            table_skeleton=tables[spec["table_skeleton"]],
            tables=[tables[table] for table in spec["tables"]],
            loa=spec["loa"],
            cols=spec["cols"] if "cols" in spec.keys() else None,
            transforms=solver.make_transforms_ordered(spec["transforms"]),
            balance=spec["balance"],
        )
        datasets[name] = dataset

    return datasets
