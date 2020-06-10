""" Defines which columns go into which models """
from typing import Dict, List
import os
from views.utils import io
from views.apps.model import api


def get_periods(run_id: str) -> List[api.Period]:
    """ Get periods for a particular run as list """
    _this_dir = os.path.dirname(__file__)
    spec = io.load_yaml(os.path.join(_this_dir, "periods.yaml"))["runs"]

    spec_run = spec[run_id]
    periods = []
    for period_name, data in spec_run.items():
        period = api.Period(
            name=period_name,
            train_start=data["train"]["start"],
            train_end=data["train"]["end"],
            predict_start=data["predict"]["start"],
            predict_end=data["predict"]["end"],
        )
        periods.append(period)
    return periods


def get_periods_by_name(run_id: str) -> Dict[str, api.Period]:
    """ Get periods for a particular run as name-index dict """
    periods_list = get_periods(run_id)
    periods_by_name = dict()
    for period in periods_list:
        periods_by_name[period.name] = period

    return periods_by_name


__all__ = ["get_periods", "get_periods_by_name"]
