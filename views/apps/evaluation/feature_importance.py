"""Feature importances module"""

from typing import List, Dict
import os
import logging
from datetime import datetime

import pandas as pd  # type: ignore
import joblib  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore


log = logging.getLogger(__name__)


def get_feature_importance_from_pickle(
    path_pickle: str, features: List[str], period: str, step: int
) -> Dict[str, float]:
    """ Get feature importance from pickle at path.

    Args:
        path: Path to pickled RandomForestRegressor.
        features: List of feature names.
        period: Which period (str).
        step: Which step (int).
    Returns:
        fi_dict: A dictionary of feature importance scores.
    """
    fi_dict = {}
    if os.path.isfile(path_pickle):
        log.debug(f"Started reading {path_pickle}")
        try:
            model = joblib.load(path_pickle)
            model = model.estimators[period][step]
            log.debug(f"Finished reading {path_pickle}")
            # Only populate if it's a RandomForestRegressor
            if isinstance(model, RandomForestRegressor):
                importances = model.feature_importances_
                for feature, value in zip(features, importances):
                    fi_dict[feature] = value

        except EOFError:
            log.warning(f"Couldn't read {path_pickle}")

    return fi_dict


def reorder_fi_dict(fi_dict: Dict[str, float], top: int = None) -> Dict:
    """ Get feature importances in an ordered (desc) table and write .tex.

    Args:
        fi_dict: Dictionary of feature importances, {feature: importance}.
        top (optional): Top number of feature importances to include.
    Returns:
        fi_dict: Ordered tab dictionary of feature importance scores, i.e.
            {"feature": [features], "importance": [importances]}.
    """
    desc = dict(
        sorted(fi_dict.items(), key=lambda item: item[1], reverse=True)
    )

    top_desc = {k: desc[k] for k in list(desc)[:top]} if top else desc

    featimps_tabular = {
        "feature": [k for k, v in top_desc.items()],
        "importance": [v for k, v in top_desc.items()],
    }

    return featimps_tabular


def write_fi_tex(df: pd.DataFrame, path: str):
    """ Write feature importances df to .tex with info added.

    Args:
        df: pd.DataFrame containing importances per row, indexed on feature.
        path: Full path including filename to write .tex to.
    """
    tex = df.to_latex()
    # Add meta information.
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    meta = f"""
    %Output created by feature_importance.py.
    %Produced on {now}, written to {path}.
    \\
    """
    tex = meta + tex

    with open(path, "w") as f:
        f.write(tex)
    log.info(f"Wrote feature importances to .tex under {path}.")
