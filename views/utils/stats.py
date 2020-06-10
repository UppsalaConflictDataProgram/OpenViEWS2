""" Statistical utils

#@TODO: Figure out numpy / pandas types here
"""
from typing import Any
import warnings

import numpy as np  # type: ignore


def prob_to_odds(p: Any, clip=True) -> Any:
    """ Cast probability into odds """

    if isinstance(p, list):
        p = np.array(p)

    if clip:
        offset = 1e-10
        offset = 1e-10
        upper = 1 - offset
        lower = 0 + offset
        p = np.clip(p, lower, upper)

    # Check for probs greq 1 because odds of 1 is inf which might break things
    if np.any(p >= 1):
        msg = "probs >= 1 passed to get_odds, expect infs"
        warnings.warn(msg)

    odds = p / (1 - p)
    return odds


def prob_to_logodds(p: Any) -> Any:
    """ Cast probability to log-odds """
    return np.log(prob_to_odds(p))


def odds_to_prob(odds: Any) -> Any:
    """ Cast odds ratio to probability """
    return odds / (odds + 1)


def logodds_to_prob(logodds: Any) -> Any:
    """ Cast logodds to probability """
    return odds_to_prob(np.exp(logodds))
