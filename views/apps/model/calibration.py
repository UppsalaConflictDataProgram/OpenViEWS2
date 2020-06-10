""" Calibration """
import logging
from typing import Tuple
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import statsmodels.api as sm  # type: ignore

from views.utils import stats

log = logging.getLogger(__name__)


def _log_missing_indices(s: pd.Series) -> None:
    log.warning(f"Missing indices: {s.loc[s.isnull()].index}")


def calibrate_real(
    s_test_pred: pd.Series, s_calib_pred: pd.Series, s_calib_actual: pd.Series
) -> pd.Series:
    """ Calibrate real value predictions

    Scaling parameters applied would, if applied to s_calib_pred,
    make them near-equal in mean and variance to s_calib_actual.

    For the case of transforming one set to have a given mean and std
    see:
    https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation

    This case is slightly more involved as we want to shift the test
    predictions by parameters "learned" from comparing calibration
    predictions to actuals.


    """

    # Compute standard deviation ratio
    std_ratio = s_calib_actual.std() / s_calib_pred.std()
    # Remoe the calib mean from test predictions
    s_test_demeaned = s_test_pred - s_calib_pred.mean()
    # Shift calib de-meaned test predictions by the calib actual mean
    # And scale to the std ratio
    s_test_pred_scaled = s_calib_actual.mean() + s_test_demeaned * std_ratio

    return s_test_pred_scaled


def calibrate_prob(
    s_test_pred: pd.Series, s_calib_pred: pd.Series, s_calib_actual: pd.Series
) -> pd.Series:
    """ Calibrate s_test_pred

    First predictions are transformed into logodds.
    Then a logit model is fit on
    "actual_outcomes ~ alpha + beta*logodds(p_calib)".
    Then alpha and beta are applied to test predictions like
    A =  e^(alpha+(beta*p_test))
    p_test_calibrated = A/(A+1)

    See: https://en.wikipedia.org/wiki/Logistic_regression

    """

    def _get_scaling_params(
        s_calib_actual: pd.Series, s_calib: pd.Series
    ) -> Tuple[float, float]:
        """ Gets scaling params """

        y = np.array(s_calib_actual)
        intercept = np.ones(len(s_calib))
        X = np.array([intercept, s_calib]).T

        model = sm.Logit(y, X).fit(disp=0)
        beta_0 = model.params[0]
        beta_1 = model.params[1]

        return beta_0, beta_1

    def _apply_scaling_params(
        s_test: pd.Series, beta_0: float, beta_1: float
    ) -> pd.Series:
        """ Scale logodds in s_test using intercept and beta"""
        numerator = np.exp(beta_0 + (beta_1 * s_test))
        denominator = numerator + 1
        scaled_probs = numerator / denominator

        return scaled_probs

    def _check_inputs(
        s_test_pred: pd.Series,
        s_calib_pred: pd.Series,
        s_calib_actual: pd.Series,
    ) -> None:
        """ Check that inputs have valid names and could be proabilities """

        if (
            s_test_pred.min() < 0
            or s_test_pred.max() > 1
            or s_calib_pred.min() < 0
            or s_calib_pred.max() > 1
        ):
            raise RuntimeError(
                "Probabilities outside (0,1) range were passed to calibrate"
            )

        if not s_calib_pred.name == s_test_pred.name:
            warnings.warn(f"{s_calib_pred.name} != {s_test_pred.name}")
        if s_test_pred.isnull().sum() > 0:
            _log_missing_indices(s_test_pred)
            raise RuntimeError("Missing values in s_test_pred")
        if s_calib_pred.isnull().sum() > 0:
            _log_missing_indices(s_calib_pred)
            raise RuntimeError("Missing values in s_calib_pred")
        if s_calib_actual.isnull().sum() > 0:
            _log_missing_indices(s_calib_actual)
            raise RuntimeError("Missing values in s_calib_actual")

        if (
            not len(s_calib_pred) == len(s_calib_actual)
            or len(s_calib_pred.index.difference(s_calib_actual.index)) > 0
        ):
            raise RuntimeError(
                f"len(s_calib_pred): {len(s_calib_pred)} "
                f"len(s_calib_actual): {len(s_calib_actual)} "
                f"index diff: "
                f"{s_calib_pred.index.difference(s_calib_actual.index)}"
                f"s_calib_pred.head() : {s_calib_pred.head()}"
                f"s_calib_pred.tail() : {s_calib_pred.tail()}"
                f"s_calib_actual.head() : {s_calib_actual.head()}"
                f"s_calib_actual.tail() : {s_calib_actual.tail()}"
            )

    _check_inputs(s_test_pred, s_calib_pred, s_calib_actual)

    beta_0, beta_1 = _get_scaling_params(
        s_calib_actual=s_calib_actual,
        s_calib=stats.prob_to_logodds(s_calib_pred.copy()),
    )
    if beta_1 < 0:
        warnings.warn(f"Beta_1 < 0. Very weak {s_calib_pred.name} ?")

    s_test_pred_scaled = _apply_scaling_params(
        stats.prob_to_logodds(s_test_pred.copy()), beta_0, beta_1
    )
    return s_test_pred_scaled
