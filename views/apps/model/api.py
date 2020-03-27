import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import copy

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from views.apps.evaluation import lib as evallib
from views.apps.transforms import lib as translib
from views.utils import data as datautils

log = logging.getLogger(__name__)


@dataclass
class Period:
    name: str
    train_start: int
    train_end: int
    predict_start: int
    predict_end: int

    @property
    def times_train(self) -> List[int]:
        return list(range(self.train_start, self.train_end + 1))

    @property
    def times_predict(self) -> List[int]:
        return list(range(self.predict_start, self.predict_end + 1))


@dataclass
class Downsampling:
    share_positive: float = 1.0
    share_negative: float = 1.0
    threshold: Union[float, int] = 0


class Model:
    def __init__(
        self,
        name: str,
        col_outcome: str,
        cols_features: List[str],
        steps: List[int],
        outcome_type: str,
        periods: Optional[List[Period]] = None,
        downsampling: Optional[Downsampling] = None,
        estimator: Optional[Any] = None,
        delta_outcome: bool = False,
    ):

        self.name = name
        self.col_outcome = col_outcome
        self.cols_features = cols_features
        self.steps = steps
        self.periods = periods if periods else []
        self.downsampling = downsampling
        self.estimator = estimator
        self.delta_outcome = delta_outcome

        allowed_outcome_types = ["real", "prob"]
        if outcome_type not in allowed_outcome_types:
            raise TypeError(
                f"Unrecognized outcome_type {outcome_type} not in allowed_outcome_types"
            )
        self.outcome_type = outcome_type

        # Prediction column naming convention
        # ss Denotes step specific
        self.cols_ss: Dict[int, str] = {
            step: f"ss_{name}_{step}" for step in steps
        }
        # sc denotes step combined, usually being interpolated ss
        self.col_sc: str = f"sc_{name}"

        # Initialize a period.step dict for keeping scores
        self.scores: Dict[str, Any] = dict()
        for period in self.periods:
            self.scores[period.name] = dict()
            for step in self.steps:
                self.scores[period.name][step] = dict()

        # If period and estimators passed,
        # copy the estimator into a period.step dictionary
        if self.periods and estimator is not None:
            self.estimators: Dict[str, Any] = dict()
            for period in self.periods:
                self.estimators[period.name] = dict()
                for step in self.steps:
                    self.estimators[period.name][step] = copy.copy(estimator)

    def _fit_estimator(
        self, df: pd.DataFrame, period: Period, step: int
    ) -> None:
        """ Fit the estimator for a particular period-step """
        # Subset times first before shifting
        df = df.loc[period.train_start : period.train_end]

        # Shift features
        df_step = df[self.cols_features].groupby(level=1).shift(step)

        # Make a Series of the outcome
        s_outcome: pd.Series = df[self.col_outcome]

        # If delta_outcome is passed, transform the outcome column
        # to a a time delta, useful for prediction-competition type
        # outcomes where a step-specific transformation of the
        # outcome is required
        if self.delta_outcome:
            s_outcome = translib.delta(s=s_outcome, time=step)

        # Don't shift outcome
        df_step[self.col_outcome] = s_outcome

        # Drop missing
        df_step = df_step.dropna()

        # Downsample if asked for
        if self.downsampling:
            log.debug(f"Downsampling by {self.downsampling} for {self.name}")
            df_step = datautils.resample(
                df=df_step,
                cols=[self.col_outcome],
                share_positives=self.downsampling.share_positive,
                share_negatives=self.downsampling.share_negative,
                threshold=self.downsampling.threshold,
            )

        self.estimators[period.name][step].fit(
            X=df_step[self.cols_features], y=df_step[self.col_outcome]
        )

    def fit_estimators(self, df: pd.DataFrame) -> None:
        """ Fit all estimators for each step in each period """

        for period in self.periods:
            for step in self.steps:
                log.info(
                    f"Fitting {self.name} for period {period.name} step {step}"
                )
                self._fit_estimator(df, period, step)

    def _predict(
        self, df: pd.DataFrame, period: Period, step: int
    ) -> pd.Series:
        """ Make step specific prediction """

        def _make_ix_pred(df_X, step):
            """ Prediction indices are time shifted step times forward """
            return [(time + step, group) for time, group in df_X.index.values]

        def _predict_real(estimator, df_X) -> Any:
            return estimator.predict(df_X)

        def _predict_prob(estimator: Any, df_X: pd.DataFrame) -> Any:
            """ Get the predicted probability for the outcome=1 case """
            return estimator.predict_proba(df_X)[:, 1]

        # Properly indexed NaN Series to hold predictions
        s_pred = pd.Series(
            data=np.nan,
            index=df.loc[period.times_predict].index,
            name=self.cols_ss[step],
        )

        # Shift the times of features to use for predicting by step
        times_shifted = [t - step for t in period.times_predict]
        df_X = df.loc[times_shifted, self.cols_features].dropna()

        ix_pred = _make_ix_pred(df_X, step)
        estimator = self.estimators[period.name][step]
        if self.outcome_type == "real":
            s_pred.loc[ix_pred] = _predict_real(estimator, df_X)
        elif self.outcome_type == "prob":
            s_pred.loc[ix_pred] = _predict_prob(estimator, df_X)

        return s_pred

    def predict_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Make predictions for all steps """

        return pd.concat(
            objs=[
                self._predict(df, period, step)
                for step in self.steps
                for period in self.periods
            ],
            axis=1,
        )

    def _evaluate_real(self, df: pd.DataFrame) -> None:
        """ Compute all evaluation metrics for real type outputs """

        for period in self.periods:
            for step in self.steps:
                log.debug(f"Evaluating {self.name} {period.name} {step}")
                col_ss = self.cols_ss[step]

                s_prediction = df[col_ss].copy()
                s_actual = df[self.col_outcome].copy()
                if self.delta_outcome:
                    s_actual = translib.delta(s_actual, time=step)

                # Drop any rows missing either actual or prediction
                df_eval = pd.concat([s_prediction, s_actual], axis=1).dropna()
                s_prediction = df_eval[col_ss]
                s_actual = df_eval[self.col_outcome]

                scores = dict()
                scores["mse"] = evallib.mean_squared_error(
                    actuals=s_actual, preds=s_prediction
                )

                self.scores[period.name][step] = copy.copy(scores)

    def evaluate(self, df: pd.DataFrame) -> None:
        """ Perform evaluation of model """

        if self.outcome_type == "real":
            self._evaluate_real(df)
