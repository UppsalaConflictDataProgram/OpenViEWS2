""" Model API

This is the modelling API for ViEWS.
Model specification has gone through many iterations in ViEWS.
The aim of this specification is to provide an extremely easy to use
interface for programmers to sanely specify and organise
models and create predictions for them using the step-ahead a.k.a.
step shifting method.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional, Tuple
import copy
import json
import logging
import os
import warnings

from typing_extensions import Literal

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.inspection import permutation_importance  # type: ignore

import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from views.apps.evaluation import lib as evallib
from views.apps.transforms import lib as translib
from views.apps.ensemble import run_ebma
from views.utils import data as datautils, misc as miscutils, io
from views import config
from . import calibration

log = logging.getLogger(__name__)

DIR_STORAGE_MODELS = os.path.join(config.DIR_STORAGE, "models")
if not os.path.isdir(DIR_STORAGE_MODELS):
    io.create_directory(DIR_STORAGE_MODELS)


# pylint: disable= too-few-public-methods
class NoEstimator:
    """ The default to EstimatorCollection estimator when no estimator
    is passed in. Perhaps the user is expecting to load it from a
    joblib file but that file is missing.
    """

    def __init__(self):
        pass

    # pylint: disable= no-self-use
    def predict(self, X):
        """ Inform user that they didn't pass an estimator """
        raise RuntimeError(
            "NoEstimator's can't do anything. "
            "Did you forget to pass an estimator to Model?"
        )


@dataclass
class Period:
    """ Defines a time period of training and predicting. """

    name: str
    train_start: int
    train_end: int
    predict_start: int
    predict_end: int

    @property
    def times_train(self) -> List[int]:
        """ A list of time ids for training """
        return list(range(self.train_start, self.train_end + 1))

    @property
    def times_predict(self) -> List[int]:
        """ A list of time ids for predicting """
        return list(range(self.predict_start, self.predict_end + 1))


@dataclass
class Downsampling:
    """ Defines a downsampling strategy. """

    share_positive: float = 1.0
    share_negative: float = 1.0
    threshold: Union[float, int] = 0


class EstimatorCollection:
    """ Transparent storage of estimator objects in pickles """

    def __init__(
        self, name: str, estimator: Any, dir_storage: str = DIR_STORAGE_MODELS,
    ) -> None:
        self.name = name
        self.initial_estimator = copy.deepcopy(estimator)
        self.dir_storage = dir_storage

    def __eq__(self, other):
        """ For testability save/restore of Models """
        return self.name == other.name and isinstance(
            self.initial_estimator, type(other.initial_estimator)
        )

    def _path(self, period_name: str, step: int) -> str:
        return os.path.join(
            DIR_STORAGE_MODELS, f"{self.name}_{period_name}_{step}.joblib",
        )

    def store(self, period_name: str, step: int, estimator: Any) -> None:
        """ Store the estimator on disk """
        path = self._path(period_name, step)
        log.debug(f"Saving {self.name} {period_name} {step} to {path}")
        with open(path, "wb") as file:
            joblib.dump(estimator, file, compress=3)
            log.debug(f"{self.name} saved to {path}")

    def get(self, period_name: str, step: int) -> Any:
        """ Get estimator from disk """
        path = self._path(period_name, step)
        if os.path.isfile(path):
            log.debug(f"Loading {path}")
            with open(path, "rb") as f:
                try:
                    return joblib.load(f)
                except ValueError:
                    log.error(f"ValueError: Couldn't load file at {path}")
                    raise
        else:
            raise RuntimeError(f"No estimator stored at {path}")

    def get_initial(self):
        """ Get a copy of initial (probably unfitted) estimator """
        log.debug(f"Getting initial_estimator for {self.name}")
        return copy.deepcopy(self.initial_estimator)


def sc_from_ss(
    df: pd.DataFrame, cols_ss: Dict[int, str], period: Period
) -> pd.Series:
    """ Combine step-specific predictions into step-combined """
    s_sc = pd.Series(dtype="float64", index=df.index)
    # s_sc = datautils.rebuild_index(s_sc.loc[period.times_predict])
    for step in cols_ss.keys():
        col_ss = cols_ss[step]
        t = period.predict_start + step - 1
        if t in period.times_predict:
            s_sc.loc[t] = df.loc[t, col_ss].values

    # Interpolate
    s_sc.loc[period.times_predict] = s_sc.groupby(level=1).apply(
        lambda group: group.interpolate(limit_direction="both")
    )
    return s_sc


# pylint: disable=too-many-instance-attributes, too-many-arguments
class Model:
    """ The model object

    Models organise everything needed to create and evaluate constituent
    model predictions.

    Provides an interface for

    * Fitting estimators for each step and period.
    * Persisting those estimators for re-use.
    * Creating uncalibrated predictions using step shifting.
    * Creating calibrated predictions using step shifting and
      calib/test period pairs
    * Evaluating both calibrated and uncalibrated predictions
    * Expose column names to be used in Ensembles

    Several model types are supported.

    * Any scikit-learn compatible estimator can be used. The only requirement
      is that it expose a `.fit(X, y)` method and a `.predict(X)` or a
      `.predict_proba(X)` method.
    * outcome_type, `"real"` or `"prob"`, selects the appropriate
      calibration functions and evaluation metrics
    * `onset_outcome` and `onset_window` allow onset models to be trained.
      If `onset_outcome` is `True` the outcome is transformed to an onset
      of itself before training. Training data is subset to only those
      rows where an onset is possible, so that rows in ongoing
      conflicts are not included. `onset_window` determines the time
      window when considering onsets.
    * `delta_outcome` allows training on delta transformed outcomes.
      The outcome in the training data is delta transformed by `step` for each
      estimator before training. So for step=1 a delta transform is
      applied to the outcome with time=1.



    Args:
        name: str, A descriptive name. Must be unique across other
            files or it will overwrite.
        col_outcome: str, The outcome column to train on
        cols_features: List[str], Features to train on
        steps: List[int], List of steps to train for
        outcome_type: Literal["real", "prob"], Outcome type.
        periods: Optional[List[Period]] = None, Which periods to train for
        downsampling: Optional[Downsampling] = None, Downsampling spec
        estimator: Optional[Any] = None, Estimator object
        delta_outcome: bool = False, Transform outcome to
            step-specific delta before training?
        onset_outcome: bool = False, Transform outcome to an onset
            before training?
        onset_window: Optional[int] = None, Time window for onset
            transform. Subsets training data to rows where onset
            is possible
        dir_storage: str = config.DIR_STORAGE, Optional storage
            directory specification. Probably don't use.
        tags: Optional[List[str]] = None, List of descriptive tags
            can be useful for filtering models later or training
            on specific datasets. Example "train_africa".



    """

    # @TODO: Figure out type annotations,
    # even Guido struggled: https://github.com/python/mypy/issues/1212
    @staticmethod
    def load(path: str):
        """ Load an instance of Model from file """
        with open(path, "rb") as file:
            obj = joblib.load(file)

        if isinstance(obj, Model):
            log.info(f"{obj.name} loaded from {path}")
        else:
            raise RuntimeError(
                f"{path} does not contain an instance of Model."
            )

        return obj

    # It's complicated, leave me alone pylint
    # pylint: disable=too-many-locals
    def __init__(
        self,
        name: str,
        col_outcome: str,
        cols_features: List[str],
        steps: List[int],
        outcome_type: Literal["real", "prob"],
        periods: Optional[List[Period]] = None,
        downsampling: Optional[Downsampling] = None,
        estimator: Optional[Any] = None,
        delta_outcome: bool = False,
        onset_outcome: bool = False,
        onset_window: Optional[int] = None,
        dir_storage: str = config.DIR_STORAGE,
        tags: Optional[List[str]] = None,
    ) -> None:

        # Check outcome type
        allowed_outcome_types = ["real", "prob"]
        if outcome_type not in allowed_outcome_types:
            raise NotImplementedError(
                f"outcome_type {outcome_type} not in allowed_outcome_types"
            )
        self.outcome_type = outcome_type

        self.tags = tags if tags else []

        self.name = name
        self.col_outcome = col_outcome
        self.cols_features = sorted(cols_features)
        self.steps = steps
        self.periods = periods if periods else []
        self.downsampling = downsampling
        self.delta_outcome = delta_outcome
        self.dir_storage = dir_storage

        self.estimators: EstimatorCollection
        if estimator is not None:
            self.estimators = EstimatorCollection(
                name=self.name, estimator=estimator
            )
        else:
            self.estimators = EstimatorCollection(
                name=self.name, estimator=NoEstimator()
            )

        # Prediction column naming convention
        # ss Denotes step specific
        self.cols_ss: Dict[int, str] = {
            step: f"ss_{name}_{step}" for step in steps
        }
        self.cols_ss_calibrated: Dict[int, str] = {
            step: f"ss_{name}_{step}_calibrated" for step in steps
        }  # sc denotes step combined, usually being interpolated ss
        self.col_sc: str = f"sc_{name}"
        self.col_sc_calibrated: str = f"sc_{name}_calibrated"

        # For holding extra attributes such as
        # feature importances or parameter estimates
        self.extras: Extras = Extras(self)
        self.evaluation: Evaluation = Evaluation(self)

        self.onset_outcome = onset_outcome
        self.onset_window = onset_window
        self._check_params()

    def _check_params(self):
        if not miscutils.lists_disjoint(
            [period.times_predict for period in self.periods]
        ):
            warnings.warn(
                f"Predict periods for model {self.name} aren't disjoint. "
                f"This will lead to predictions overwriting each other."
            )

        if self.onset_outcome:
            # Check onset outcome isn't combined with delta
            if self.delta_outcome:
                raise TypeError("delta_outcome and onset_outcome both True")
            # Check onset outcome IS combined with onset_window
            if not self.onset_window:
                raise TypeError("onset_outcome requires onset_window")
            # Check onset_outcome combined with outcome_type prob
            if not self.outcome_type == "prob":
                raise TypeError("onset_outcome only makes sense for probs")

        if not len(self.cols_features) == len(list(set(self.cols_features))):
            raise TypeError(
                f"There are duplicates in "
                f"cols_features {self.cols_features}"
            )

    def __str__(self):
        return self.as_json

    def __repr__(self):
        return self.as_json

    @property
    def as_json(self) -> str:
        """ Get the json serializable parts of the model as json string """
        stringable_attributes = {
            "name": self.name,
            "col_outcome": self.col_outcome,
            "cols_features": self.cols_features,
            "steps": self.steps,
            "periods": self.periods,
            "outcome_type": self.outcome_type,
            "estimators": self.estimators,
            "downsampling": self.downsampling,
            "delta_outcome": self.delta_outcome,
            "dir_storage": self.dir_storage,
            "tags": self.tags,
            "onset_outcome": self.onset_outcome,
            "onset_window": self.onset_window,
        }
        return json.dumps(
            stringable_attributes, default=lambda x: x.__dict__, indent=2
        )

    @property
    def _default_path(self) -> str:
        return os.path.join(self.dir_storage, "models", f"{self.name}.joblib")

    def _fit_estimator(
        self, df: pd.DataFrame, period: Period, step: int
    ) -> None:
        """ Fit and persist an estimator """

        def delta_transform_outcome(
            s_outcome: pd.Series, step: int
        ) -> pd.Series:
            """ delta-transform the outcome column
            useful for prediction-competition type outcomes where a
            step-specific delta transformation of the outcome is required
            """
            log.debug("delta_outcome, transforming outcome")
            return translib.delta(s=s_outcome, time=step)

        def get_onset_impos(s_outcome: pd.Series, window: int) -> pd.Series:
            """ Get bool Series of whether onset impossible """
            s_onset_pos = translib.onset_possible(s_outcome, window)
            s_onset_impos = ~s_onset_pos.astype(bool)
            return s_onset_impos

        # Subset times first before shifting
        df = df.loc[period.train_start : period.train_end]

        # Shift features
        df_step = df[self.cols_features].groupby(level=1).shift(step)
        s_outcome: pd.Series = df[self.col_outcome].copy()

        if self.delta_outcome:
            s_outcome = delta_transform_outcome(s_outcome, step)

        elif self.onset_outcome and self.onset_window:
            log.debug(f"{self.name} has onset_outcome, transforming outcome")
            s_onset_impos = get_onset_impos(s_outcome, self.onset_window)
            s_outcome = translib.onset(s_outcome, window=self.onset_window)

        # Don't shift outcome
        df_step[self.col_outcome] = s_outcome

        if self.onset_outcome:
            # Drop all rows where onset not possible
            df_step = df_step.drop(df_step.loc[s_onset_impos].index)

        df_step = df_step.dropna()

        if self.downsampling:
            log.debug(f"Downsampling by {self.downsampling} for {self.name}")
            len_pre = len(df_step)
            df_step = datautils.resample(
                df=df_step,
                cols=[self.col_outcome],
                share_positives=self.downsampling.share_positive,
                share_negatives=self.downsampling.share_negative,
                threshold=self.downsampling.threshold,
            )
            log.debug(f"{self.name} downsampled away {len_pre - len(df_step)}")

        log.debug(f"Fitting {self.name} on {len(df_step)} rows")
        estimator = self.estimators.get_initial()
        estimator = estimator.fit(
            X=df_step[self.cols_features], y=df_step[self.col_outcome]
        )
        self.estimators.store(period.name, step, estimator)

    def fit_estimators(
        self, df: pd.DataFrame, populate_extras: bool = True
    ) -> None:
        """ Fit all estimators for each step in each period """

        self.check_df_has_cols(df, check_outcome=True)

        if not self.periods:
            raise RuntimeError(f"{self.name} doesn't have any periods set.")

        log.info(f"Fitting estimators for {self.name}")
        for period in self.periods:
            for step in self.steps:
                log.debug(
                    f"Fitting {self.name} for period {period.name} step {step}"
                )
                self._fit_estimator(df, period, step)

        if populate_extras:
            self.extras.populate(df)

    def populate_extras(self, df):
        """ Populate the extras if this wasn't done during fitting """

        if not self.extras.populated:
            log.debug(f"Extras for {self.name} not populated, populating now.")
            self.extras.populate(df)
        else:
            log.debug(f"Extras for {self.name} already populated")

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
        # estimator = self.estimators[period.name][step]
        estimator = self.estimators.get(period.name, step)
        if self.outcome_type == "real":
            s_pred.loc[ix_pred] = _predict_real(estimator, df_X)
        elif self.outcome_type == "prob":
            s_pred.loc[ix_pred] = _predict_prob(estimator, df_X)

        return s_pred

    def predict_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Deprecated predict method """
        warnings.warn("predict_steps is deprecated, use predict")
        return self.predict(df)

    def _check_period_times_train_match(self, period: Period) -> None:
        """ Check that the training times of supplied period matches the
        training times of the instance's period with the same name
        """

        try:
            own_matching_period = [
                p for p in self.periods if p.name == period.name
            ][0]
        except IndexError:
            log.exception(
                "Couldn't find a matching period, check period names"
            )
            raise
        if not period.times_train == own_matching_period.times_train:
            warnings.warn(
                f"Periods times_train don't match for period {period.name}"
            )

    def check_df_has_cols(self, df: pd.DataFrame, check_outcome=False) -> None:
        """ Check all necessary cols in df """
        cols_missing = [col for col in self.cols_features if col not in df]
        if check_outcome and self.col_outcome not in df:
            cols_missing.append(self.col_outcome)
        if any(cols_missing):
            raise RuntimeError(
                f"Model {self.name} is missing cols {cols_missing}"
            )

    def predict(
        self,
        df: pd.DataFrame,
        period: Optional[Period] = None,
        periods: Optional[List[Period]] = None,
    ) -> pd.DataFrame:
        """ Predict """

        self.check_df_has_cols(df)

        if period and periods:
            raise TypeError("predict takes period or periods, not both.")

        # If a single period stick it in a list of one
        if period:
            periods = [period]
        # If neither period nor periods passed, use models own periods
        if not periods:
            periods = self.periods

        log.info(f"Predicting for {self.name}")
        log.debug(f"Predicting for {self.name} periods: {periods}")

        dfs_per_period = []
        # pylint: disable=redefined-argument-from-local
        for period in periods:
            self._check_period_times_train_match(period)
            df_ss = pd.concat(
                [self._predict(df, period, step) for step in self.steps],
                axis=1,
            )
            df_ss[self.col_sc] = sc_from_ss(df_ss, self.cols_ss, period)
            dfs_per_period.append(df_ss.copy())

        df_predictions = pd.concat(objs=dfs_per_period, axis=0)
        df_predictions = datautils.rebuild_index(df_predictions)

        return df_predictions

    def predict_calibrated(
        self,
        df: pd.DataFrame,
        period_calib: Period,
        period_test: Period,
        col_outcome: Optional[str] = None,
    ) -> pd.DataFrame:
        """ Make calibrated predictions """

        log.info(
            f"Predicting calibrated for {self.name} "
            f"period_calib: {period_calib.name} "
            f"period_test: {period_test.name} "
        )

        self.check_df_has_cols(df)

        # Allow other col_outcome than the default one trained on
        if not col_outcome:
            col_outcome = self.col_outcome

        df_calib = self.predict(df, period=period_calib)
        df_test = self.predict(df, period=period_test)
        df_calibrated = df.loc[period_test.times_predict, []]

        for step in self.steps:

            s_calib_actual = df.loc[:, col_outcome]

            # Do nothing or delta or onset
            if self.delta_outcome:
                s_calib_actual = translib.delta(s_calib_actual, time=step)
            elif self.onset_outcome and self.onset_window:
                s_calib_actual = translib.onset(
                    s=s_calib_actual, window=self.onset_window
                )

            # Subset actuals to calib predict times
            s_calib_actual = s_calib_actual.loc[period_calib.times_predict]

            log.debug(f"Calibrating {self.name} step {step}")

            # Get calib and test predictions
            s_test_pred = df_test[self.cols_ss[step]]
            s_calib_pred = df_calib[self.cols_ss[step]]

            # Choose the appropriate calibration function
            if self.outcome_type == "prob":
                calib_func = calibration.calibrate_prob
            elif self.outcome_type == "real":
                calib_func = calibration.calibrate_real
            else:
                warnings.warn(
                    f"Don't have a calibration function "
                    f"matching {self.outcome_type}. Returning empty df"
                )
                return pd.DataFrame()

            s_calibrated_step = calib_func(
                s_test_pred, s_calib_pred, s_calib_actual
            )
            df_calibrated[
                self.cols_ss_calibrated[step]
            ] = s_calibrated_step.values

        df_calibrated[self.col_sc_calibrated] = sc_from_ss(
            df_calibrated, self.cols_ss_calibrated, period_test
        )

        return df_calibrated

    def evaluate(
        self,
        df: pd.DataFrame,
        period: Optional[Period] = None,
        periods: Optional[List[Period]] = None,
    ) -> None:
        """ Evaluate, optionaly subsetting by period or periods """
        self.evaluation.evaluate(df, period, periods)

    @property
    def scores(self):
        """ Get evaluation scores """
        return self.evaluation.scores

    def save(self, path: Optional[str] = None) -> None:
        """ Save the entire model object, including estimators, in a pickle """
        if not path:
            path = self._default_path
        log.debug(f"Saving {self.name} to {path}")
        with open(path, "wb") as file:
            joblib.dump(self, file, compress=3)
            log.info(f"{self.name} saved to {path}")


class Ensemble:
    """ Define an Ensemble as a collection of Models """

    def __init__(
        self,
        name: str,
        models: List[Model],
        method: Literal["average", "ebma", "crosslevel"],
        outcome_type: Literal["prob", "real"],
        col_outcome: str,
        periods: List[Period],
        delta_outcome: bool = False,
        onset_outcome: bool = False,
        onset_window: Optional[int] = None,
    ):
        self.models: List[Model] = models
        self.name = name
        known_methods = ["average", "ebma", "crosslevel"]
        if method not in known_methods:
            raise TypeError(
                f"Unrecognised method {method}, "
                f"only {known_methods} understood."
            )
        self.method = method
        self.outcome_type = outcome_type
        self.col_outcome = col_outcome

        self.delta_outcome = delta_outcome
        if self.delta_outcome:
            raise NotImplementedError(
                "delta_outcome not implemented yet for ensembles"
            )

        self.steps: List[int] = self.steps_in_common(models)
        self.cols_ss: Dict[int, str] = {
            step: f"ss_{name}_{step}" for step in self.steps
        }
        self.cols_ss_calibrated = self.cols_ss
        self.col_sc = f"sc_{name}"
        # All ensemble predictions are calibrated
        self.col_sc_calibrated = self.col_sc
        self.weights: Dict[str, Dict[int, Dict[str, float]]] = dict()

        self.periods = periods
        self.evaluation = Evaluation(self)

        # @TODO: Add onset transforms to ensembles
        self.onset_outcome = onset_outcome
        self.onset_window = onset_window
        if self.onset_outcome or self.onset_window:
            raise NotImplementedError(
                "Sorry, onset not implemented yet for ensembles"
            )

    @property
    def cols_needed(self) -> List[str]:
        """ Get a list of cols needed """
        cols = []
        # EBMA needs uncalibrated constituents and an outcome
        if self.method == "ebma":
            cols.append(self.col_outcome)
            for model in self.models:
                for col in model.cols_ss.values():
                    cols.append(col)
        # Average needs calibrated constituent
        elif self.method == "average":
            for model in self.models:
                for col in model.cols_ss_calibrated.values():
                    cols.append(col)
        return cols

    @property
    def scores(self):
        """ Get evaluation scores """
        return self.evaluation.scores

    @staticmethod
    def steps_in_common(models: List[Model]):
        """ Find steps that all models have in common """
        return sorted(
            set.intersection(*[set(model.steps) for model in models])
        )

    def _predict_average(
        self, df: pd.DataFrame, period_calib: Period, period_test: Period
    ) -> pd.DataFrame:
        """ Compute simple unweighted average predictions """
        df_pred = df.loc[period_test.times_predict, []]

        self.weights[period_calib.name] = dict()
        for step in self.steps:
            cols = [model.cols_ss_calibrated[step] for model in self.models]
            df_pred[self.cols_ss[step]] = df.loc[
                period_test.times_predict, cols
            ].mean(axis=1)

            # Equal weights
            self.weights[period_calib.name][step] = {
                model.name: 1 / len(self.models) for model in self.models
            }

        return df_pred

    def _predict_ebma(
        self, df: pd.DataFrame, period_calib: Period, period_test: Period
    ) -> pd.DataFrame:
        """ Compute EBMA weights and make predictions """
        df_pred = df.loc[period_test.times_predict, []]

        self.weights[period_calib.name] = dict()
        for step in self.steps:
            cols = [model.cols_ss[step] for model in self.models]

            missing_cols = [col for col in cols if col not in df.columns]
            if any(missing_cols):
                raise RuntimeError(
                    f"Missing cols for {self.name}: {missing_cols}"
                )

            s_ebma, weights_ebma = run_ebma(
                df_calib=df.loc[period_calib.times_predict, cols],
                df_test=df.loc[period_test.times_predict, cols],
                s_calib_actual=df.loc[
                    period_calib.times_predict, self.col_outcome
                ],
            )

            weights_renamed = {
                model.name: weights_ebma[model.cols_ss[step]]
                for model in self.models
            }
            self.weights[period_calib.name][step] = weights_renamed.copy()

            df_pred[self.cols_ss[step]] = s_ebma
        return df_pred

    def check_df_has_cols(self, df):
        """ Check df has all needed cols """
        cols_missing = [col for col in self.cols_needed if col not in df]
        if any(cols_missing):
            raise RuntimeError(
                f"Ensemble {self.name} missing cols {cols_missing}"
            )

    def predict(
        self, df: pd.DataFrame, period_calib: Period, period_test: Period
    ) -> pd.DataFrame:
        """ Predict using method determined by the method param """
        log.info(f"Predicting for {self.name}")

        if self.method == "average":
            df_pred = self._predict_average(df, period_calib, period_test)
        elif self.method == "ebma":
            df_pred = self._predict_ebma(df, period_calib, period_test)

        df_pred.loc[period_test.times_predict, self.col_sc] = sc_from_ss(
            df=df_pred, cols_ss=self.cols_ss, period=period_test
        )
        return df_pred

    def evaluate(
        self,
        df: pd.DataFrame,
        period: Optional[Period] = None,
        periods: Optional[List[Period]] = None,
    ) -> None:
        """ Evaluate, optionaly subsetting by period or periods """
        self.evaluation.evaluate(df, period, periods)


class Evaluation:
    """ Evaluation class that holds all evaluation related logic """

    def __init__(self, model: Union[Model, Ensemble]):
        self.model = model
        self.delta_outcome = model.delta_outcome

        # Initialize a period.step/sc.uncalibrated/calib dict for scores
        self.scores: Dict[str, Any] = dict()
        for period in self.model.periods:
            self.scores[period.name] = dict()
            for step in self.model.steps:
                self.scores[period.name][step] = dict()
                self.scores[period.name][step]["uncalibrated"] = dict()
                self.scores[period.name][step]["calibrated"] = dict()
            self.scores[period.name]["sc"] = dict()
            self.scores[period.name]["sc"]["uncalibrated"] = dict()
            self.scores[period.name]["sc"]["calibrated"] = dict()

    def __eq__(self, other):
        return self.scores == other.scores

    @staticmethod
    def _get_scores_real(s_actual, s_prediction, delta_outcome):
        """ Get available scores for a real-value prediction """
        scores = dict()
        scores["mse"] = evallib.mean_squared_error(
            actuals=s_actual, preds=s_prediction
        )
        scores["mae"] = evallib.mean_absolute_error(
            actuals=s_actual, preds=s_prediction
        )
        scores["r2"] = evallib.r2_score(actuals=s_actual, preds=s_prediction)

        if delta_outcome:  # TODO(Remco): Epsilon assumes log.
            scores["tadda_score"] = evallib.tadda_score(
                y_deltas=s_actual, f_deltas=s_prediction, epsilon=0.048
            )

        return scores

    @staticmethod
    def _get_scores_probs(s_actual, s_prediction):
        """ Get available scores for a probability prediction """

        scores = dict()

        scores["average_precision"] = evallib.average_precision(
            actuals=s_actual, probs=s_prediction
        )
        scores["area_under_roc"] = evallib.area_under_roc(
            actuals=s_actual, probs=s_prediction
        )
        scores["brier"] = evallib.brier(actuals=s_actual, probs=s_prediction)

        return scores

    def make_eval_data(
        self,
        df: pd.DataFrame,
        period: Period,
        step: int,
        calibrated: bool,
        step_combined: bool,
    ) -> Tuple[pd.Series, pd.Series]:
        """ Make evaluation data

        The function

        * Gets predictions by period
        * Gets calibrated if calibrated is true, else uncalibrated
        * Subsets the data so that only rows with both prediction and
            actual values are included
        * Returns actuals and predictions

        """

        if step_combined:
            if calibrated:
                s_pred = df.loc[
                    period.times_predict, self.model.col_sc_calibrated
                ].copy()
            else:
                s_pred = df.loc[period.times_predict, self.model.col_sc].copy()
        else:
            if calibrated:
                s_pred = df.loc[
                    period.times_predict, self.model.cols_ss_calibrated[step]
                ].copy()
            else:
                s_pred = df.loc[
                    period.times_predict, self.model.cols_ss[step]
                ].copy()

        s_actual = df.loc[:, self.model.col_outcome].copy()

        if self.model.delta_outcome:
            s_actual = translib.delta(s_actual, time=step)

        if self.model.onset_outcome and self.model.onset_window:
            s_actual = translib.onset(s_actual, window=self.model.onset_window)

        s_actual = s_actual.loc[period.times_predict]

        # Concat and drop missing
        df_eval = pd.concat([s_actual, s_pred], axis=1).dropna()
        s_pred = df_eval[s_pred.name]
        s_actual = df_eval[s_actual.name]

        return s_actual, s_pred

    def evaluate(
        self,
        df: pd.DataFrame,
        period: Optional[Period] = None,
        periods: Optional[List[Period]] = None,
    ) -> None:
        """ Evaluate model for all periods and steps """

        if period and periods:
            raise TypeError(
                "evaluate() takes period or periods or None, not both."
            )
        # If a single period stick it in a list of one
        if period:
            periods = [period]
        # If neither period nor periods passed, use models own periods
        if not periods:
            periods = self.model.periods

        log.info(f"Evaluating {self.model.name}")
        # pylint: disable=redefined-argument-from-local
        for period in periods:
            # Evaluate sc predictions, assuming step=1 for delta_outcome on sc
            log.debug(
                f"Evaluating uncalibrated predictions for "
                f"{self.model.name} period {period.name} step-combined"
            )
            s_actual, s_pred = self.make_eval_data(
                df, period, step=1, calibrated=False, step_combined=True
            )
            if self.model.outcome_type == "prob":
                self.scores[period.name]["sc"][
                    "uncalibrated"
                ] = self._get_scores_probs(s_actual, s_pred)
            elif self.model.outcome_type == "real":
                self.scores[period.name]["sc"][
                    "uncalibrated"
                ] = self._get_scores_real(s_actual, s_pred, self.delta_outcome)
            # Evaluate calibrated sc predictions, if produced
            if self.model.col_sc_calibrated in df.columns:
                s_actual, s_pred = self.make_eval_data(
                    df, period, step=1, calibrated=True, step_combined=True
                )
                # And we have some predictions for this period
                if len(s_pred) > 0:
                    log.debug(
                        f"Evaluating calibrated predictions "
                        f"for {self.model.name} "
                        f"period {period.name} step-combined"
                    )
                    if self.model.outcome_type == "prob":
                        self.scores[period.name]["sc"][
                            "calibrated"
                        ] = self._get_scores_probs(s_actual, s_pred)
                    elif self.model.outcome_type == "real":
                        self.scores[period.name]["sc"][
                            "calibrated"
                        ] = self._get_scores_real(
                            s_actual, s_pred, self.delta_outcome
                        )
            # Evaluate ss predictions by period and step
            for step in self.model.steps:
                log.debug(
                    f"Evaluating uncalibrated predictions for "
                    f"{self.model.name} period {period.name} step {step}"
                )
                s_actual, s_pred = self.make_eval_data(
                    df, period, step, calibrated=False, step_combined=False
                )
                if self.model.outcome_type == "prob":
                    self.scores[period.name][step][
                        "uncalibrated"
                    ] = self._get_scores_probs(s_actual, s_pred)
                elif self.model.outcome_type == "real":
                    self.scores[period.name][step][
                        "uncalibrated"
                    ] = self._get_scores_real(
                        s_actual, s_pred, self.delta_outcome
                    )
                # If any calibrated predictions made
                if self.model.cols_ss_calibrated[step] in df.columns:
                    s_actual, s_pred = self.make_eval_data(
                        df, period, step, calibrated=True, step_combined=False
                    )
                    # And we have some predictions for this period
                    if len(s_pred) > 0:
                        log.debug(
                            f"Evaluating calibrated predictions "
                            f"for {self.model.name} "
                            f"period {period.name} step {step}"
                        )
                        if self.model.outcome_type == "prob":
                            self.scores[period.name][step][
                                "calibrated"
                            ] = self._get_scores_probs(s_actual, s_pred)
                        elif self.model.outcome_type == "real":
                            self.scores[period.name][step][
                                "calibrated"
                            ] = self._get_scores_real(
                                s_actual, s_pred, self.delta_outcome
                            )


class Extras:
    """ All extra metadata functionality specific to certain estimators """

    def __init__(self, model: Model):
        self.model: Model = model

        # Stored as [period.name][step][col_feature]: importance
        self.feature_importances: Optional[
            Dict[str, Dict[int, Dict[str, float]]]
        ] = None

        self.permutation_importances: Optional[Dict[Any, Any]] = None

        self.coefficients: Optional[
            Dict[str, Dict[int, Dict[str, float]]]
        ] = None
        self.regtables: Optional[Dict[str, Dict[int, str]]] = None
        self.populated: bool = False

    def __eq__(self, other):
        """ For testability save/restore of Models """
        return (
            self.feature_importances == other.feature_importances
            and self.coefficients == other.coefficients
            and self.regtables == other.regtables
        )

    def populate(self, df: pd.DataFrame):
        """ Populate all the relevant attributes """
        if isinstance(
            self.model.estimators.initial_estimator,
            (RandomForestClassifier, RandomForestRegressor),
        ):

            try:
                self._get_feature_importances()
            except:  # noqa: E722 # pylint: disable=bare-except
                log.exception(
                    "Something went wrong with feature importances "
                    f"for {self.model.name}."
                )

            try:
                self._get_permutation_importances(df)
            except:  # noqa: E722 # pylint: disable=bare-except
                log.exception(
                    "Something went wrong with permutation importances "
                    f"for {self.model.name}."
                )

        if isinstance(
            self.model.estimators.initial_estimator, LinearRegression
        ):
            try:
                self._get_coefficients()
            except:  # noqa: E722 # pylint: disable=bare-except
                log.exception(
                    "Something went wrong with coefficients "
                    f"for {self.model.name}."
                )
        self.populated = True

    def _get_permutation_importances(self, df: pd.DataFrame) -> None:
        """ Compute permutation importances for train and predict sets

        See: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html # noqa: E501 # pylint: disable=line-too-long
        """

        log.debug(f"Building permutation importances for {self.model.name}")
        self.permutation_importances = dict()

        for period in self.model.periods:
            self.permutation_importances[period.name] = dict()
            for step in self.model.steps:
                log.debug(
                    f"Building permutation importances for {self.model.name}"
                    f"for period {period.name} step {step}"
                )
                self.permutation_importances[period.name][step] = dict()
                pi_dict: Dict[str, Any] = dict()
                pi_dict["train"] = dict()
                pi_dict["test"] = dict()

                estimator = self.model.estimators.get(period.name, step)

                # Shift data
                df_step = (
                    df[self.model.cols_features].groupby(level=1).shift(step)
                )
                s_outcome: pd.Series = df[self.model.col_outcome].copy()
                df_step[self.model.col_outcome] = s_outcome
                df_train = df_step.loc[period.times_train].dropna()
                df_test = df_step.loc[period.times_predict].dropna()

                permi_result_train = permutation_importance(
                    estimator=estimator,
                    X=df_train.loc[:, self.model.cols_features],
                    y=df_train.loc[:, self.model.col_outcome],
                    n_jobs=-1,
                    n_repeats=10,
                )
                permi_result_test = permutation_importance(
                    estimator=estimator,
                    X=df_test.loc[:, self.model.cols_features],
                    y=df_test.loc[:, self.model.col_outcome],
                    n_jobs=-1,
                    n_repeats=10,
                )

                for feature, score in zip(
                    self.model.cols_features,
                    permi_result_train["importances_mean"],
                ):
                    pi_dict["train"][feature] = score
                for feature, score in zip(
                    self.model.cols_features,
                    permi_result_test["importances_mean"],
                ):
                    pi_dict["test"][feature] = score

                self.permutation_importances[period.name][step] = copy.copy(
                    pi_dict
                )

    def _get_feature_importances(self) -> None:

        log.debug(f"Building feature importances for {self.model.name}")
        self.feature_importances = dict()
        for period in self.model.periods:
            self.feature_importances[period.name] = dict()
            for step in self.model.steps:
                log.debug(
                    f"Getting feature importances for {self.model.name}"
                    f"for period {period.name} step {step}"
                )
                fi_dict = dict()
                # estimator = self.model.estimators[period.name][step]
                estimator = self.model.estimators.get(period.name, step)
                for importance, feature in zip(
                    estimator.feature_importances_, self.model.cols_features
                ):
                    fi_dict[feature] = importance
                self.feature_importances[period.name][step] = fi_dict

    def _get_coefficients(self) -> None:

        log.debug(f"Retrieving coefficient estimates for {self.model.name}")
        self.coefficients = dict()
        for period in self.model.periods:
            self.coefficients[period.name] = dict()
            for step in self.model.steps:
                coef_dict = dict()
                estimator = self.model.estimators.get(period.name, step)
                for coefficient, feature in zip(
                    estimator.coef_, self.model.cols_features
                ):
                    coef_dict[feature] = coefficient
                self.coefficients[period.name][step] = coef_dict
