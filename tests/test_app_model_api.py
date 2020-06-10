import os
import tempfile

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
import pytest  # type: ignore

from views.apps.model import api
from views.utils import mocker


def test_model_constructor() -> None:
    """ Test the basic Model constructor """

    model1 = api.Model(
        name="test",
        col_outcome="test_outcome",
        cols_features=["col_a", "col_b"],
        steps=[1, 3, 6],
        outcome_type="real",
    )

    assert isinstance(model1, api.Model)
    assert model1.name == "test"
    assert model1.col_outcome == "test_outcome"
    assert model1.steps == [1, 3, 6]
    assert model1.outcome_type == "real"
    assert model1.periods == []
    assert model1.scores == dict()
    assert model1.downsampling == None


def test_period_constructor() -> None:
    period = api.Period(
        name="testname",
        train_start=1,
        train_end=10,
        predict_start=11,
        predict_end=20,
    )
    assert period.name == "testname"
    assert period.train_start == 1
    assert period.train_end == 10
    assert period.predict_start == 11
    assert period.predict_end == 20
    assert period.times_train == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert period.times_predict == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def test_outcome_type_raise() -> None:
    """ Test that model constructor rejects broken params """

    with pytest.raises(NotImplementedError) as excinfo:
        api.Model(
            name="test",
            col_outcome="test_outcome",
            cols_features=["col_a", "col_b"],
            steps=[1, 3, 6],
            outcome_type="BROKEN",  # type: ignore
        )
        assert "BROKEN not in allowed_outcome_types" in str(excinfo.value)


# @TODO: Do we want this?
def dont_test_estimator_period_raise() -> None:
    """ Model constructor should reject when estimator but no periods """

    with pytest.raises(RuntimeError) as excinfo:
        api.Model(
            name="test",
            col_outcome="test_outcome",
            cols_features=["col_a", "col_b"],
            steps=[1, 3, 6],
            outcome_type="prob",
            estimator=RandomForestRegressor(),
        )
        assert "An estimator was passed but no periods" in str(excinfo.value)


def test_fit_probs() -> None:
    df = mocker.DfMocker(n_t=31).df
    period_a = api.Period(
        name="p_a",
        train_start=1,
        train_end=10,
        predict_start=11,
        predict_end=20,
    )
    period_b = api.Period(
        name="p_b",
        train_start=1,
        train_end=10,
        predict_start=21,
        predict_end=30,
    )
    model = api.Model(
        name="test",
        col_outcome="b_a",
        cols_features=["c_a", "r_a"],
        steps=[1, 3, 6],
        outcome_type="real",
        estimator=RandomForestClassifier(n_estimators=10),
        periods=[period_a, period_b],
    )
    model.fit_estimators(df)
    df_pred = model.predict(df)
    assert isinstance(df_pred, pd.DataFrame)
    assert list(df_pred.columns) == [
        "ss_test_1",
        "ss_test_3",
        "ss_test_6",
        "sc_test",
    ]


# @TODO: do we want this?
def test_persistence() -> None:
    """ Test that saving and loading a model yields identical objects """
    model1 = api.Model(
        name="test",
        col_outcome="test_outcome",
        cols_features=["col_a", "col_b"],
        steps=[1, 3, 6],
        outcome_type="real",
    )
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "model.pickle")
        model1.save(path)
        model2 = api.Model.load(path)

    assert model1.__dict__ == model2.__dict__


def test_persistence_fitted() -> None:
    """ Test that saving and loading a model yields identical objects """
    est = RandomForestClassifier()

    df = mocker.DfMocker(n_t=31).df
    # Predict for 11-20
    p1 = api.Period(
        name="same_name",
        train_start=1,
        train_end=10,
        predict_start=11,
        predict_end=20,
    )

    model1 = api.Model(
        name="test",
        col_outcome="b_a",
        cols_features=["r_a", "p_a"],
        steps=[1, 3, 6],
        outcome_type="real",
        periods=[p1],
        estimator=est,
    )
    # model1.fit_estimators(df)
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "model.pickle")
        model1.fit_estimators(df)
        model1.save(path)
        model2 = api.Model.load(path)

    # Check all components of model except estimators
    # because estimators are never equal after unpickling.
    exclude = ["initial_estimator", "estimators"]
    for key in model1.__dict__.keys():
        if key not in exclude:
            try:
                assert model1.__dict__[key] == model2.__dict__[key]
            except AssertionError:
                print(key)
                raise

    # Test that estimators are actually stored as they should
    # by comparing predictions, if they predict the same
    # they are the same
    pred_1 = model1.predict(df)
    pred_2 = model2.predict(df)
    assert (pred_1 == pred_2).all(axis=None)


def test_predict_periods() -> None:
    """ Test predicting for other periods than the initalial ones """

    # Data for t_0 to t_30
    df = mocker.DfMocker(n_t=31).df
    # Predict for 11-20
    p1 = api.Period(
        name="same_name",
        train_start=1,
        train_end=10,
        predict_start=11,
        predict_end=20,
    )
    # Predict for 21-30
    p2 = api.Period(
        name="same_name",
        train_start=1,
        train_end=10,
        predict_start=21,
        predict_end=30,
    )
    # Predict for 21-30 but not matching train times
    p_mismatch_train = api.Period(
        name="same_name",
        train_start=1,
        train_end=9,
        predict_start=21,
        predict_end=30,  # Note non-matching train_end
    )

    model = api.Model(
        name="test",
        col_outcome="b_a",
        cols_features=["c_a", "r_a"],
        steps=[1, 3, 6],
        outcome_type="real",
        estimator=RandomForestClassifier(n_estimators=10),
        periods=[p1],
    )
    model.fit_estimators(df)

    df_p1 = model.predict(df, period=p1)
    df_p2 = model.predict(df, period=p2)

    assert list(df_p1.index.levels[0].values) == p1.times_predict
    assert list(df_p2.index.levels[0].values) == p2.times_predict

    # Make sure we warn when the training periods don't match
    with pytest.warns(UserWarning):
        _ = model.predict(df, period=p_mismatch_train)


def dont_test_estimatorcollection() -> None:
    est = RandomForestClassifier()
    with tempfile.TemporaryDirectory() as td:
        est_col = api.EstimatorCollection(
            name="whatever", estimator=est, dir_storage=td
        )
        assert isinstance(
            est_col.get(period_name="whatever", step=5), RandomForestClassifier
        )


@pytest.mark.filterwarnings("ignore:Beta_1 < 0")
def test_calibrated() -> None:
    """ Test that calibration raises calibrated probs """

    df = mocker.DfMocker(n_t=31).df
    p_calib = api.Period(
        name="calib",
        train_start=1,
        train_end=10,
        predict_start=11,
        predict_end=20,
    )
    # Predict for 21-30
    p_test = api.Period(
        name="test",
        train_start=1,
        train_end=10,
        predict_start=21,
        predict_end=30,
    )
    # inflate p_calib actuals
    df.loc[13:20, "b_a"] = 1
    with tempfile.TemporaryDirectory() as td:
        model = api.Model(
            name="test",
            col_outcome="b_a",
            cols_features=["c_a", "r_a"],
            steps=[1, 3, 6],
            outcome_type="prob",
            estimator=RandomForestClassifier(n_estimators=10),
            periods=[p_calib, p_test],
            dir_storage=td,
        )
        model.fit_estimators(df)
        df_calibrated = model.predict_calibrated(
            df, period_calib=p_calib, period_test=p_test
        )
        df_p_test_raw = model.predict(df, p_test)
        # Check that inflated actuals in calib raises predictions in test
        assert df_calibrated.mean().mean() > df_p_test_raw.mean().mean()


def test_sc_from_ss():
    """ Test that sc_from_ss makes correctly interpolated predictions """
    cols_ss = {1: "s1", 3: "s3", 5: "s5"}
    period = api.Period(
        name="test",
        train_start=1,
        train_end=1,
        predict_start=11,
        predict_end=15,
    )
    times = pd.Series(list(range(11, 16)), name="time")
    groups = pd.Series([1, 2], name="group")
    ix = pd.MultiIndex.from_product([times, groups])
    s1 = pd.Series(np.repeat(1, 10), index=ix)
    s3 = pd.Series(np.repeat(3, 10), index=ix)
    s5 = pd.Series(np.repeat(5, 10), index=ix)
    df = pd.DataFrame({"s1": s1, "s3": s3, "s5": s5,}, index=ix).sort_index()
    df["s1"] = s1
    df["s3"] = s3
    df["s5"] = s5
    df["sc"] = api.sc_from_ss(df, cols_ss, period)
    # Check it's the interpolation
    assert all(df["sc"].values == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])


def test_warns_not_disjoint() -> None:
    p_a = api.Period(
        name="calib",
        train_start=1,
        train_end=1,
        predict_start=2,
        predict_end=5,
    )
    # Shares 5 in predict times
    p_b = api.Period(
        name="test",
        train_start=1,
        train_end=1,
        predict_start=5,
        predict_end=8,
    )
    with pytest.warns(UserWarning):
        model = api.Model(
            name="test",
            col_outcome="b_a",
            cols_features=["c_a", "r_a"],
            steps=[1, 3, 6],
            outcome_type="prob",
            periods=[p_a, p_b],
        )


def test_onset():
    """ Test that model doesn't error when onset_outcome=True """
    # Data for t_0 to t_30
    df = mocker.DfMocker(n_t=31).df
    # Predict for 11-20
    p = api.Period(
        name="p",
        train_start=1,
        train_end=20,
        predict_start=21,
        predict_end=30,
    )

    model = api.Model(
        name="test",
        col_outcome="b_a",
        cols_features=["c_a", "r_a"],
        steps=[1, 3, 6],
        outcome_type="prob",
        estimator=RandomForestClassifier(n_estimators=10),
        periods=[p],
        onset_outcome=True,
        onset_window=3,
    )
    model.fit_estimators(df)

    df_pred = model.predict(df)
