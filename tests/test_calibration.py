import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from views.apps.model.calibration import calibrate_real
from views.utils import mocker


def test_calibrate_real_perfect_on_calib() -> None:
    """ Test that calibrated values when test=calib match perfectly """

    df = mocker.DfMocker(datatypes=["reals"]).df

    s_calibrated = calibrate_real(
        s_test_pred=df["r_a"],
        s_calib_pred=df["r_a"],
        s_calib_actual=df["r_b"],
    )

    assert np.isclose(s_calibrated.mean(), df["r_b"].mean())
    assert np.isclose(s_calibrated.std(), df["r_b"].std())


def test_calibrate_real_scales_right_way() -> None:
    """ Test that calibration shifts mean the right way """

    calib_pred = [100, 200]  # <- Off by factor 0.5
    calib_actual = [50, 100]
    test_pred = [200, 400]
    test_expected = [100, 200]  # <- test_pred * 0.5

    s_calibrated = calibrate_real(
        s_test_pred=pd.Series(test_pred),
        s_calib_pred=pd.Series(calib_pred),
        s_calib_actual=pd.Series(calib_actual),
    )

    assert all(s_calibrated == pd.Series(test_expected))
