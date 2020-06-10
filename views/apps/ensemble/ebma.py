""" Wrapper for EBMA """

from typing import Any, Dict, Tuple, List

import logging
import os
import string
import tempfile
import subprocess

import pandas as pd  # type: ignore

from views.utils import data as datautils

log = logging.getLogger(__name__)


def _read_template() -> string.Template:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_dir, "templates", "run_ebma.R"), "r") as f:
        template_str = f.read()

    template = string.Template(template_str)

    return template


def _run_subproc(cmd: List[str]) -> None:
    """ Run cmd in subprocess and log output to debug """

    log.debug(f"Running cmd: {cmd}")
    p: Any
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    ) as p:
        for line in p.stdout:
            log.debug(line.strip("\n"))

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)


# pylint: disable=too-many-arguments, too-many-locals
def run_ebma(
    df_calib: pd.DataFrame,
    df_test: pd.DataFrame,
    s_calib_actual: pd.Series,
    tolerance: float = 0.001,
    shrinkage: float = 3,
    const: float = 0.01,
    maxiter: int = 10_000,
) -> Tuple[pd.Series, Dict[str, float]]:
    """ Compute EBMA predictions and weights using wrapped R EBMAforecast

    Args:
        df_calib: Dataframe with constituent models predictions for calibration
        df_test: Dataframe with constituent model
        predictions for test period
        s_calib_actual: Series with actuals for the calibration partition
        tolerance: See R docs
        shrinkage: See R docs
        const: See R docs
        maxiter: See R docs
    Returns:
        s_ebma: Series with ebma predictions
        weights: Dictionary of model weights

    R docs at:
    https://cran.r-project.org/web/packages/EBMAforecast/EBMAforecast.pdf

    Ensure df_calib, df_test and s_calib_actual have multiindex set.

    """

    # Copy data so we don't mess with callers data
    df_calib = df_calib.copy()
    df_test = df_test.copy()
    s_calib_actual = s_calib_actual.copy()
    s_calib_actual.name = "actual"

    # Make sure we're all indexed as expected
    datautils.check_has_multiindex(df_calib)
    datautils.check_has_multiindex(df_test)
    datautils.check_has_multiindex(s_calib_actual)

    if not len(s_calib_actual) == len(df_calib):
        msg = "Number of rows in df_calib and s_calib_actual don't match"
        raise RuntimeError(msg)

    offset = 1e-10
    upper = 1 - offset
    lower = 0 + offset

    # Sort indexes so they're aligned
    # Clip predictions
    df_calib = df_calib.sort_index().clip(lower, upper)
    df_test = df_test.sort_index().clip(lower, upper)
    df_calib_actual = pd.DataFrame(s_calib_actual.sort_index())

    with tempfile.TemporaryDirectory() as tempdir:

        path_csv_calib = os.path.join(tempdir, "calib.csv")
        path_csv_test = os.path.join(tempdir, "test.csv")
        path_csv_actuals = os.path.join(tempdir, "actuals.csv")
        path_csv_ebma = os.path.join(tempdir, "ebma.csv")
        path_csv_weights = os.path.join(tempdir, "weights.csv")
        path_rscript = os.path.join(tempdir, "ebma_script.R")

        values = {
            "PATH_CSV_ACTUALS": path_csv_actuals,
            "PATH_CSV_CALIB": path_csv_calib,
            "PATH_CSV_TEST": path_csv_test,
            "PATH_CSV_EBMA": path_csv_ebma,
            "PATH_CSV_WEIGHTS": path_csv_weights,
            "PARAM_TOLERANCE": tolerance,
            "PARAM_SHRINKAGE": shrinkage,
            "PARAM_CONST": const,
            "PARAM_MAXITER": maxiter,
        }

        template = _read_template()
        rscript = template.substitute(values)

        df_calib.to_csv(path_csv_calib, index=False)
        df_test.to_csv(path_csv_test, index=False)
        df_calib_actual.to_csv(path_csv_actuals, index=False)

        with open(path_rscript, "w") as f:
            f.write(rscript)
        cmd = ["Rscript", path_rscript]
        _run_subproc(cmd)

        df_ebma = pd.read_csv(path_csv_ebma)
        df_weights = pd.read_csv(path_csv_weights)

    df_ebma.index = df_test.index
    s_ebma = df_ebma["x"]
    s_ebma.name = "ebma"

    s_weights = df_weights["x"]
    s_weights.index = df_calib.columns
    weights_dict = s_weights.to_dict()

    return s_ebma, weights_dict
