""" Amelia python-R wrapper """
from typing import List
import logging
import multiprocessing as mp
import os
import string
import subprocess
import tempfile

import pandas as pd  # type: ignore

from views.utils import data
from views.utils.log import logtime

log = logging.getLogger(__name__)


def run_subproc(cmd):
    """ Run cmd in subprocess and log output to debug """

    log.info(f"Running cmd: {cmd}")
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


# pylint: disable=too-many-locals
@logtime
def impute_amelia(df: pd.DataFrame, n_imp: int) -> List[pd.DataFrame]:
    """ Wrapper for calling Amelia in an R subprocess

    Args:
        df: Dataframe with MultiIndex set
        n_imp: Number of imputations to perform
    Return:
        dfs: List of imputed dataframes
    """

    def read_template():
        this_dir = os.path.dirname(os.path.abspath(__file__))
        path_template = os.path.join(this_dir, "amelia_template.R")
        with open(path_template, "r") as f:
            template_str = f.read()

        template = string.Template(template_str)

        return template

    log.info("Started impute_amelia()")

    data.check_has_multiindex(df)
    timevar, groupvar = df.index.names

    log.debug(f"n_imp: {n_imp}")
    log.debug(f"timevar: {timevar}")
    log.debug(f"groupvar: {groupvar}")
    log.debug(f"df shape: {df.shape}")
    log.debug(f"Share missing: {df.isnull().mean().mean()}")

    with tempfile.TemporaryDirectory() as tempdir:

        path_csv_in = os.path.join(tempdir, "input.csv")
        path_rscript = os.path.join(tempdir, "impute_script.R")
        path_out_stem = os.path.join(tempdir, "imputed_")

        values = {
            "PATH_CSV_INPUT": path_csv_in,
            "PATH_CSV_OUTPUT_STEM": path_out_stem,
            "TIMEVAR": timevar,
            "GROUPVAR": groupvar,
            "N_IMP": n_imp,
            "N_CPUS": mp.cpu_count(),
        }

        template = read_template()
        rscript = template.substitute(values)

        df.to_csv(path_csv_in, index=True)
        log.info(f"Wrote {path_csv_in}")

        with open(path_rscript, "w") as f:
            f.write(rscript)
        log.info(f"Wrote {path_rscript}")
        log.debug(rscript)

        cmd = ["Rscript", path_rscript]
        run_subproc(cmd)

        dfs = []
        for i in range(n_imp):
            path_imputed = f"{path_out_stem}{i+1}.csv"
            df_imp = pd.read_csv(path_imputed)
            df_imp = df_imp.drop(columns=["Unnamed: 0"])
            df_imp = df_imp.set_index([timevar, groupvar])
            dfs.append(df_imp)
            log.info(f"Read {path_imputed}")

    log.info("Finished impute_amelia()")
    return dfs
