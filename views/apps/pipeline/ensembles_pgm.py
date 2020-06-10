""" All PGM Ensemble objects

The following models are included in the JPR 2020 PGM
ensemble:
allthemes
hist_legacy
onset24_100_all
onset24_1_all
pgd_natural
pgd_social
sptime

These 4 are not included yet but will be when implemented in this repo
ds_25
ds_dummy
xgb
crosslevel

]
"""

# pylint: disable=invalid-name

from typing import Dict, List
from views.apps.model.api import Ensemble, Model, Period
from views.specs.periods import get_periods
from . import models_pgm


# The currently latest model development run id
run_id = "d_2020_04_01"
periods: List[Period] = get_periods(run_id=run_id)


models_pgm_sb_prelim: List[Model] = [
    models_pgm.pgm_sb_hist_legacy,
    models_pgm.pgm_sb_allthemes,
    models_pgm.pgm_sb_onset24_100_all,
    models_pgm.pgm_sb_onset24_1_all,
    models_pgm.pgm_sb_pgd_natural,
    models_pgm.pgm_sb_pgd_social,
    models_pgm.pgm_sb_sptime,
]

models_pgm_ns_prelim: List[Model] = [
    models_pgm.pgm_ns_hist_legacy,
    models_pgm.pgm_ns_allthemes,
    models_pgm.pgm_ns_onset24_100_all,
    models_pgm.pgm_ns_onset24_1_all,
    models_pgm.pgm_ns_pgd_natural,
    models_pgm.pgm_ns_pgd_social,
    models_pgm.pgm_ns_sptime,
]

models_pgm_os_prelim: List[Model] = [
    models_pgm.pgm_os_hist_legacy,
    models_pgm.pgm_os_allthemes,
    models_pgm.pgm_os_onset24_100_all,
    models_pgm.pgm_os_onset24_1_all,
    models_pgm.pgm_os_pgd_natural,
    models_pgm.pgm_os_pgd_social,
    models_pgm.pgm_os_sptime,
]

pgm_sb_prelim = Ensemble(
    name="pgm_sb_prelim",
    models=models_pgm_sb_prelim,
    method="average",
    outcome_type="prob",
    col_outcome="ged_dummy_sb",
    periods=periods,
)

pgm_ns_prelim = Ensemble(
    name="pgm_ns_prelim",
    models=models_pgm_ns_prelim,
    method="average",
    outcome_type="prob",
    col_outcome="ged_dummy_ns",
    periods=periods,
)

pgm_os_prelim = Ensemble(
    name="pgm_os_prelim",
    models=models_pgm_os_prelim,
    method="average",
    outcome_type="prob",
    col_outcome="ged_dummy_os",
    periods=periods,
)

all_pgm_ensembles: List[Ensemble] = [
    pgm_sb_prelim,
    pgm_ns_prelim,
    pgm_os_prelim,
]

all_pgm_ensembles_by_name: Dict[str, Ensemble] = dict()
for ensemble in all_pgm_ensembles:
    all_pgm_ensembles_by_name[ensemble.name] = ensemble
