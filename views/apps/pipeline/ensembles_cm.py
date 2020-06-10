""" All CM Ensemble objects


The following models are included in the JPR 2020 CM
ensemble:
cm_sb_cflong
cm_sb_acled_violence
cm_ns_neibhist
cm_sb_cdummies
cm_sb_acled_protest
cm_sb_reign_coups
cm_sb_icgcw
cm_sb_reign_drought
cm_sb_reign_global
cm_sb_vdem_global
cm_sb_demog
cm_sb_wdi_global
cm_sb_all_global
cm_sbonset24_25_all

and are all included in the prelim ensembles below.
@TODO: Not ready in this repo yet are and are to be added later:

ds_25
ds_dummy

"""

# pylint: disable=invalid-name


from typing import Dict, List
from views.apps.model.api import Ensemble, Model, Period
from views.specs.periods import get_periods
from . import models_cm


# The currently latest model development run id
run_id = "d_2020_04_01"
periods: List[Period] = get_periods(run_id=run_id)

models_cm_sb_prelim: List[Model] = [
    models_cm.cm_sb_cflong,
    models_cm.cm_sb_acled_violence,
    models_cm.cm_sb_neibhist,
    models_cm.cm_sb_cdummies,
    models_cm.cm_sb_acled_protest,
    models_cm.cm_sb_reign_coups,
    models_cm.cm_sb_icgcw,
    models_cm.cm_sb_reign_drought,
    models_cm.cm_sb_reign_global,
    models_cm.cm_sb_vdem_global,
    models_cm.cm_sb_demog,
    models_cm.cm_sb_wdi_global,
    models_cm.cm_sb_all_global,
    models_cm.cm_sbonset24_25_all,
]

models_cm_ns_prelim: List[Model] = [
    models_cm.cm_ns_cflong,
    models_cm.cm_ns_acled_violence,
    models_cm.cm_ns_neibhist,
    models_cm.cm_ns_cdummies,
    models_cm.cm_ns_acled_protest,
    models_cm.cm_ns_reign_coups,
    models_cm.cm_ns_icgcw,
    models_cm.cm_ns_reign_drought,
    models_cm.cm_ns_reign_global,
    models_cm.cm_ns_vdem_global,
    models_cm.cm_ns_demog,
    models_cm.cm_ns_wdi_global,
    models_cm.cm_ns_all_global,
    models_cm.cm_nsonset24_25_all,
]

models_cm_os_prelim: List[Model] = [
    models_cm.cm_os_cflong,
    models_cm.cm_os_acled_violence,
    models_cm.cm_os_neibhist,
    models_cm.cm_os_cdummies,
    models_cm.cm_os_acled_protest,
    models_cm.cm_os_reign_coups,
    models_cm.cm_os_icgcw,
    models_cm.cm_os_reign_drought,
    models_cm.cm_os_reign_global,
    models_cm.cm_os_vdem_global,
    models_cm.cm_os_demog,
    models_cm.cm_os_wdi_global,
    models_cm.cm_os_all_global,
    models_cm.cm_osonset24_25_all,
]

cm_sb_prelim = Ensemble(
    name="cm_sb_prelim",
    models=models_cm_sb_prelim,
    method="ebma",
    outcome_type="prob",
    col_outcome="greq_25_ged_best_sb",
    periods=periods,
    delta_outcome=False,
)

cm_ns_prelim = Ensemble(
    name="cm_ns_prelim",
    models=models_cm_ns_prelim,
    method="ebma",
    outcome_type="prob",
    col_outcome="greq_25_ged_best_ns",
    periods=periods,
    delta_outcome=False,
)
cm_os_prelim = Ensemble(
    name="cm_os_prelim",
    models=models_cm_os_prelim,
    method="ebma",
    outcome_type="prob",
    col_outcome="greq_25_ged_best_os",
    periods=periods,
    delta_outcome=False,
)

all_cm_ensembles: List[Ensemble] = [
    cm_sb_prelim,
    cm_ns_prelim,
    cm_os_prelim,
]
all_cm_ensembles_by_name: Dict[str, Ensemble] = dict()
for ensemble in all_cm_ensembles:
    all_cm_ensembles_by_name[ensemble.name] = ensemble
