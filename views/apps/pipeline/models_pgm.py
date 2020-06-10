""" All CM model objects """

# pylint: disable=invalid-name,too-many-lines

from typing import Dict, List
import logging

from sklearn.ensemble import RandomForestClassifier  # type: ignore

from views.apps.model import api
from views.specs.models import pgm
from views.specs.periods import get_periods

log = logging.getLogger(__name__)

rf = RandomForestClassifier(n_jobs=-1, n_estimators=1_000)

# The currently latest model development run id
run_id = "d_2020_04_01"
periods: List[api.Period] = get_periods(run_id=run_id)
steps = [1, 3, 6, 9, 12, 18, 24, 30, 36, 38]


pgm_sb_allthemes = api.Model(
    name="pgm_sb_allthemes",
    col_outcome=pgm["sb_allthemes"]["col_outcome"],
    cols_features=pgm["sb_allthemes"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_sb_pgd_natural = api.Model(
    name="pgm_sb_pgd_natural",
    col_outcome=pgm["sb_pgd_natural"]["col_outcome"],
    cols_features=pgm["sb_pgd_natural"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_sb_pgd_social = api.Model(
    name="pgm_sb_pgd_social",
    col_outcome=pgm["sb_pgd_social"]["col_outcome"],
    cols_features=pgm["sb_pgd_social"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_sb_inst = api.Model(
    name="pgm_sb_inst",
    col_outcome=pgm["sb_inst"]["col_outcome"],
    cols_features=pgm["sb_inst"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_sb_demog = api.Model(
    name="pgm_sb_demog",
    col_outcome=pgm["sb_demog"]["col_outcome"],
    cols_features=pgm["sb_demog"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_sb_oil = api.Model(
    name="pgm_sb_oil",
    col_outcome=pgm["sb_oil"]["col_outcome"],
    cols_features=pgm["sb_oil"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_sb_hist_legacy = api.Model(
    name="pgm_sb_hist_legacy",
    col_outcome=pgm["sb_hist_legacy"]["col_outcome"],
    cols_features=pgm["sb_hist_legacy"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_sb_speisubset_leghist = api.Model(
    name="pgm_sb_speisubset_leghist",
    col_outcome=pgm["sb_speisubset_leghist"]["col_outcome"],
    cols_features=pgm["sb_speisubset_leghist"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)

pgm_ns_allthemes = api.Model(
    name="pgm_ns_allthemes",
    col_outcome=pgm["ns_allthemes"]["col_outcome"],
    cols_features=pgm["ns_allthemes"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_pgd_natural = api.Model(
    name="pgm_ns_pgd_natural",
    col_outcome=pgm["ns_pgd_natural"]["col_outcome"],
    cols_features=pgm["ns_pgd_natural"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_pgd_social = api.Model(
    name="pgm_ns_pgd_social",
    col_outcome=pgm["ns_pgd_social"]["col_outcome"],
    cols_features=pgm["ns_pgd_social"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_inst = api.Model(
    name="pgm_ns_inst",
    col_outcome=pgm["ns_inst"]["col_outcome"],
    cols_features=pgm["ns_inst"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_demog = api.Model(
    name="pgm_ns_demog",
    col_outcome=pgm["ns_demog"]["col_outcome"],
    cols_features=pgm["ns_demog"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_oil = api.Model(
    name="pgm_ns_oil",
    col_outcome=pgm["ns_oil"]["col_outcome"],
    cols_features=pgm["ns_oil"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_hist_legacy = api.Model(
    name="pgm_ns_hist_legacy",
    col_outcome=pgm["ns_hist_legacy"]["col_outcome"],
    cols_features=pgm["ns_hist_legacy"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_speisubset_leghist = api.Model(
    name="pgm_ns_speisubset_leghist",
    col_outcome=pgm["ns_speisubset_leghist"]["col_outcome"],
    cols_features=pgm["ns_speisubset_leghist"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)

pgm_os_allthemes = api.Model(
    name="pgm_os_allthemes",
    col_outcome=pgm["os_allthemes"]["col_outcome"],
    cols_features=pgm["os_allthemes"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_pgd_natural = api.Model(
    name="pgm_os_pgd_natural",
    col_outcome=pgm["os_pgd_natural"]["col_outcome"],
    cols_features=pgm["os_pgd_natural"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_pgd_social = api.Model(
    name="pgm_os_pgd_social",
    col_outcome=pgm["os_pgd_social"]["col_outcome"],
    cols_features=pgm["os_pgd_social"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_inst = api.Model(
    name="pgm_os_inst",
    col_outcome=pgm["os_inst"]["col_outcome"],
    cols_features=pgm["os_inst"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_demog = api.Model(
    name="pgm_os_demog",
    col_outcome=pgm["os_demog"]["col_outcome"],
    cols_features=pgm["os_demog"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_oil = api.Model(
    name="pgm_os_oil",
    col_outcome=pgm["os_oil"]["col_outcome"],
    cols_features=pgm["os_oil"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_hist_legacy = api.Model(
    name="pgm_os_hist_legacy",
    col_outcome=pgm["os_hist_legacy"]["col_outcome"],
    cols_features=pgm["os_hist_legacy"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_speisubset_leghist = api.Model(
    name="pgm_os_speisubset_leghist",
    col_outcome=pgm["os_speisubset_leghist"]["col_outcome"],
    cols_features=pgm["os_speisubset_leghist"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)


pgm_pr_allthemes = api.Model(
    name="pgm_pr_allthemes",
    col_outcome=pgm["pr_allthemes"]["col_outcome"],
    cols_features=pgm["pr_allthemes"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_pgd_natural = api.Model(
    name="pgm_pr_pgd_natural",
    col_outcome=pgm["pr_pgd_natural"]["col_outcome"],
    cols_features=pgm["pr_pgd_natural"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_pgd_social = api.Model(
    name="pgm_pr_pgd_social",
    col_outcome=pgm["pr_pgd_social"]["col_outcome"],
    cols_features=pgm["pr_pgd_social"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_inst = api.Model(
    name="pgm_pr_inst",
    col_outcome=pgm["pr_inst"]["col_outcome"],
    cols_features=pgm["pr_inst"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_demog = api.Model(
    name="pgm_pr_demog",
    col_outcome=pgm["pr_demog"]["col_outcome"],
    cols_features=pgm["pr_demog"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_oil = api.Model(
    name="pgm_pr_oil",
    col_outcome=pgm["pr_oil"]["col_outcome"],
    cols_features=pgm["pr_oil"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_hist_legacy = api.Model(
    name="pgm_pr_hist_legacy",
    col_outcome=pgm["pr_hist_legacy"]["col_outcome"],
    cols_features=pgm["pr_hist_legacy"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)


pgm_sb_sptime = api.Model(
    name="pgm_sb_sptime",
    col_outcome=pgm["sb_sptime"]["col_outcome"],
    cols_features=pgm["sb_sptime"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_sptime = api.Model(
    name="pgm_ns_sptime",
    col_outcome=pgm["ns_sptime"]["col_outcome"],
    cols_features=pgm["ns_sptime"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_sptime = api.Model(
    name="pgm_os_sptime",
    col_outcome=pgm["os_sptime"]["col_outcome"],
    cols_features=pgm["os_sptime"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_sptime = api.Model(
    name="pgm_pr_sptime",
    col_outcome=pgm["pr_sptime"]["col_outcome"],
    cols_features=pgm["pr_sptime"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)

pgm_sb_spei_full = api.Model(
    name="pgm_sb_spei_full",
    col_outcome=pgm["sb_spei_full"]["col_outcome"],
    cols_features=pgm["sb_spei_full"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_spei_full = api.Model(
    name="pgm_ns_spei_full",
    col_outcome=pgm["ns_spei_full"]["col_outcome"],
    cols_features=pgm["ns_spei_full"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_spei_full = api.Model(
    name="pgm_os_spei_full",
    col_outcome=pgm["os_spei_full"]["col_outcome"],
    cols_features=pgm["os_spei_full"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_spei_full = api.Model(
    name="pgm_pr_spei_full",
    col_outcome=pgm["pr_spei_full"]["col_outcome"],
    cols_features=pgm["pr_spei_full"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)

pgm_sb_sptime_and_all_themes = api.Model(
    name="pgm_sb_sptime_and_all_themes",
    col_outcome=pgm["sb_sptime_and_all_themes"]["col_outcome"],
    cols_features=pgm["sb_sptime_and_all_themes"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_sptime_and_all_themes = api.Model(
    name="pgm_ns_sptime_and_all_themes",
    col_outcome=pgm["ns_sptime_and_all_themes"]["col_outcome"],
    cols_features=pgm["ns_sptime_and_all_themes"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_sptime_and_all_themes = api.Model(
    name="pgm_os_sptime_and_all_themes",
    col_outcome=pgm["os_sptime_and_all_themes"]["col_outcome"],
    cols_features=pgm["os_sptime_and_all_themes"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_pr_sptime_and_all_themes = api.Model(
    name="pgm_pr_sptime_and_all_themes",
    col_outcome=pgm["pr_sptime_and_all_themes"]["col_outcome"],
    cols_features=pgm["pr_sptime_and_all_themes"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)

pgm_sb_all = api.Model(
    name="pgm_sb_all",
    col_outcome=pgm["sb_all"]["col_outcome"],
    cols_features=pgm["sb_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_ns_all = api.Model(
    name="pgm_ns_all",
    col_outcome=pgm["ns_all"]["col_outcome"],
    cols_features=pgm["ns_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)
pgm_os_all = api.Model(
    name="pgm_os_all",
    col_outcome=pgm["os_all"]["col_outcome"],
    cols_features=pgm["os_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
)

pgm_sb_onset24_1_all = api.Model(
    name="pgm_sb_onset24_1_all",
    col_outcome=pgm["sb_onset24_1_all"]["col_outcome"],
    cols_features=pgm["sb_onset24_1_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)
pgm_sb_onset24_5_all = api.Model(
    name="pgm_sb_onset24_5_all",
    col_outcome=pgm["sb_onset24_5_all"]["col_outcome"],
    cols_features=pgm["sb_onset24_5_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)
pgm_sb_onset24_100_all = api.Model(
    name="pgm_sb_onset24_100_all",
    col_outcome=pgm["sb_onset24_100_all"]["col_outcome"],
    cols_features=pgm["sb_onset24_100_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)

pgm_ns_onset24_1_all = api.Model(
    name="pgm_ns_onset24_1_all",
    col_outcome=pgm["ns_onset24_1_all"]["col_outcome"],
    cols_features=pgm["ns_onset24_1_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)
pgm_ns_onset24_5_all = api.Model(
    name="pgm_ns_onset24_5_all",
    col_outcome=pgm["ns_onset24_5_all"]["col_outcome"],
    cols_features=pgm["ns_onset24_5_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)
pgm_ns_onset24_100_all = api.Model(
    name="pgm_ns_onset24_100_all",
    col_outcome=pgm["ns_onset24_100_all"]["col_outcome"],
    cols_features=pgm["ns_onset24_100_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)

pgm_os_onset24_1_all = api.Model(
    name="pgm_os_onset24_1_all",
    col_outcome=pgm["os_onset24_1_all"]["col_outcome"],
    cols_features=pgm["os_onset24_1_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)
pgm_os_onset24_5_all = api.Model(
    name="pgm_os_onset24_5_all",
    col_outcome=pgm["os_onset24_5_all"]["col_outcome"],
    cols_features=pgm["os_onset24_5_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)
pgm_os_onset24_100_all = api.Model(
    name="pgm_os_onset24_100_all",
    col_outcome=pgm["os_onset24_100_all"]["col_outcome"],
    cols_features=pgm["os_onset24_100_all"]["cols_features"],
    steps=steps,
    outcome_type="prob",
    estimator=rf,
    periods=periods,
    tags=["train_africa"],
    onset_outcome=True,
    onset_window=24,
)

all_pgm_models: List[api.Model] = [
    pgm_sb_allthemes,
    pgm_sb_pgd_natural,
    pgm_sb_pgd_social,
    pgm_sb_inst,
    pgm_sb_demog,
    pgm_sb_oil,
    pgm_sb_hist_legacy,
    pgm_sb_speisubset_leghist,
    pgm_ns_allthemes,
    pgm_ns_pgd_natural,
    pgm_ns_pgd_social,
    pgm_ns_inst,
    pgm_ns_demog,
    pgm_ns_oil,
    pgm_ns_hist_legacy,
    pgm_ns_speisubset_leghist,
    pgm_os_allthemes,
    pgm_os_pgd_natural,
    pgm_os_pgd_social,
    pgm_os_inst,
    pgm_os_demog,
    pgm_os_oil,
    pgm_os_hist_legacy,
    pgm_os_speisubset_leghist,
    pgm_pr_allthemes,
    pgm_pr_pgd_natural,
    pgm_pr_pgd_social,
    pgm_pr_inst,
    pgm_pr_demog,
    pgm_pr_oil,
    pgm_pr_hist_legacy,
    pgm_sb_sptime,
    pgm_ns_sptime,
    pgm_os_sptime,
    pgm_pr_sptime,
    pgm_sb_spei_full,
    pgm_ns_spei_full,
    pgm_os_spei_full,
    pgm_pr_spei_full,
    pgm_sb_sptime_and_all_themes,
    pgm_ns_sptime_and_all_themes,
    pgm_os_sptime_and_all_themes,
    pgm_pr_sptime_and_all_themes,
    pgm_sb_all,
    pgm_ns_all,
    pgm_os_all,
    pgm_sb_onset24_1_all,
    pgm_sb_onset24_5_all,
    pgm_sb_onset24_100_all,
    pgm_ns_onset24_1_all,
    pgm_ns_onset24_5_all,
    pgm_ns_onset24_100_all,
    pgm_os_onset24_1_all,
    pgm_os_onset24_5_all,
    pgm_os_onset24_100_all,
]

all_pgm_models_by_name: Dict[str, api.Model] = dict()
for model in all_pgm_models:
    all_pgm_models_by_name[model.name] = model
