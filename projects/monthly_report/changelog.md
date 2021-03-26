# r_2021_03_01

**General changes (to merge into master)**:
* Reduced cores per job by one via `mem_per_job` in `monthly.py` to avoid crashing multiprocessing pool.
* Simplified argparse of `train_model.py` and `train_slurm.py` to allow listing of steps and models like `--model model_a model_b` rather than `--model model_a --model model_b`.
* `get_files_latest_fetch` in `common.py` fetched first rather than last item in sorted list. This has applied to ICGCW and REIGN. Now fetches the correct latest version of the data.
* Allowed passing 0 to `tlag` in transforms lib for tlag_0 variables.
* Adding current violent history input to the model specs. Zero time-lagged to avoid step-shifting our outcome in model api. See specific changes to column sets below.

**Changes at cm**:
* Added to `cfshort`:
```
    - tlags_0_ged_dummy_sb_ns_os
    - ged_best_sb_ns_os
    - greq_5_ged_best_sb_ns_os
    - tlags_0_greq_5_ged_best_sb_ns_os
    - tlags_0_greq_25_ged_best_sb_ns_os
    - greq_100_ged_best_sb_ns_os
    - tlags_0_greq_100_ged_best_sb_ns_os
```
* Added to `cflong`:
```
    - tlags_0_ged_dummy_sb_ns_os
    - ged_best_sb_ns_os
    - greq_5_ged_best_sb_ns_os
    - tlags_0_greq_5_ged_best_sb_ns_os
    - tlags_0_greq_25_ged_best_sb_ns_os
    - greq_100_ged_best_sb_ns_os
    - tlags_0_greq_100_ged_best_sb_ns_os
```

**Changes at pgm**:

* Added to `legacy_hist_common`:
```
    - tlags_0_ged_dummy_sb_ns_os
    - ged_best_sb_ns_os  # TODO?
    - tlags_0_greq_5_ged_best_sb_ns_os
    - tlags_0_greq_25_ged_best_sb_ns_os
    - tlags_0_greq_100_ged_best_sb_ns_os
    - acled_protest
```
* Added colset `acled_protest`:
```
    - acled_dummy_pr
    - tlag_0_acled_dummy_pr
    - acled_count_pr
```


**Retrained models at cm**:
* cm_sb_cfshort
* cm_sb_cflong
* cm_sb_acled_violence
* cm_sb_acled_protest 
* cm_sbonset24_25_all
* cm_sb_all_global


**Retrained models at pgm**:
* pgm_sb_hist_legacy
* pgm_sb_allthemes
* pgm_sb_onset24_100_all
* pgm_sb_onset24_1_all
* pgm_sb_all_gxgb