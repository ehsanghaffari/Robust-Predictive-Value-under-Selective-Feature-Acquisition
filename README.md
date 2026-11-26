# Robust Predictive Value under Selective Feature Acquisition

Code for the article *Robust Predictive Value under Selective Feature Acquisition: Exact Identification Boundaries in Crash Alcohol Testing*.

The analysis asks whether a selectively measured feature (here, driver blood alcohol in the U.S. Crash Report Sampling System, CRSS, 2016-2024) is worth keeping once the unmeasured values are treated as partially identified. It computes the robust feature value (RFV), the worst-case Brier risk reduction over all completions of the unmeasured values allowed by a monotone-acquisition (MAS) assumption, together with its zero-value boundary, the external-information frontier, design-based uncertainty and conventional machine-learning (ML) comparisons.

Notation used in the code: `A` alcohol test given, `R` test result resolves the alcohol state, `Z` latent alcohol state (BAC > 0, or BAC >= 0.08), `W` crash covariates, `Y` fatal injury.

## Repository layout

```
1_identification_gate/
    crss_identification_gate.py        CRSS extraction, coding audit and per-year identified quantities
2_bounds_and_frontier/
    bounds_frontier_oracle.py          MAS bounds, boundary distances, risk intervals, frontier, OracleGain
    value_identity_and_skill.py        value-identity check, robust skill, robust value beyond the boundary
    fig9_robust_skill.py               Fig. 9
3_ml_premise/
    ml_premise_temporal_annual.py      pooled, forward-temporal and annual ML experiments (LR, XGBoost, CatBoost)
4_design_based_and_attribution/
    design_based_and_attribution.py    PSU jackknife, effective sample sizes, cohort and testing-status checks,
                                       block-order attribution, conditional RFV (Online Resource 1, Tables S1-S15)
    bac08_result_block.py              ML result-block models for BAC >= 0.08 (Tables S9b-S13b, S16)
    conditional_rfv_cell_detail.py     cell-level counts behind the conditional RFV values (Sect. 6.11)
    make_online_resource.py            Online Resource 1 as a Word document from existing outputs (no re-run)
5_supplementary_audits/
    crss_drug_coding_audit.py          CRSS drug-variable coding audit (not used in the article)
    penndot_compatibility_audit.py     PennDOT schema audit (not used in the article)
    penndot_admin_flag_supplement.py   PennDOT administrative-flag diagnostic (not used in the article)
docs/
    decision_rule_freeze.md            RFV decision rule, frozen before pooled results were viewed
    data_and_code_availability.md      the article's data and code availability statements
results/split_ids/                     released case-level train/val/test splits of the pooled ML experiment
data/                                  input data (not tracked; see below)
outputs/                               generated results, one sub-folder per script folder (not tracked)
LICENSE                                MIT License (code only)
```

Every script resolves its default paths relative to the repository root, so it can be run from any working directory. Every default can be overridden on the command line (`--help` lists the options).

## Data

**CRSS.** The scripts read the NHTSA files `CRSS2016CSV.zip` ... `CRSS2024CSV.zip` from `data/crss_zips/`. The identification gate and the ML script download any missing year from `https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip`; `design_based_and_attribution.py` does so when run with `--download`. Raw CRSS files are not redistributed here. `design_based_and_attribution.py` records the SHA-256 checksum of every input file in `run_manifest.json`.

**PennDOT** (supplementary audits only). The Pennsylvania crash data were obtained through the [PennDOT Crash Information Tool](https://crashinfo.penndot.pa.gov/). PennDOT Statewide crash files are not redistributed. To run the scripts in `5_supplementary_audits/`, place them as `data/penndot/Statewide_<year>/<TABLE>_<year>.csv`.

## Installation

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the core numerical and machine-learning packages at the versions used for the reported results; it is not a complete lockfile.

## Reproducing the results

Run from the repository root, in this order.

```bash
# 1. Identification gate (downloads CRSS 2016-2024 on first use)
python 1_identification_gate/crss_identification_gate.py --all-years

# 2. Bounds, boundary distances, frontier and OracleGain (seconds; no ML)
python 2_bounds_and_frontier/bounds_frontier_oracle.py
python 2_bounds_and_frontier/value_identity_and_skill.py
python 2_bounds_and_frontier/fig9_robust_skill.py

# 3. ML experiments (smoke test first, then the full five-seed run)
python 3_ml_premise/ml_premise_temporal_annual.py --quick
python 3_ml_premise/ml_premise_temporal_annual.py

# 4. Design-based uncertainty and attribution (after steps 1 and 3)
python 4_design_based_and_attribution/design_based_and_attribution.py \
    --attribution --dagjk-groups 15 --dagjk-families lr,xgb

# 5. ML result-block models for BAC >= 0.08 (after step 4; smoke test first)
python 4_design_based_and_attribution/bac08_result_block.py --quick
python 4_design_based_and_attribution/bac08_result_block.py
python 4_design_based_and_attribution/make_online_resource.py
```

Notes on step 4:

* Without `--ml` or `--attribution` the script runs only the identification analyses (jackknife, effective sample sizes, cohort harmonization, conditional RFV; Tables S1-S8), which takes minutes.
* `--attribution` fits the models reported in the article (tables S9-S12); `--dagjk-groups 15` adds the refitting delete-a-group jackknife (table S13). `--ml --three-block` instead fits every model, adding M2 with three-level status and the three-block Shapley decomposition.
* ML fits are checkpointed in `ml_fits_<cohort>.csv`, so an interrupted run resumes where it stopped. To shorten a run, use `--seeds`, `--families` or `--tiers`.
* The ML analyses import `3_ml_premise/ml_premise_temporal_annual.py` and reuse its data preparation, covariates, case split, models and metrics, so M0, M1 and M2 on the ML cohort are the Table 7 fits. They need `--zipdir` (default) or `--raw-dir`.
* Two reproduction checks run automatically when their reference files exist: the identification quantities against `outputs/1_identification_gate/crss_identification_gate.csv` (`gate_reproduction_check.csv`), and M0/M1/M2 against `outputs/3_ml_premise/all_model_metrics.csv` (`ml_reproduction_check.csv`). The log reports `MATCH` when they agree (Tables S14 and S15). Run step 4 after steps 1 and 3 so both checks can run.
* Step 5 repeats, for BAC >= 0.08, every reported ML result that uses result resolution or the alcohol value (M_R, M_AR, M2 in the pooled block-order analysis, M2 in the delete-a-group jackknife and in the annual analysis), with the same case splits, seeds and hyperparameters. The temporal holdout analysis reports only M1 - M0, so its M2 fits are not repeated. M0 and M1 do not use alcohol results, so their BAC > 0 results are reused rather than refitted (M0 is refitted inside the delete-a-group replicates, whose levels were not stored). Before fitting, it checks that it rebuilds the BAC > 0 features and fits exactly (`bac08_alignment_check.csv`, `dagjk_grouping_check_bac08.csv`). It is checkpointed and resumes after an interruption.
* At the end, the script writes Online Resource 1 (`online_resource_1.docx`, every table captioned) and `supplementary_tables.xlsx` from its CSV outputs. To rebuild them from an existing outputs folder without re-running anything: `python 4_design_based_and_attribution/make_online_resource.py`.

## Outputs and where they appear in the article

| Script | Output (under `outputs/`) | Article |
|---|---|---|
| `crss_identification_gate.py` | `1_identification_gate/crss_annual_acquisition_support.csv`, `crss_identification_gate.csv` (column `q_test_BAC_gt0_weighted` is the article's q_s) | Table 2 |
| | `1_identification_gate/crss_pooled_fatal_stability.csv` | Table 3 |
| | `1_identification_gate/<year>_ALC_STATUS_by_ALC_RES_raw_crosstab.csv` | coding audit (Sect. 4.2) |
| `bounds_frontier_oracle.py` | `2_bounds_and_frontier/all_year_risk_intervals_and_RFV.csv` | Table 5 |
| | `2_bounds_and_frontier/all_year_boundary_distances.csv`, `Fig1_boundary_distances.png` | Table 6, Fig. 1 |
| | `2_bounds_and_frontier/Fig2_testing_rates.png`, `Fig3_survivor_positivity.png` | Figs. 2 and 3 |
| | `2_bounds_and_frontier/year_parameters_gt0.csv` | annual inputs for the two scripts below |
| | `2_bounds_and_frontier/joint_assumption_frontier_table.csv`, `Fig8_joint_assumption_frontier.png` | Table 10, Fig. 8 |
| | `2_bounds_and_frontier/Fig4_oraclegain_2024.png` | Fig. 4 |
| | `2_bounds_and_frontier/full_accounting_audit.csv` | accounting audit (Sect. 4.5) |
| `value_identity_and_skill.py` | `2_bounds_and_frontier/value_identity_and_skill/oraclegain_sensitivity_2024.csv` | Table 4 |
| | `2_bounds_and_frontier/value_identity_and_skill/rfv_beyond_boundary.csv` | Table 11 |
| | `2_bounds_and_frontier/value_identity_and_skill/check_value_identity.csv` | numerical check of the value identity |
| `fig9_robust_skill.py` | `2_bounds_and_frontier/fig9_robust_skill_2024.png` | Fig. 9 |
| `ml_premise_temporal_annual.py` | `3_ml_premise/all_model_metrics.csv`, `paired_arm_delta_seed_summary.csv` | Tables 7 and 8; data for Figs. 5 and 6 |
| | `3_ml_premise/annual_apparent_value_vs_qtest_summary.csv`, `annual_qtest_spearman_descriptive.csv` (column `q_test` is the article's q_ML) | Table 9; data for Fig. 7 |
| | `3_ml_premise/split_ids/split_ids_<experiment>.csv.gz` | case-level train/val/test split for each seed (the pooled one is released in `results/split_ids/`) |
| `design_based_and_attribution.py` | `4_design_based_and_attribution/attribution_summary.csv` | Sect. 6.7 (block order, testing status, cohort) |
| `bac08_result_block.py` | `4_design_based_and_attribution/attribution_summary_bac08.csv`, `annual_spearman_bac08.csv` | Sect. 6.7 and 6.9 (BAC >= 0.08) |
| `design_based_and_attribution.py`, `make_online_resource.py` | `4_design_based_and_attribution/online_resource_1.docx`, `supplementary_tables.xlsx` | Online Resource 1, Tables S1-S16 |

`value_identity_and_skill.py` and `fig9_robust_skill.py` read the unrounded annual quantities in `outputs/2_bounds_and_frontier/year_parameters_gt0.csv`, written by `bounds_frontier_oracle.py`; run them after it. Without that file they fall back to the rounded values printed in the article (the log and `report.json` say which inputs were used).

The ML results in `design_based_and_attribution.py` (block order, testing status, cohort, jackknife) are computed on the same fits, splits and covariates as Table 7.

## Methods notes

* **Coding.** `ALC_STATUS = 2` is a test given; `0` (not given) and `1` (refused, 2016) are known not tested; every other value is unknown and is never counted as untested. `ALC_RES` 0-940 is a numeric BAC (x1000); `998` (positive, no value) resolves BAC > 0 but not BAC >= 0.08; other codes are unresolved. Drivers who died before the crash (`INJ_SEV = 6`) are excluded; the ML cohort keeps `INJ_SEV` 0-4.
* **Design-based variance** follows NHTSA DOT HS 812 688: strata `PSUSTRAT`, clusters `PSU_VAR`, weight `WEIGHT`, delete-one-PSU jackknife with the with-replacement approximation; domain estimates use the full annual sample. Single-PSU strata contribute no variance and are counted. Pooled ML test sets use `PSUSTRAT` across years as strata, which is conservative when PSUs recur across years.
* **Decision rule.** The RFV decision rule (OracleGain floor and retention thresholds) was frozen after the 2023 sanity check and before pooled 2016-2024 results were viewed; see `docs/decision_rule_freeze.md`. The gate writes the same record to `rfv_decision_rule_freeze.txt`.
* **Seeds.** 42, 1337, 2026, 31415, 271828.

## Supplementary audits

The scripts in `5_supplementary_audits/` are not used in the article. The PennDOT audit shows that PennDOT Statewide files carry final administrative alcohol flags but no CRSS-style test acquisition or result fields, so they cannot replace CRSS for the A/R/Z analysis. The drug audit inventories CRSS drug-testing fields before any drug replication; it does not build a drug RFV.

## Data and code availability

* **Source data.** CRSS 2016-2024 files are publicly available from [NHTSA](https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system) and are not mirrored here. The PennDOT data used by the supplementary audits are also excluded.
* **Code.** Everything needed to reproduce the article's tables and figure data from the source data. Figs. 1 to 4, 8 and 9 are drawn as published, at 600 dpi; Figs. 5 to 7 are plotted from the CSV outputs listed above, and their plotting code is not included.
* **Results.** Generated results are not committed, except the case-level split identifiers in `results/split_ids/`. The article reports its results in the main text and Online Resource 1 (identification and machine-learning results for both BAC definitions). The released pipeline reproduces the identification gate output and the Table 7 fits exactly (Online Resource 1, Tables S14 and S15).
* **License.** The code is released under the MIT License (`LICENSE`), which does not cover the NHTSA or PennDOT source data.

The article's data and code availability statements are in `docs/data_and_code_availability.md`.

## Citation

If you use this code, please cite the accompanying article (full reference to be added on publication).
