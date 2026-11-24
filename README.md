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
                                       block-order attribution, conditional RFV (supplementary tables S0-S13)
5_supplementary_audits/
    crss_drug_coding_audit.py          CRSS drug-variable coding audit (not used in the article)
    penndot_compatibility_audit.py     PennDOT schema audit (not used in the article)
    penndot_admin_flag_supplement.py   PennDOT administrative-flag diagnostic (not used in the article)
docs/
    data_and_code_availability.md       Manuscript-ready availability statements and precise code coverage
    decision_rule_freeze.md            RFV decision rule, frozen before pooled results were viewed
data/                                  input data (not tracked; see below)
outputs/                               results, one sub-folder per script folder (not tracked)
```

Every script resolves its default paths relative to the repository root, so it can be run from any working directory. Every default can be overridden on the command line (`--help` lists the options).

## Data

**CRSS.** The scripts read the NHTSA files `CRSS2016CSV.zip` ... `CRSS2024CSV.zip` from `data/crss_zips/`. The identification gate and the ML script download any missing year from `https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip`; `design_based_and_attribution.py` does so when run with `--download`. Raw CRSS files are not redistributed here. `design_based_and_attribution.py` records the SHA-256 checksum of every input file in `run_manifest.json`.

**PennDOT** (supplementary audits only). The Pennsylvania crash data were provided through the [PennDOT Crash Information Tool](https://crashinfo.penndot.pa.gov/), which links to public crash databases. PennDOT Statewide crash files are not redistributed. To run the scripts in `5_supplementary_audits/`, place them as `data/penndot/Statewide_<year>/<TABLE>_<year>.csv`.

## Installation

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` specifies the supplied package requirements; it is not a complete lockfile for reproducing an identical software environment.

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

# 4. Design-based uncertainty and attribution
python 4_design_based_and_attribution/design_based_and_attribution.py \
    --ml --three-block --dagjk-groups 15 --dagjk-families lr,xgb
```

Notes on step 4:

* Without `--ml` the script runs only the identification analyses (jackknife, effective sample sizes, cohort harmonization, conditional RFV; tables S0-S8), which takes minutes.
* `--attribution` runs only the models needed for the testing-status, cohort and block-order results (tables S9-S12).
* ML fits are checkpointed in `ml_checkpoint_*.csv`, so an interrupted run resumes where it stopped. To shorten a run, use `--seeds`, `--families` or `--tiers`.
* When `outputs/1_identification_gate/crss_identification_gate.csv` exists, the script compares its identification quantities with the gate and writes `gate_reproduction_check.csv`; the log reports `MATCH` when they agree.

## Outputs and where they appear in the article

| Script | Output (under `outputs/`) | Article |
|---|---|---|
| `crss_identification_gate.py` | `1_identification_gate/crss_annual_acquisition_support.csv`, `crss_identification_gate.csv` | Table 2; data for Figs. 2 and 3 |
| | `1_identification_gate/crss_pooled_fatal_stability.csv` | Table 3 |
| | `1_identification_gate/<year>_ALC_STATUS_by_ALC_RES_raw_crosstab.csv` | coding audit (Sect. 4.2) |
| `bounds_frontier_oracle.py` | `2_bounds_and_frontier/all_year_risk_intervals_and_RFV.csv` | Table 5 |
| | `2_bounds_and_frontier/all_year_boundary_distances.csv`, `Figure_boundary_distances.png` | Table 6, Fig. 1 |
| | `2_bounds_and_frontier/joint_assumption_frontier_table.csv`, `Figure_joint_assumption_frontier.png` | Table 10, Fig. 8 |
| | `2_bounds_and_frontier/OracleGain_2024_balanced.png` | Fig. 4 |
| | `2_bounds_and_frontier/full_accounting_audit.csv` | accounting audit (Sect. 4.5) |
| `value_identity_and_skill.py` | `2_bounds_and_frontier/value_identity_and_skill/oraclegain_sensitivity_2024.csv` | Table 4 |
| | `2_bounds_and_frontier/value_identity_and_skill/rfv_beyond_boundary.csv` | Table 11 |
| | `2_bounds_and_frontier/value_identity_and_skill/check_value_identity.csv` | numerical check of the value identity |
| `fig9_robust_skill.py` | `2_bounds_and_frontier/fig9_robust_skill_2024.png` | Fig. 9 |
| `ml_premise_temporal_annual.py` | `3_ml_premise/all_model_metrics.csv`, `paired_arm_delta_seed_summary.csv` | Tables 7 and 8; data for Figs. 5 and 6 |
| | `3_ml_premise/annual_apparent_value_vs_qtest_summary.csv`, `annual_qtest_spearman_descriptive.csv` | Table 9; data for Fig. 7 |
| | `3_ml_premise/split_ids/split_ids_<experiment>.csv.gz` | case-level train/val/test split for each seed |
| `design_based_and_attribution.py` | `4_design_based_and_attribution/supplementary_tables.xlsx` | Online Resource 1, Tables S0-S13 |

`value_identity_and_skill.py` and `fig9_robust_skill.py` start from the rounded annual values reported in the article, so they need no microdata and agree with the full pipeline up to rounding.

The ML covariates in `design_based_and_attribution.py` are a re-implementation of the covariate sets in `ml_premise_temporal_annual.py`, so its absolute baselines differ slightly from Table 7. Every comparison in that script (testing status, cohort, block order, jackknife) is made within the same run.

## Methods notes

* **Coding.** `ALC_STATUS = 2` is a test given; `0` (not given) and `1` (refused, 2016) are known not tested; every other value is unknown and is never counted as untested. `ALC_RES` 0-940 is a numeric BAC (x1000); `998` (positive, no value) resolves BAC > 0 but not BAC >= 0.08; other codes are unresolved. Drivers who died before the crash (`INJ_SEV = 6`) are excluded; the ML cohort keeps `INJ_SEV` 0-4.
* **Design-based variance** follows NHTSA DOT HS 812 688: strata `PSUSTRAT`, clusters `PSU_VAR`, weight `WEIGHT`, delete-one-PSU jackknife with the with-replacement approximation; domain estimates use the full annual sample. Single-PSU strata contribute no variance and are counted. Pooled ML test sets use `PSUSTRAT` across years as strata, which is conservative when PSUs recur across years.
* **Decision rule.** The RFV decision rule (OracleGain floor and retention thresholds) was frozen after the 2023 sanity check and before pooled 2016-2024 results were viewed; see `docs/decision_rule_freeze.md`. The gate writes the same record to `rfv_decision_rule_freeze.txt`.
* **Seeds.** 42, 1337, 2026, 31415, 271828.

## Supplementary audits

The scripts in `5_supplementary_audits/` are not used in the article. The PennDOT audit shows that PennDOT Statewide files carry final administrative alcohol flags but no CRSS-style test acquisition or result fields, so they cannot replace CRSS for the A/R/Z analysis. The drug audit inventories CRSS drug-testing fields before any drug replication; it does not build a drug RFV.

## Data and code availability

- **Source data:** Publicly available NHTSA Crash Report Sampling System (CRSS) files for 2016–2024 can be downloaded from [NHTSA CRSS](https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system). Source CRSS archives are **not** mirrored in this repository. Supplementary PennDOT data were provided through the [PennDOT Crash Information Tool](https://crashinfo.penndot.pa.gov/) and are also excluded.
- **Analysis code:** This repository includes the CRSS extraction/identification code, bounds/frontier analysis, machine-learning premise experiments, design-based analyses, and auxiliary diagnostic scripts.
- **Results:** No precomputed article results, derived CSV/XLSX reports, model predictions, or plot images are committed. The listed analytical scripts write outputs to the Git-ignored `outputs/` directory when run with the required input data. The original numerical figures and results were not independently re-created or verified by this repository packaging pass.
- **Licensing:** Code is shared under the MIT License in [`LICENSE`](LICENSE). This license does not cover NHTSA or PennDOT source data.

For manuscript-ready **Data availability** and **Code availability** wording, see [`docs/data_and_code_availability.md`](docs/data_and_code_availability.md).

## Citation

If you use this code, please cite the accompanying article (full reference to be added on publication).
