# Robust Predictive Value under Selective Feature Acquisition

Analysis code for *Robust Predictive Value under Selective Feature Acquisition: Exact Identification Boundaries in Crash Alcohol Testing*.

## Repository structure

- `1_identification_gate/` — CRSS data preparation and annual identification quantities
- `2_bounds_and_frontier/` — RFV bounds, sensitivity analysis, and frontier calculations
- `3_ml_premise/` — machine-learning comparisons
- `4_design_based_and_attribution/` — uncertainty and attribution analyses
- `5_supplementary_audits/` — optional CRSS and PennDOT diagnostics
- `6_plot_figures/` — code for Figures 2, 3, and 5–7

## Data

The main analysis uses [NHTSA CRSS data (2016–2024)](https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system). Supplementary PennDOT data were obtained from the [PennDOT Crash Information Tool](https://crashinfo.penndot.pa.gov/).

Place input files in `data/` as described in [data/README.md](data/README.md). Source data are not included.

## Installation

Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

## Run

Run from the repository root:

```bash
python 1_identification_gate/crss_identification_gate.py --all-years
python 2_bounds_and_frontier/bounds_frontier_oracle.py
python 2_bounds_and_frontier/value_identity_and_skill.py
python 2_bounds_and_frontier/fig9_robust_skill.py
python 3_ml_premise/ml_premise_temporal_annual.py
python 4_design_based_and_attribution/design_based_and_attribution.py --ml --three-block --dagjk-groups 15 --dagjk-families lr,xgb
```

To recreate Figures 2, 3, and 5–7 from the analysis CSVs, run `python 6_plot_figures/plot_paper_figures.py` (or `RUN_FIGURES.cmd` on Windows).

See `--help` for script options. Generated outputs are written to `outputs/` and are not tracked in Git.

## Availability and license

See [data and code availability](docs/data_and_code_availability.md). Code is released under the [MIT License](LICENSE).
