# Robust Predictive Value under Selective Feature Acquisition

Reproducibility code for the manuscript **“Robust Predictive Value under Selective Feature Acquisition: Exact Identification Boundaries with an Application to Crash Alcohol Testing.”**

## Data source

The reported analyses in the manuscript use **public-use U.S. National Highway Traffic Safety Administration (NHTSA) Crash Report Sampling System (CRSS) data for 2016–2024**. The analysis scripts download the CRSS annual ZIP files from NHTSA and use CRSS variables including `ALC_STATUS` and `ALC_RES`.

**Pennsylvania Department of Transportation (PennDOT) statewide crash files are not used for the RFV, identification-boundary, OracleGain, or machine-learning results reported in the manuscript.** Some original local working directories happened to be named `penndot Data`; that folder name is only a local workspace label and does not identify the data source used by these scripts.

## Included code

This repository intentionally contains only the scripts needed for the reported analyses:

1. `CRSS_OTFA_Alcohol_Feasibility_Gate_v0_6_FROZEN_RFV.py` — downloads and audits CRSS 2016–2024, generates the frozen RFV gate inputs, annual acquisition/support outputs, pooled fatal-cell stability summaries, and the post-2023/pre-pooled decision-freeze record.
2. `02_CRSS_Theory_Accounting_Oracle_Finalizer.py` — deterministic theory/accounting calculations, MAS bounds, zero-RFV classification, boundary distances, assumption frontiers, OracleGain sensitivity calculations, and associated figures/tables.
3. `03_CRSS_ML_Premise_Temporal_Annual.py` — five-seed Logistic Regression/XGBoost/CatBoost premise test, forward temporal analysis, and within-year apparent-value/selectivity analysis.

Compatibility audits, drug-replication diagnostics, PennDOT supplementary experiments, batch files, generated results, and raw crash data are intentionally excluded because they are not required to reproduce the results reported in the manuscript.

## Run order

Install the Python dependencies in `requirements.txt`, then run the frozen gate first:

```bash
python CRSS_OTFA_Alcohol_Feasibility_Gate_v0_6_FROZEN_RFV.py --all-years --outputdir <output-directory>
```

Next run the deterministic theory finalizer after pointing its `BASE`/input paths to the directory containing `CRSS_OTFA_alcohol_gate_v0_6.csv`:

```bash
python 02_CRSS_Theory_Accounting_Oracle_Finalizer.py
```

Finally run the full machine-learning analysis after setting its `BASE` path to the desired local working directory:

```bash
python 03_CRSS_ML_Premise_Temporal_Annual.py
```

A quick one-seed Logistic Regression smoke test is also available:

```bash
python 03_CRSS_ML_Premise_Temporal_Annual.py --quick
```

The two later scripts retain the local paths used in the original analysis for provenance. Change only those local filesystem paths as needed; doing so does not alter the analysis definitions or decision rules.

## Reproducibility note

The primary gate script records the RFV decision rule as a **post-2023, pre-pooled validity correction** and preserves the frozen thresholds used for the decisive 2016–2024 evaluation. No raw CRSS data are redistributed in this repository.
