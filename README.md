# Robust Predictive Value under Selective Feature Acquisition

Repository associated with the manuscript **“Robust Predictive Value under Selective Feature Acquisition: Exact Identification Boundaries with an Application to Crash Alcohol Testing.”**

## Data source

The intended crash-data source for this study is **Pennsylvania Department of Transportation (PennDOT), Pennsylvania Crash Information Tool (PCIT)**:

https://crashinfo.penndot.pa.gov/

No NHTSA Crash Report Sampling System (CRSS) data should be represented as the source of the PennDOT study.

## Provenance correction

A prior repository version contained scripts that downloaded NHTSA CRSS files and used CRSS-specific fields such as `ALC_STATUS` and `ALC_RES`. Those scripts were removed because they do **not** match the stated PennDOT-only data provenance.

The PennDOT statewide public files currently used in this project contain alcohol-related administrative flags such as `ALCOHOL_RELATED`, `DRINKING_DRIVER`, and `MC_DRINKING_DRIVER`; they do not expose the CRSS-style alcohol-test acquisition/result-resolution fields used by the removed scripts. Those administrative flags must not be relabeled as alcohol-test acquisition or BAC-result-resolution variables.

## Reproducibility status

A reproducibility release will be posted only after the empirical analysis is reconciled with the PennDOT-only source and the code reproduces the manuscript from the PennDOT data without substituting variables from another crash database.

Until that reconciliation is complete, this repository should **not** be cited as a complete reproducibility archive for the manuscript’s current empirical results.
