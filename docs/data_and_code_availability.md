# Data and code availability

Statements as they appear in the article.

## Data availability

The Crash Report Sampling System (CRSS) data analyzed in this study are publicly available from the National Highway Traffic Safety Administration (NHTSA) at https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system. The analyses use publicly available CRSS data from 2016 through 2024. Raw CRSS data are not redistributed by the author.

## Code availability

The CRSS analysis code used for this study (version 0.7, release tag `v0.7`) is publicly available at https://github.com/ehsanghaffari/Robust-Predictive-Value-under-Selective-Feature-Acquisition under the MIT License. It contains the data-extraction code with all CRSS variable mappings and code-semantics audits, the identification and frontier analyses, the machine-learning experiments, the case-level split identifiers for each seed of the pooled machine-learning experiment, the revision analyses (design-based jackknife, effective sample sizes, block-order and testing-status analyses, conditional RFV, and the machine-learning result-block analyses for BAC ≥ 0.08), and the package requirements. Raw CRSS files are not redistributed; the pipeline downloads them from NHTSA and records their checksums. Complete identification and machine-learning results for both BAC definitions are reported in Online Resource 1, which also gives the checks that the released pipeline reproduces the identification audit and the Table 7 fits (Tables S14 and S15). The acquisition contrast M1 − M0 uses neither BAC values nor result resolution, so it is shared by both definitions. Every reported machine-learning result that uses result resolution or the alcohol value (the pooled block-order analysis, the refitting jackknife, and the annual analysis) is given for BAC > 0 and for BAC ≥ 0.08; the temporal holdout analysis reports only M1 − M0, so its M2 fits were not repeated.

## Ethics approval and consent

Not applicable. The study uses only publicly available, de-identified CRSS records and involves no contact with human participants.

## Notes

- The scripts in `5_supplementary_audits/` use Pennsylvania crash data obtained through the [PennDOT Crash Information Tool](https://crashinfo.penndot.pa.gov/). These audits are not part of the article's analysis, and the PennDOT data are not redistributed.
- The MIT License covers the code only, not the NHTSA or PennDOT source data.
