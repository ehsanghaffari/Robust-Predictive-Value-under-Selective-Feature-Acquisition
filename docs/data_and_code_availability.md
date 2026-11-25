# Data and code availability

Statements as they appear in the article.

## Data availability

The Crash Report Sampling System (CRSS) data analyzed in this study are publicly available from the National Highway Traffic Safety Administration (NHTSA) at https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system. The analyses use publicly available CRSS data from 2016 through 2024. Raw CRSS data are not redistributed by the author.

## Code availability

The CRSS analysis code used for this study (version 0.7) is publicly available at https://github.com/ehsanghaffari/Robust-Predictive-Value-under-Selective-Feature-Acquisition under the MIT License. It contains the data-extraction code with all CRSS variable mappings and code-semantics audits, the identification and frontier analyses, the machine-learning experiments, the case-level split identifiers for each seed of the pooled machine-learning experiment, the revision analyses (design-based jackknife, effective sample sizes, block-order and testing-status analyses, and conditional RFV), and the package requirements. Raw CRSS files are not redistributed; the pipeline downloads them from NHTSA and records their checksums. Complete identification results for both BAC definitions are reported in Online Resource 1; the machine-learning experiments use BAC > 0, and their acquisition contrast M1 − M0 does not depend on the BAC definition.

## Notes

- The scripts in `5_supplementary_audits/` use Pennsylvania crash data obtained through the [PennDOT Crash Information Tool](https://crashinfo.penndot.pa.gov/). These audits are not part of the article's analysis, and the PennDOT data are not redistributed.
- The MIT License covers the code only, not the NHTSA or PennDOT source data.
