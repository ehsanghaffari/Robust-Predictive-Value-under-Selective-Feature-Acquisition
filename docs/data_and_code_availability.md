# Data and code availability

## Data availability

The Crash Report Sampling System (CRSS) data analyzed in this study are publicly available from the National Highway Traffic Safety Administration (NHTSA) at https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system. The analyses use CRSS data from 2016 through 2024. Raw CRSS data are not redistributed.

## Code availability

The CRSS analysis pipeline (version 0.7) is publicly available under the MIT License at the [exact analysis commit](https://github.com/ehsanghaffari/Robust-Predictive-Value-under-Selective-Feature-Acquisition/tree/4dbebd01dbde14c423191b6866b4b2530b02441f). The commit contains the variable mappings, code-semantics audits, identification and frontier analyses, machine-learning scripts, design-based analyses, pooled case-level split identifiers for all five seeds, and core package requirements. Run commands and output mappings are documented in its README. The pipeline records input checksums and package versions when executed.

Online Resource 1 reports annual identification results for both BAC > 0 and BAC ≥ 0.08, including prevalence bounds, separation margins, frontiers, effective sample sizes, and conditional RFV. The reported machine-learning experiments use BAC > 0. The acquisition-only contrast M1 − M0 does not use BAC values or result resolution and is invariant to the BAC threshold; contrasts involving the result block are specific to BAC > 0.

## Notes

- The scripts in `5_supplementary_audits/` are not part of the article's analysis. Pennsylvania crash data for these audits were obtained through the [PennDOT Crash Information Tool](https://crashinfo.penndot.pa.gov/) and are not redistributed.
- The MIT License covers the code only, not the NHTSA or PennDOT source data.
