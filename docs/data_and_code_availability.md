# Manuscript data and code availability statements

The statements below reflect the **files actually included in the public repository**. Do not claim that precomputed findings or all figure-generation code are hosted here.

## Data availability

The primary analysis used publicly available 2016–2024 Crash Report Sampling System (CRSS) data released by the National Highway Traffic Safety Administration (NHTSA). The files and documentation are available through the [NHTSA CRSS data portal](https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system). Pennsylvania crash data used for the supplementary schema and administrative-flag audits were provided through the [Pennsylvania Department of Transportation (PennDOT) Crash Information Tool](https://crashinfo.penndot.pa.gov/). The source data are available from their respective providers and are not redistributed in the GitHub repository.

## Code availability

Analysis code is publicly available at https://github.com/ehsanghaffari/Robust-Predictive-Value-under-Selective-Feature-Acquisition under the MIT License. The repository contains scripts for data preparation, identification, sensitivity analysis, and machine-learning evaluation.

## Scope and verification notes

- These statements do not promise individual requests for derived results or claim that published numeric outputs are archived on GitHub.
- Running scripts requires NHTSA source data and dependencies. The packaged code has not been fully re-executed against all original inputs during this repository edit.
- The supplementary PennDOT audits are separate from the CRSS empirical analysis and do not replace CRSS alcohol-acquisition variables.
- Verify the corresponding statements in the final manuscript separately; updating this repository does not automatically change the manuscript.
