# RFV decision rule freeze

**Post-2023 / pre-pooled validity correction. Frozen: 2026-09-03.**

## History

The 2023 CRSS sanity result was inspected before this correction. These rules are therefore not backdated or described as a pre-2023 preregistration. They were frozen before pooled 2016-2024 results were viewed.

## 1. Whole-population unresolved mass

The former 5% conservative unresolved-mass threshold is demoted to an accounting diagnostic only. It cannot pass or fail RFV feasibility, because severe nonresolution within the rare informative strata can appear small when divided by all eligible drivers.

## 2. Primary support diagnostics

Report, by year and BAC definition, the CRSS-weighted resolved-Z share among all survivors and among all fatalities, plus the resolved share among tested survivors and tested fatalities. Annual survivor and fatal testing rates are shown alongside pooled fatal a1/a0 counts.

## 3. BAC-definition-specific result resolution

`ALC_RES = 998` (positive reading, no actual value) resolves Z for BAC > 0 but does not resolve Z for BAC >= 0.08. Unresolved tested-Z masks are definition-specific and include every tested row not assigned to the corresponding positive or negative cell.

## 4. RFV / OracleGain evaluation population

Primary RFV and OracleGain are evaluated on a standardized diagnostic population:

- years 2016-2024 receive equal total weight;
- within each year, fatalities (Y = 1) and survivors (Y = 0) each receive 50% weight;
- CRSS survey weights are normalized within each year x outcome cell.

This population is used for feature-value diagnosis, not as a deployment prevalence estimate.

## 5. Primary gain metric

Absolute Brier-score reduction relative to the feature-deletion (pre-outcome-only) baseline.

## 6. OracleGain floor

Minimum absolute OracleGain = 0.010 Brier-score reduction. If OracleGain < 0.010, RFV/OracleGain retention is not reported, and the verdict is `INCONCLUSIVE_ORACLE_TOO_WEAK`.

## 7. RFV retention decision rule

Retention = RFV / OracleGain, computed only when OracleGain >= 0.010.

- retention >= 0.25: GO for the current robust-ML method;
- 0.10 <= retention < 0.25: STOP the current method and reframe;
- retention < 0.10: HARD STOP for the robust-prediction approach.

## 8. Role of pooled CRSS

Pooling 2016-2024 may stabilize the fatal a1/a0 raw cells. It does not erase annual acquisition regimes. Annual survivor and fatal testing and Z-resolution rates remain primary diagnostics and are reported rather than replaced by pooled rates.
