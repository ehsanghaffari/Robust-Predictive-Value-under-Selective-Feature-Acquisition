# Robust Predictive Value under Selective Feature Observation

Reproducibility code for the manuscript **“Robust Predictive Value under Selective Feature Observation: Exact Identification Boundaries with an Application to Police-Reported Driver Condition.”**

## Data source

The crash data used in this study come **only** from the Pennsylvania Department of Transportation (PennDOT) **Pennsylvania Crash Information Tool (PCIT)**:

https://crashinfo.penndot.pa.gov/

Download the annual public **Statewide** crash database ZIP files for 2016–2025 and place them in one local directory as `Statewide_2016.zip`, ..., `Statewide_2025.zip`. Raw crash data are not redistributed in this repository.

The empirical feature is the PennDOT `DVR_PED_CONDITION` field in the PERSON table. For drivers (`PERSON_TYPE=1`), codes 0–6 are treated as resolved conditions, code 9 or missing as unresolved, and code 1 is the administrative category **“Had Been Drinking.”** This is not a BAC or toxicology variable.

## Reproduce the analysis

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the complete identification/RFV analysis and five-seed LightGBM premise audit:

```bash
python PennDOT_RFV_Selective_Observation.py \
  --data-dir /path/to/penndot_statewide_zips \
  --output-dir results
```

For the deterministic identification/RFV calculations and figures only:

```bash
python PennDOT_RFV_Selective_Observation.py \
  --data-dir /path/to/penndot_statewide_zips \
  --output-dir results \
  --skip-ml
```

The script uses crash-record-number grouping for the machine-learning splits so records from the same crash do not cross train, validation, and test partitions.
