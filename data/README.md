# data

Input data are not tracked in this repository.

* `crss_zips/` : NHTSA CRSS files `CRSS2016CSV.zip` ... `CRSS2024CSV.zip`. Downloaded automatically by `1_identification_gate/crss_identification_gate.py` and `3_ml_premise/ml_premise_temporal_annual.py`, or with `--download` by `4_design_based_and_attribution/design_based_and_attribution.py`.
* `penndot/` : PennDOT Statewide files obtained through the [PennDOT Crash Information Tool](https://crashinfo.penndot.pa.gov/), needed only for `5_supplementary_audits/` (`penndot/Statewide_<year>/<TABLE>_<year>.csv`). Not redistributed.
