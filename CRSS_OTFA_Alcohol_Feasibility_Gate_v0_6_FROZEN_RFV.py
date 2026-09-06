#!/usr/bin/env python
"""
CRSS OTFA Alcohol Feasibility Gate v0.6
=======================================

PURPOSE
-------
Run the binding real-data feasibility gate BEFORE any further OTFA theory or
synthetic work.

Primary empirical quantity:
    P(A = 1 | Y = 0, driver)

where:
    A = alcohol test completed / "Test Given"
    Y = fatal injury
    driver = driver of a motor vehicle in-transport

This version explicitly prevents unresolved acquisition status from making the
RFV gate artificially optimistic.

IMPORTANT DESIGN CHANGES IN v0.3
--------------------------------
1) TWO survivor-unobserved masses are reported:
   - optimistic_u:
         survivors explicitly coded A=0 only
   - conservative_u:
         survivors explicitly coded A=0 PLUS survivors whose ALC_STATUS is
         unknown/not reported

   The conservative version governs feasibility. Unknown A is not allowed to
   shrink the ignorance mass.

2) TWO estimands are made explicit:
   - resolved-core estimand:
         renormalizes a1+a0+b1+b0+optimistic_u to 1
   - full-population conservative accounting:
         keeps original weighted masses and sends unknown-A survivors into u

   No RFV formula should silently treat an incomplete core as the full
   population.

3) Fatal non-ascertainment is surfaced BEFORE RFV:
       v_rate = 1 - P(A=1 | Y=1)
   If v_rate is materially > 0, Theorem 1 is not the operative real-data result;
   Proposition 1 is.

4) Fatal cell stability:
   a1/a0 raw counts are printed prominently.
   Annual fatal-side inference is NOT recommended when cells are thin.
   A pooled 2016-2024 fatal-side summary is produced after --all-years.

5) Tested-but-Z-unknown rows (997/995/999) are surfaced as a second
   unresolved layer. They are never silently assigned positive or negative Z.

IMPORTANT DESIGN CHANGES IN v0.6
--------------------------------
6) The former 5% whole-population unresolved-mass threshold is demoted to an
   accounting diagnostic. It cannot pass/fail feasibility.

7) Feasibility-relevant support is reported within outcome strata: resolved-Z
   share among all survivors and among all fatalities, separately for BAC>0
   and BAC>=.08.

8) ALC_RES=998 is definition-specific: it resolves BAC>0 but remains unresolved
   for BAC>=.08. v0.6 therefore uses definition-specific unresolved masks.

9) The RFV decision rule is frozen AFTER inspection of 2023 but BEFORE pooled
   2016-2024 results: primary gain metric = absolute Brier-score reduction;
   OracleGain floor = 0.010; retention GO threshold = 0.25.

10) RFV/OracleGain will be evaluated on an equal-year, within-year 50/50
    fatal/survivor standardized population with CRSS survey weights normalized
    within year x outcome. If OracleGain <0.010, retention is not reported and
    the verdict is INCONCLUSIVE_ORACLE_TOO_WEAK.

DEFAULT RUN
-----------
2023 only, for download/code sanity checking:
    python CRSS_OTFA_Alcohol_Feasibility_Gate_v0_6_FROZEN_RFV.py

After 2023 is confirmed:
    python CRSS_OTFA_Alcohol_Feasibility_Gate_v0_6_FROZEN_RFV.py --all-years

NO MACHINE-LEARNING MODEL IS TRAINED.
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
VERSION = '0.6'
ALL_YEARS = list(range(2016, 2025))
DEFAULT_SANITY_YEAR = 2023
NHTSA_STATIC = 'https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip'
PER_TYP_DRIVER = 1
INJ_FATAL = 4
INJ_SEV_UNKNOWN = 5
INJ_DIED_PRIOR = 6
INJ_UNKNOWN_NOT_REPORTED = 9
ALC_TEST_NOT_GIVEN = 0
ALC_TEST_REFUSED = 1
ALC_TEST_GIVEN = 2
ALC_NOT_REPORTED = 8
ALC_UNKNOWN = 9
ALC_RES_NOT_REPORTED = 995
ALC_RES_TEST_NOT_GIVEN = 996
ALC_RES_TESTED_UNKNOWN = 997
ALC_RES_POSITIVE_NO_VALUE = 998
ALC_RES_UNKNOWN = 999
FATAL_CELL_WARN = 50
FATAL_CELL_STRONG = 100
STATUS0_RES996_MIN_SHARE = 0.95
THEOREM1_MAX_V_RATE = 0.01
ACCOUNTING_UNRESOLVED_REFERENCE = 0.05
RFV_FREEZE_STATUS = 'POST_2023_PRE_POOLED_VALIDITY_CORRECTION'
RFV_FREEZE_DATE = '2026-09-03'
RFV_RETENTION_GO_MIN = 0.25
RFV_RETENTION_HARD_STOP_MAX = 0.1
RFV_PRIMARY_GAIN_METRIC = 'ABSOLUTE_BRIER_SCORE_REDUCTION'
ORACLE_GAIN_MIN_ABS_BRIER = 0.01
RFV_EVAL_POPULATION = 'EQUAL_YEAR_2016_2024__WITHIN_YEAR_50_50_FATAL_SURVIVOR__CRSS_WEIGHTS_NORMALIZED_WITHIN_YEAR_X_OUTCOME'

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def find_member(zf: zipfile.ZipFile, basename: str) -> str:
    target = basename.lower()
    matches = [n for n in zf.namelist() if Path(n).name.lower() == target]
    if not matches:
        raise FileNotFoundError(f'{basename} not found in ZIP. First members: {zf.namelist()[:30]}')
    return matches[0]

def read_csv_from_zip(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    raw = zf.read(member)
    last = None
    for enc in ('utf-8-sig', 'utf-8', 'latin1'):
        try:
            return norm_cols(pd.read_csv(io.BytesIO(raw), low_memory=False, encoding=enc))
        except Exception as exc:
            last = exc
    raise last

def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000000:
        log(f'Reusing {dest.name} ({dest.stat().st_size / 1000000.0:.1f} MB)')
        return
    log(f'Downloading {url}')
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    log(f'Saved {dest.name} ({dest.stat().st_size / 1000000.0:.1f} MB)')

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')

def wsum(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights[mask]))

def wrate(num_mask: np.ndarray, den_mask: np.ndarray, weights: np.ndarray) -> float:
    den = wsum(den_mask, weights)
    if den <= 0:
        return np.nan
    return wsum(num_mask, weights) / den

def raw_rate(num_mask: np.ndarray, den_mask: np.ndarray) -> float:
    den = int(np.sum(den_mask))
    if den <= 0:
        return np.nan
    return float(np.sum(num_mask) / den)

def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else np.nan

def fatal_cell_stability(n: int) -> str:
    if n < FATAL_CELL_WARN:
        return 'THIN_LT50_POOL_REQUIRED'
    if n < FATAL_CELL_STRONG:
        return 'MODERATE_50_TO_99_POOL_PREFERRED'
    return 'ADEQUATE_100_PLUS'

def code_semantics_check(driver_mask: np.ndarray, alc_status: np.ndarray, alc_res: np.ndarray) -> dict:
    """
    Empirical sanity check of the codebook mapping.

    If ALC_STATUS==0 truly means "test not given", essentially all such driver
    rows should carry ALC_RES==996 ("test not given"). The 95% threshold is a
    pre-specified data-semantic sanity gate, not a substantive research cutoff.
    """
    status0 = driver_mask & (alc_status == ALC_TEST_NOT_GIVEN)
    n0 = int(np.sum(status0))
    if n0 == 0:
        share = np.nan
        verdict = 'FAIL_NO_STATUS0_ROWS'
    else:
        share = float(np.mean(alc_res[status0] == ALC_RES_TEST_NOT_GIVEN))
        verdict = 'PASS' if share >= STATUS0_RES996_MIN_SHARE else 'FAIL_CODE_SEMANTICS'
    status2 = driver_mask & (alc_status == ALC_TEST_GIVEN)
    n2 = int(np.sum(status2))
    share_status2_996 = float(np.mean(alc_res[status2] == ALC_RES_TEST_NOT_GIVEN)) if n2 else np.nan
    return {'status0_driver_raw_n': n0, 'status0_with_ALC_RES_996_share': share, 'status2_driver_raw_n': n2, 'status2_with_ALC_RES_996_share': share_status2_996, 'code_semantics_verdict': verdict}

def hierarchical_gate_verdict(row: dict, zname: str='gt0') -> str:
    """
    Enforce the pre-RFV read order without using whole-population unresolved
    mass as a pass/fail criterion.

      1) code semantics
      2) v-rate / operative theorem
      3) within-outcome Z-resolution diagnostics
      4) fatal a1/a0 stability
      5) only then permit the later RFV probe

    The later RFV probe must additionally enforce the frozen OracleGain floor
    and retention threshold. This function does NOT calculate retention.
    """
    if row['code_semantics_verdict'] != 'PASS':
        return 'STOP_INVALID_CODE_SEMANTICS'
    v_rate = row['v_rate_weighted_1_minus_fatal_test_rate']
    if not np.isfinite(v_rate):
        return 'INCONCLUSIVE_FATAL_TEST_RATE_UNAVAILABLE'
    survivor_resolved = row.get(f'{zname}_survivor_Z_resolved_share_all_survivors', np.nan)
    fatal_resolved = row.get(f'{zname}_fatal_Z_resolved_share_all_fatal', np.nan)
    if not np.isfinite(survivor_resolved) or not np.isfinite(fatal_resolved):
        return 'INCONCLUSIVE_WITHIN_OUTCOME_Z_RESOLUTION_UNAVAILABLE'
    if survivor_resolved <= 0 or fatal_resolved <= 0:
        return 'INCONCLUSIVE_NO_RESOLVED_Z_SUPPORT_IN_OUTCOME_STRATUM'
    theory = 'PROP1' if v_rate > THEOREM1_MAX_V_RATE else 'THM1'
    a1_n = int(row[f'{zname}_a1_raw_n'])
    a0_n = int(row[f'{zname}_a0_raw_n'])
    if min(a1_n, a0_n) < FATAL_CELL_WARN:
        return f'POOL_FATAL_SIDE_REQUIRED_{theory}'
    if theory == 'PROP1':
        return 'READY_FOR_PROP1_RFV_PROBE_AFTER_FATAL_POOLING_CHECK'
    return 'READY_FOR_THEOREM1_RFV_PROBE_AFTER_FATAL_POOLING_CHECK'

def rfv_retention_verdict(oracle_gain: float, rfv_gain: float) -> tuple[float, str]:
    """Apply the frozen post-2023/pre-pooled RFV decision rule.

    This helper is intentionally not called by the extraction gate because v0.6
    still only prepares empirical inputs. The later RFV probe must call the same
    logic without changing thresholds after pooled CRSS results are viewed.
    """
    if not np.isfinite(oracle_gain) or not np.isfinite(rfv_gain):
        return (np.nan, 'INCONCLUSIVE_GAIN_UNAVAILABLE')
    if oracle_gain < ORACLE_GAIN_MIN_ABS_BRIER:
        return (np.nan, 'INCONCLUSIVE_ORACLE_TOO_WEAK')
    retention = float(rfv_gain / oracle_gain)
    if retention >= RFV_RETENTION_GO_MIN:
        return (retention, 'GO_RFV_RETENTION_GE_0_25')
    if retention < RFV_RETENTION_HARD_STOP_MAX:
        return (retention, 'HARD_STOP_RFV_RETENTION_LT_0_10')
    return (retention, 'STOP_AND_REFRAME_RFV_RETENTION_0_10_TO_LT_0_25')

def make_status_result_crosstab(year: int, p: pd.DataFrame, outdir: Path) -> None:
    tab = pd.crosstab(num(p['ALC_STATUS']), num(p['ALC_RES']), dropna=False)
    tab.index.name = 'ALC_STATUS'
    tab.columns.name = 'ALC_RES'
    tab.to_csv(outdir / f'{year}_ALC_STATUS_by_ALC_RES_raw_crosstab.csv')

def bac_masks(tested_mask: np.ndarray, alc_res: np.ndarray) -> dict[str, np.ndarray]:
    finite = np.isfinite(alc_res)
    numeric_bac = tested_mask & finite & (alc_res >= 0) & (alc_res <= 940)
    positive_unspecified = tested_mask & (alc_res == ALC_RES_POSITIVE_NO_VALUE)
    result_997 = tested_mask & (alc_res == ALC_RES_TESTED_UNKNOWN)
    result_995 = tested_mask & (alc_res == ALC_RES_NOT_REPORTED)
    result_999 = tested_mask & (alc_res == ALC_RES_UNKNOWN)
    gt0_positive = numeric_bac & (alc_res > 0) | positive_unspecified
    gt0_negative = numeric_bac & (alc_res == 0)
    gt0_known = gt0_positive | gt0_negative
    ge08_positive = numeric_bac & (alc_res >= 80)
    ge08_negative = numeric_bac & (alc_res < 80)
    ge08_known = ge08_positive | ge08_negative
    gt0_unresolved = tested_mask & ~gt0_known
    ge08_unresolved = tested_mask & ~ge08_known
    z_unknown_any = result_997 | result_995 | result_999
    return {'numeric': numeric_bac, 'positive_unspecified': positive_unspecified, 'result_997': result_997, 'result_995': result_995, 'result_999': result_999, 'z_unknown_any': z_unknown_any, 'gt0_positive': gt0_positive, 'gt0_negative': gt0_negative, 'gt0_known': gt0_known, 'gt0_unresolved': gt0_unresolved, 'ge08_positive': ge08_positive, 'ge08_negative': ge08_negative, 'ge08_known': ge08_known, 'ge08_unresolved': ge08_unresolved}

def add_mass(row, prefix, mask, weights, denominator_weight):
    raw_n = int(np.sum(mask))
    wc = wsum(mask, weights)
    wm = safe_ratio(wc, denominator_weight)
    row[f'{prefix}_raw_n'] = raw_n
    row[f'{prefix}_weighted_count'] = wc
    row[f'{prefix}_weighted_mass'] = wm

def analyze_year(year: int, zpath: Path, outdir: Path) -> dict:
    log(f'{year}: reading CRSS ZIP')
    with zipfile.ZipFile(zpath, 'r') as zf:
        person = read_csv_from_zip(zf, find_member(zf, 'person.csv'))
        accident = read_csv_from_zip(zf, find_member(zf, 'accident.csv'))
    req_p = {'CASENUM', 'PER_TYP', 'INJ_SEV', 'ALC_STATUS', 'ALC_RES'}
    missing_p = req_p - set(person.columns)
    if missing_p:
        raise KeyError(f'{year}: person.csv missing {sorted(missing_p)}')
    req_a = {'CASENUM', 'WEIGHT'}
    missing_a = req_a - set(accident.columns)
    if missing_a:
        raise KeyError(f'{year}: accident.csv missing {sorted(missing_a)}')
    aw = accident[['CASENUM', 'WEIGHT']].drop_duplicates('CASENUM').copy()
    aw['WEIGHT'] = num(aw['WEIGHT'])
    if 'WEIGHT' in person.columns:
        p = person.copy()
        p['WEIGHT'] = num(p['WEIGHT'])
        check = p[['CASENUM', 'WEIGHT']].drop_duplicates('CASENUM').merge(aw.rename(columns={'WEIGHT': 'WEIGHT_ACCIDENT'}), on='CASENUM', how='left', validate='one_to_one')
        if check['WEIGHT_ACCIDENT'].isna().any():
            raise ValueError(f"{year}: {int(check['WEIGHT_ACCIDENT'].isna().sum())} CASENUM values in person.csv did not match accident.csv weights.")
        both = check['WEIGHT'].notna() & check['WEIGHT_ACCIDENT'].notna()
        if both.any():
            diff = np.abs(check.loc[both, 'WEIGHT'].to_numpy(float) - check.loc[both, 'WEIGHT_ACCIDENT'].to_numpy(float))
            max_diff = float(np.nanmax(diff)) if len(diff) else 0.0
            if not np.allclose(check.loc[both, 'WEIGHT'].to_numpy(float), check.loc[both, 'WEIGHT_ACCIDENT'].to_numpy(float), rtol=1e-09, atol=1e-09, equal_nan=True):
                raise ValueError(f'{year}: person.csv and accident.csv WEIGHT disagree; max absolute difference={max_diff:.12g}')
        weight_source = 'PERSON_EXISTING_CROSSCHECKED_AGAINST_ACCIDENT'
    else:
        p = person.merge(aw, on='CASENUM', how='left', validate='many_to_one')
        weight_source = 'ACCIDENT_MERGED_INTO_PERSON'
    if p['WEIGHT'].isna().any():
        raise ValueError(f"{year}: {int(p['WEIGHT'].isna().sum())} person rows missing case weight.")
    make_status_result_crosstab(year, p, outdir)
    per_typ = num(p['PER_TYP']).to_numpy(float)
    inj = num(p['INJ_SEV']).to_numpy(float)
    alc_status = num(p['ALC_STATUS']).to_numpy(float)
    alc_res = num(p['ALC_RES']).to_numpy(float)
    weights = num(p['WEIGHT']).to_numpy(float)
    driver = per_typ == PER_TYP_DRIVER
    semantic = code_semantics_check(driver, alc_status, alc_res)
    fatal = inj == INJ_FATAL
    died_prior = inj == INJ_DIED_PRIOR
    eligible = driver & ~died_prior
    fatal_driver = eligible & fatal
    survivor_driver = eligible & ~fatal
    inj5_survivor = survivor_driver & (inj == INJ_SEV_UNKNOWN)
    inj9_survivor = survivor_driver & (inj == INJ_UNKNOWN_NOT_REPORTED)
    test_given = alc_status == ALC_TEST_GIVEN
    known_a0 = (alc_status == ALC_TEST_NOT_GIVEN) | (alc_status == ALC_TEST_REFUSED)
    known_a = test_given | known_a0
    unknown_a = ~known_a
    tested_survivor = survivor_driver & test_given
    tested_fatal = fatal_driver & test_given
    survivor_known_a0 = survivor_driver & known_a0
    fatal_known_a0 = fatal_driver & known_a0
    survivor_unknown_a = survivor_driver & unknown_a
    fatal_unknown_a = fatal_driver & unknown_a
    sres = bac_masks(tested_survivor, alc_res)
    fres = bac_masks(tested_fatal, alc_res)
    total_w = wsum(eligible, weights)
    survivor_w = wsum(survivor_driver, weights)
    fatal_w = wsum(fatal_driver, weights)
    row = {'year': year, 'weight_source': weight_source, 'eligible_driver_raw_n': int(np.sum(eligible)), 'eligible_driver_weighted_count': total_w, 'survivor_driver_raw_n': int(np.sum(survivor_driver)), 'survivor_driver_weighted_count': survivor_w, 'fatal_driver_raw_n': int(np.sum(fatal_driver)), 'fatal_driver_weighted_count': fatal_w, 'survivor_test_rate_raw_known_A': raw_rate(tested_survivor, survivor_driver & known_a), 'survivor_test_rate_weighted_known_A': wrate(tested_survivor, survivor_driver & known_a, weights), 'survivor_test_rate_raw_all_survivors': raw_rate(tested_survivor, survivor_driver), 'survivor_test_rate_weighted_all_survivors': wrate(tested_survivor, survivor_driver, weights), 'fatal_test_rate_raw_all_fatal': raw_rate(tested_fatal, fatal_driver), 'fatal_test_rate_weighted_all_fatal': wrate(tested_fatal, fatal_driver, weights), 'survivor_unknown_A_raw_n': int(np.sum(survivor_unknown_a)), 'survivor_unknown_A_weighted_share': wrate(survivor_unknown_a, survivor_driver, weights), 'fatal_unknown_A_raw_n': int(np.sum(fatal_unknown_a)), 'fatal_unknown_A_weighted_share': wrate(fatal_unknown_a, fatal_driver, weights), 'survivor_INJ_SEV_5_raw_n': int(np.sum(inj5_survivor)), 'survivor_INJ_SEV_5_weighted_share': wrate(inj5_survivor, survivor_driver, weights), 'survivor_INJ_SEV_9_raw_n': int(np.sum(inj9_survivor)), 'survivor_INJ_SEV_9_weighted_share': wrate(inj9_survivor, survivor_driver, weights), 'tested_survivor_raw_n': int(np.sum(tested_survivor)), 'tested_fatal_raw_n': int(np.sum(tested_fatal))}
    row.update(semantic)
    fatal_test_rate = row['fatal_test_rate_weighted_all_fatal']
    row['v_rate_weighted_1_minus_fatal_test_rate'] = 1.0 - fatal_test_rate if np.isfinite(fatal_test_rate) else np.nan
    if np.isfinite(row['v_rate_weighted_1_minus_fatal_test_rate']):
        if row['v_rate_weighted_1_minus_fatal_test_rate'] <= 0.01:
            row['operative_theory'] = 'THEOREM_1_APPROXIMATION_PLAUSIBLE'
        else:
            row['operative_theory'] = 'PROPOSITION_1_REQUIRED_V_GT_0'
    else:
        row['operative_theory'] = 'UNDETERMINED'
    for group, masks, tested in (('survivor', sres, tested_survivor), ('fatal', fres, tested_fatal)):
        for code, key in (('995', 'result_995'), ('997', 'result_997'), ('998', 'positive_unspecified'), ('999', 'result_999')):
            row[f'{group}_ALC_RES_{code}_raw_n'] = int(np.sum(masks[key]))
            row[f'{group}_ALC_RES_{code}_weighted_share_of_tested'] = wrate(masks[key], tested, weights)
        row[f'{group}_tested_Z_unknown_any_raw_n'] = int(np.sum(masks['z_unknown_any']))
        row[f'{group}_tested_Z_unknown_any_weighted_share_of_tested'] = wrate(masks['z_unknown_any'], tested, weights)
    row['q_test_BAC_gt0_raw'] = raw_rate(sres['gt0_positive'], sres['gt0_known'])
    row['q_test_BAC_gt0_weighted'] = wrate(sres['gt0_positive'], sres['gt0_known'], weights)
    row['q_test_BAC_ge008_raw'] = raw_rate(sres['ge08_positive'], sres['ge08_known'])
    row['q_test_BAC_ge008_weighted'] = wrate(sres['ge08_positive'], sres['ge08_known'], weights)
    zspecs = {'gt0': {'s_pos': sres['gt0_positive'], 's_neg': sres['gt0_negative'], 's_known': sres['gt0_known'], 's_unresolved': sres['gt0_unresolved'], 'f_pos': fres['gt0_positive'], 'f_neg': fres['gt0_negative'], 'f_known': fres['gt0_known'], 'f_unresolved': fres['gt0_unresolved']}, 'ge008': {'s_pos': sres['ge08_positive'], 's_neg': sres['ge08_negative'], 's_known': sres['ge08_known'], 's_unresolved': sres['ge08_unresolved'], 'f_pos': fres['ge08_positive'], 'f_neg': fres['ge08_negative'], 'f_known': fres['ge08_known'], 'f_unresolved': fres['ge08_unresolved']}}
    for zname, spec in zspecs.items():
        a1mask = spec['f_pos']
        a0mask = spec['f_neg']
        b1mask = spec['s_pos']
        b0mask = spec['s_neg']
        u_opt_mask = survivor_known_a0
        u_cons_mask = survivor_known_a0 | survivor_unknown_a
        add_mass(row, f'{zname}_a1', a1mask, weights, total_w)
        add_mass(row, f'{zname}_a0', a0mask, weights, total_w)
        add_mass(row, f'{zname}_b1', b1mask, weights, total_w)
        add_mass(row, f'{zname}_b0', b0mask, weights, total_w)
        add_mass(row, f'{zname}_u_optimistic', u_opt_mask, weights, total_w)
        add_mass(row, f'{zname}_u_conservative', u_cons_mask, weights, total_w)
        add_mass(row, f'{zname}_survivor_tested_Z_unresolved', spec['s_unresolved'], weights, total_w)
        add_mass(row, f'{zname}_fatal_tested_Z_unresolved', spec['f_unresolved'], weights, total_w)
        add_mass(row, f'{zname}_fatal_A0', fatal_known_a0, weights, total_w)
        add_mass(row, f'{zname}_fatal_A_unknown', fatal_unknown_a, weights, total_w)
        row[f'{zname}_survivor_Z_resolved_share_all_survivors'] = wrate(spec['s_known'], survivor_driver, weights)
        row[f'{zname}_fatal_Z_resolved_share_all_fatal'] = wrate(spec['f_known'], fatal_driver, weights)
        row[f'{zname}_survivor_Z_resolved_share_of_tested'] = wrate(spec['s_known'], tested_survivor, weights)
        row[f'{zname}_fatal_Z_resolved_share_of_tested'] = wrate(spec['f_known'], tested_fatal, weights)
        full_partition_masks = [a1mask, a0mask, spec['f_unresolved'], fatal_known_a0, fatal_unknown_a, b1mask, b0mask, spec['s_unresolved'], survivor_known_a0, survivor_unknown_a]
        partition_sum_w = sum((wsum(m, weights) for m in full_partition_masks))
        row[f'{zname}_full_partition_mass_sum'] = safe_ratio(partition_sum_w, total_w)
        row[f'{zname}_full_partition_mass_residual'] = 1.0 - row[f'{zname}_full_partition_mass_sum'] if np.isfinite(row[f'{zname}_full_partition_mass_sum']) else np.nan
        opt_core = sum((row[f'{zname}_{k}_weighted_mass'] for k in ('a1', 'a0', 'b1', 'b0', 'u_optimistic')))
        cons_core = sum((row[f'{zname}_{k}_weighted_mass'] for k in ('a1', 'a0', 'b1', 'b0', 'u_conservative')))
        row[f'{zname}_optimistic_core_mass'] = opt_core
        row[f'{zname}_conservative_core_mass'] = cons_core
        row[f'{zname}_optimistic_unresolved_mass'] = max(0.0, 1.0 - opt_core)
        row[f'{zname}_conservative_unresolved_mass'] = max(0.0, 1.0 - cons_core)
        row[f'{zname}_resolved_core_denominator_optimistic'] = opt_core
        row[f'{zname}_resolved_core_denominator_conservative'] = cons_core
        if opt_core > 0:
            for k in ('a1', 'a0', 'b1', 'b0', 'u_optimistic'):
                row[f'{zname}_{k}_renorm_resolved_core'] = row[f'{zname}_{k}_weighted_mass'] / opt_core
        if cons_core > 0:
            for k in ('a1', 'a0', 'b1', 'b0', 'u_conservative'):
                row[f'{zname}_{k}_renorm_conservative_core'] = row[f'{zname}_{k}_weighted_mass'] / cons_core
        row[f'{zname}_a1_stability'] = fatal_cell_stability(row[f'{zname}_a1_raw_n'])
        row[f'{zname}_a0_stability'] = fatal_cell_stability(row[f'{zname}_a0_raw_n'])
    row['pre_rfv_verdict_gt0'] = hierarchical_gate_verdict(row, 'gt0')
    row['pre_rfv_verdict_ge008'] = hierarchical_gate_verdict(row, 'ge008')
    return row

def pooled_fatal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pool raw fatal-side cells only for stability; keep acquisition annual.

    CRSS weights are year-specific, so weighted masses are not naively summed.
    The table deliberately reports annual min/median/max acquisition and
    within-outcome Z-resolution alongside pooled raw fatal a1/a0 counts.
    """
    rows = []
    for zname in ('gt0', 'ge008'):
        rows.append({'z_definition': zname, 'years': f'{int(df.year.min())}-{int(df.year.max())}', 'pooled_raw_a1': int(df[f'{zname}_a1_raw_n'].sum()), 'pooled_raw_a0': int(df[f'{zname}_a0_raw_n'].sum()), 'pooled_raw_fatal_tested': int(df['tested_fatal_raw_n'].sum()), 'pooled_a1_stability': fatal_cell_stability(int(df[f'{zname}_a1_raw_n'].sum())), 'pooled_a0_stability': fatal_cell_stability(int(df[f'{zname}_a0_raw_n'].sum())), 'median_annual_survivor_test_rate_weighted_all': float(df['survivor_test_rate_weighted_all_survivors'].median()), 'min_annual_survivor_test_rate_weighted_all': float(df['survivor_test_rate_weighted_all_survivors'].min()), 'max_annual_survivor_test_rate_weighted_all': float(df['survivor_test_rate_weighted_all_survivors'].max()), 'median_annual_fatal_test_rate_weighted_all': float(df['fatal_test_rate_weighted_all_fatal'].median()), 'min_annual_fatal_test_rate_weighted_all': float(df['fatal_test_rate_weighted_all_fatal'].min()), 'max_annual_fatal_test_rate_weighted_all': float(df['fatal_test_rate_weighted_all_fatal'].max()), 'median_annual_v_rate_weighted': float(df['v_rate_weighted_1_minus_fatal_test_rate'].median()), 'median_annual_survivor_Z_resolved_share': float(df[f'{zname}_survivor_Z_resolved_share_all_survivors'].median()), 'min_annual_survivor_Z_resolved_share': float(df[f'{zname}_survivor_Z_resolved_share_all_survivors'].min()), 'max_annual_survivor_Z_resolved_share': float(df[f'{zname}_survivor_Z_resolved_share_all_survivors'].max()), 'median_annual_fatal_Z_resolved_share': float(df[f'{zname}_fatal_Z_resolved_share_all_fatal'].median()), 'min_annual_fatal_Z_resolved_share': float(df[f'{zname}_fatal_Z_resolved_share_all_fatal'].min()), 'max_annual_fatal_Z_resolved_share': float(df[f'{zname}_fatal_Z_resolved_share_all_fatal'].max())})
    return pd.DataFrame(rows)

def rfv_freeze_record() -> dict:
    return {'freeze_status': RFV_FREEZE_STATUS, 'freeze_date': RFV_FREEZE_DATE, 'history': '2023 CRSS sanity results were inspected before this correction; rules frozen before pooled 2016-2024 results.', 'primary_gain_metric': RFV_PRIMARY_GAIN_METRIC, 'evaluation_population': RFV_EVAL_POPULATION, 'oracle_gain_min_absolute_brier': ORACLE_GAIN_MIN_ABS_BRIER, 'retention_go_min': RFV_RETENTION_GO_MIN, 'retention_hard_stop_below': RFV_RETENTION_HARD_STOP_MAX, 'retention_mid_band': '0.10 <= retention < 0.25 => STOP_AND_REFRAME', 'oracle_floor_rule': 'If OracleGain < 0.010 absolute Brier reduction, do not report RFV/OracleGain retention; verdict=INCONCLUSIVE_ORACLE_TOO_WEAK.', 'whole_population_unresolved_rule': 'Whole-population conservative unresolved mass is accounting-only and cannot pass or fail RFV feasibility.'}

def write_rfv_freeze(outdir: Path) -> None:
    record = rfv_freeze_record()
    jpath = outdir / 'CRSS_OTFA_RFV_decision_freeze_POST2023_PREPOOLED_v0_6.json'
    tpath = outdir / 'CRSS_OTFA_RFV_decision_freeze_POST2023_PREPOOLED_v0_6.txt'
    jpath.write_text(json.dumps(record, indent=2), encoding='utf-8')
    lines = ['CRSS OTFA RFV DECISION FREEZE - POST-2023 / PRE-POOLED', '=' * 66] + [f'{k}: {v}' for k, v in record.items()]
    tpath.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def annual_support_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['year', 'survivor_test_rate_weighted_all_survivors', 'fatal_test_rate_weighted_all_fatal', 'v_rate_weighted_1_minus_fatal_test_rate', 'gt0_survivor_Z_resolved_share_all_survivors', 'gt0_fatal_Z_resolved_share_all_fatal', 'gt0_survivor_Z_resolved_share_of_tested', 'gt0_fatal_Z_resolved_share_of_tested', 'ge008_survivor_Z_resolved_share_all_survivors', 'ge008_fatal_Z_resolved_share_all_fatal', 'ge008_survivor_Z_resolved_share_of_tested', 'ge008_fatal_Z_resolved_share_of_tested', 'survivor_unknown_A_weighted_share', 'fatal_unknown_A_weighted_share']
    return df[cols].copy()

def print_year_summary(row: dict) -> None:
    print()
    print('=' * 82)
    print(f"CRSS OTFA ALCOHOL GATE v{VERSION} — {row['year']}")
    print('=' * 82)
    print(f"Weight source: {row['weight_source']}")
    print('\n0) CODE SEMANTICS — READ THIS FIRST')
    print(f"ALC_STATUS=0 driver rows: {row['status0_driver_raw_n']:,}")
    print(f"Share with ALC_RES=996: {row['status0_with_ALC_RES_996_share']:.2%}")
    print(f"ALC_STATUS=2 rows incorrectly carrying ALC_RES=996: {row['status2_with_ALC_RES_996_share']:.3%}")
    print(f"Code-semantics verdict: {row['code_semantics_verdict']}")
    print('\n1) OPERATIVE THEORY CHECK')
    print(f"Fatal drivers raw n: {row['fatal_driver_raw_n']:,}")
    print(f"Fatal alcohol testing rate (weighted): {row['fatal_test_rate_weighted_all_fatal']:.2%}")
    print(f"v-rate = 1 - fatal test rate: {row['v_rate_weighted_1_minus_fatal_test_rate']:.2%}")
    print(f"Operative result: {row['operative_theory']}")
    print('\n2) SURVIVOR ACQUISITION')
    print(f"Surviving drivers raw n: {row['survivor_driver_raw_n']:,}")
    print(f"Test rate, known-A denominator: {row['survivor_test_rate_weighted_known_A']:.2%}")
    print(f"Test rate, all-survivor denominator: {row['survivor_test_rate_weighted_all_survivors']:.2%}")
    print(f"Unknown/not-reported A share: {row['survivor_unknown_A_weighted_share']:.2%}")
    print('\n3) WITHIN-OUTCOME Z SUPPORT — FEASIBILITY-RELEVANT DENOMINATORS')
    print(f"BAC>0 resolved Z among ALL survivors: {row['gt0_survivor_Z_resolved_share_all_survivors']:.2%}")
    print(f"BAC>0 resolved Z among ALL fatalities: {row['gt0_fatal_Z_resolved_share_all_fatal']:.2%}")
    print(f"BAC>0 resolved Z among TESTED survivors: {row['gt0_survivor_Z_resolved_share_of_tested']:.2%}")
    print(f"BAC>0 resolved Z among TESTED fatalities: {row['gt0_fatal_Z_resolved_share_of_tested']:.2%}")
    print(f"BAC>=.08 resolved Z among ALL survivors: {row['ge008_survivor_Z_resolved_share_all_survivors']:.2%}")
    print(f"BAC>=.08 resolved Z among ALL fatalities: {row['ge008_fatal_Z_resolved_share_all_fatal']:.2%}")
    print('Legacy unresolved-result-code diagnostic (995/997/999 only; 998 is resolved for BAC>0 but unresolved for BAC>=.08):')
    print(f"  survivor share of tested: {row['survivor_tested_Z_unknown_any_weighted_share_of_tested']:.2%} (raw n={row['survivor_tested_Z_unknown_any_raw_n']:,})")
    print(f"  fatal share of tested: {row['fatal_tested_Z_unknown_any_weighted_share_of_tested']:.2%} (raw n={row['fatal_tested_Z_unknown_any_raw_n']:,})")
    print('\n4) OUTCOME-CODE UNCERTAINTY ABSORBED INTO Y=0')
    print(f"INJ_SEV=5 weighted survivor share: {row['survivor_INJ_SEV_5_weighted_share']:.3%}")
    print(f"INJ_SEV=9 weighted survivor share: {row['survivor_INJ_SEV_9_weighted_share']:.3%}")
    print('\n5) q_test')
    print(f"BAC>0 q_test weighted: {row['q_test_BAC_gt0_weighted']:.2%}")
    print(f"BAC>=.08 q_test weighted: {row['q_test_BAC_ge008_weighted']:.2%}")
    print('\n6) RFV INPUTS — BAC > 0')
    for k in ('a1', 'a0', 'b1', 'b0'):
        print(f"{k}: mass={row[f'gt0_{k}_weighted_mass']:.8f}, raw n={row[f'gt0_{k}_raw_n']:,}")
    print(f"a1 stability: {row['gt0_a1_stability']}")
    print(f"a0 stability: {row['gt0_a0_stability']}")
    print(f"u optimistic (known A=0 only): {row['gt0_u_optimistic_weighted_mass']:.8f}")
    print(f"u conservative (+ unknown A survivors): {row['gt0_u_conservative_weighted_mass']:.8f}")
    print('\n7) MASS ACCOUNTING — DIAGNOSTIC ONLY')
    print(f"Optimistic core mass: {row['gt0_optimistic_core_mass']:.6f}; population residual={row['gt0_optimistic_unresolved_mass']:.6f}")
    print(f"Conservative core mass: {row['gt0_conservative_core_mass']:.6f}; population residual={row['gt0_conservative_unresolved_mass']:.6f}")
    print(f"BAC>0 full partition sum: {row['gt0_full_partition_mass_sum']:.9f}; residual={row['gt0_full_partition_mass_residual']:.3e}")
    print(f"BAC>=.08 full partition sum: {row['ge008_full_partition_mass_sum']:.9f}; residual={row['ge008_full_partition_mass_residual']:.3e}")
    print('The former 5% whole-population unresolved threshold is NOT a feasibility gate in v0.6.')
    print('\n8) PRE-RFV HIERARCHICAL VERDICT')
    print(f"BAC>0:    {row['pre_rfv_verdict_gt0']}")
    print(f"BAC>=.08: {row['pre_rfv_verdict_ge008']}")
    print('Later RFV decision: OracleGain must first be >=0.010 absolute Brier gain on the frozen standardized population; only then report retention.')
    print('Retention >=0.25 => GO; 0.10 to <0.25 => STOP/REFRAME; <0.10 => HARD STOP.')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--zipdir', default=str(Path.home() / 'Downloads' / 'CRSS_OTFA_gate' / 'zips'), help='Directory used to cache downloaded CRSS ZIP files. Default preserves the prior CRSS_OTFA_gate ZIP cache.')
    ap.add_argument('--outputdir', default=str(Path.home() / 'Downloads' / 'penndot Data' / 'Output'), help='Directory for all generated CSV/JSON/crosstab outputs. Default: ~/Downloads/penndot Data/Output')
    ap.add_argument('--years', nargs='*', type=int, default=None, help='Explicit years, e.g. --years 2023. Default is 2023 only.')
    ap.add_argument('--all-years', action='store_true', help='Run 2016-2024 after the 2023 sanity check.')
    args = ap.parse_args()
    if args.all_years:
        years = ALL_YEARS
    elif args.years:
        years = args.years
    else:
        years = [DEFAULT_SANITY_YEAR]
    bad = [y for y in years if y not in ALL_YEARS]
    if bad:
        raise ValueError(f'Unsupported years: {bad}')
    zdir = Path(args.zipdir).expanduser()
    outdir = Path(args.outputdir).expanduser()
    zdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    log(f'CRSS OTFA Alcohol Gate v{VERSION}')
    log(f'Years requested: {years}')
    log(f'CRSS ZIP cache: {zdir}')
    log(f'Output directory: {outdir}')
    log('No ML training will occur.')
    write_rfv_freeze(outdir)
    log(f'RFV freeze status: {RFV_FREEZE_STATUS}')
    log(f'RFV primary gain metric: {RFV_PRIMARY_GAIN_METRIC}')
    log(f'RFV evaluation population: {RFV_EVAL_POPULATION}')
    log(f'OracleGain floor: {ORACLE_GAIN_MIN_ABS_BRIER:.3f} absolute Brier')
    log(f'RFV retention GO threshold: {RFV_RETENTION_GO_MIN:.2f}')
    rows = []
    for year in years:
        url = NHTSA_STATIC.format(year=year)
        zpath = zdir / f'CRSS{year}CSV.zip'
        download(url, zpath)
        row = analyze_year(year, zpath, outdir)
        rows.append(row)
        print_year_summary(row)
    df = pd.DataFrame(rows).sort_values('year')
    csv_path = outdir / 'CRSS_OTFA_alcohol_gate_v0_6.csv'
    json_path = outdir / 'CRSS_OTFA_alcohol_gate_v0_6.json'
    support_path = outdir / 'CRSS_OTFA_annual_acquisition_support_v0_6.csv'
    df.to_csv(csv_path, index=False)
    annual_support_table(df).to_csv(support_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
    log(f'Saved: {csv_path}')
    log(f'Saved: {json_path}')
    log(f'Saved: {support_path}')
    if len(df) > 1:
        pool = pooled_fatal_summary(df)
        pool_path = outdir / 'CRSS_OTFA_pooled_fatal_stability_v0_6.csv'
        pool.to_csv(pool_path, index=False)
        log(f'Saved: {pool_path}')
        print()
        print('=' * 82)
        print('ANNUAL ACQUISITION / Z-SUPPORT — DO NOT HIDE REGIME SHIFTS BY POOLING')
        print('=' * 82)
        display_cols = ['year', 'survivor_test_rate_weighted_all_survivors', 'fatal_test_rate_weighted_all_fatal', 'gt0_survivor_Z_resolved_share_all_survivors', 'gt0_fatal_Z_resolved_share_all_fatal']
        annual_display = df[display_cols].copy()
        for c in display_cols[1:]:
            annual_display[c] = annual_display[c].map(lambda x: f'{x:.2%}')
        print(annual_display.to_string(index=False))
        print()
        print('=' * 82)
        print('POOLED FATAL-SIDE STABILITY + ANNUAL RATE RANGE')
        print('=' * 82)
        print(pool.to_string(index=False))
    if years == [DEFAULT_SANITY_YEAR]:
        print()
        print('=' * 82)
        print('NEXT ACTION')
        print('=' * 82)
        print(f'Inspect {DEFAULT_SANITY_YEAR} in this exact order:\n  1. ALC_STATUS x ALC_RES crosstab / code-semantics verdict,\n  2. v-rate and operative theorem,\n  3. within-outcome resolved-Z shares,\n  4. full partition arithmetic residual,\n  5. a1/a0 raw counts and stability.\nWhole-population unresolved mass is accounting-only in v0.6.')
        print()
        print('Only if the coding/rates look plausible, run:')
        print(f'  python "{Path(sys.argv[0]).name}" --all-years')
if __name__ == '__main__':
    main()
