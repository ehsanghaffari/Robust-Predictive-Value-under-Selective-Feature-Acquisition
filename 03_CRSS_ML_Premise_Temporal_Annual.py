#!/usr/bin/env python
"""
CRSS ML Premise + Temporal + Annual Selectivity Audit
=====================================================

Purpose
-------
Run the manuscript's conventional-ML strengthening experiments on CRSS,
NOT on PennDOT Statewide files. The PennDOT files do not contain direct
ALC_STATUS/ALC_RES equivalents and must not be substituted for A/R/Z.

Experiments
-----------
1) Pooled premise test (2016-2024):
     M0 = W
     M1 = W + A
     M2 = W + A + R + RZ
   where A = alcohol test given, R = BAC>0 state resolved, RZ = resolved-positive.

2) Forward temporal test:
     train on 2016-2019; evaluate unchanged on 2020, 2021, ..., 2024.
     M0 is the temporal-shift control.

3) Within-year apparent-value test:
     fit M0 and M2 separately in each year and relate M2-M0 apparent value
     to q_test.

Model families
--------------
Regularized Logistic Regression, XGBoost, CatBoost.
No neural baseline.

Weighting
---------
CRSS case weights are used as sample weights in model fitting and in point
metrics. Seed variability is reported. This script does NOT claim design-based
survey standard errors; any final inferential CI should use the CRSS survey
design variables if the paper requires design-based inference.

Outputs
-------
C:\\Users\\ehsanghaffari\\Downloads\\penndot Data\\Output\\New Output\\CRSS_ML

Usage
-----
Smoke test:
  python 03_CRSS_ML_Premise_Temporal_Annual.py --quick

Full run:
  python 03_CRSS_ML_Premise_Temporal_Annual.py
"""
from __future__ import annotations
import argparse
import io
import json
import math
import sys
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False
BASE = Path('C:\\Users\\ehsanghaffari\\Downloads\\penndot Data')
OUT = BASE / 'Output' / 'New Output' / 'CRSS_ML'
ZIP_DIR = OUT / 'zips'
CACHE_DIR = OUT / 'prepared_years'
NHTSA_STATIC = 'https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip'
YEARS = list(range(2016, 2025))
SEEDS_FULL = [42, 1337, 2026, 31415, 271828]
PER_TYP_DRIVER = 1
INJ_FATAL = 4
INJ_DIED_PRIOR = 6
ALC_TEST_GIVEN = 2
ALC_RES_POSITIVE_NO_VALUE = 998
PERSON_CONCEPTS = {'AGE': ['AGE'], 'SEX': ['SEX']}
ACCIDENT_CONCEPTS = {'MONTH': ['MONTH'], 'DAY_WEEK': ['DAY_WEEK', 'DAY_OF_WEEK'], 'HOUR': ['HOUR'], 'WEATHER': ['WEATHER', 'WEATHER1'], 'LIGHT_COND': ['LGT_COND', 'LIGHT_COND', 'ILLUMINATION'], 'RURAL_URBAN': ['RUR_URB', 'URBAN_RURAL'], 'REL_ROAD': ['REL_ROAD', 'RELATION_TO_ROAD'], 'TYP_INT': ['TYP_INT', 'INTERSECT_TYPE', 'INTERSECTION_TYPE'], 'FUNC_SYS': ['FUNC_SYS'], 'ROUTE': ['ROUTE'], 'WRK_ZONE': ['WRK_ZONE', 'WORK_ZONE_IND']}
VEHICLE_CORE_CONCEPTS = {'BODY_TYP': ['BODY_TYP', 'BODY_TYPE'], 'TRAV_SP': ['TRAV_SP', 'TRAVEL_SP', 'TRAVEL_SPD'], 'SPEEDREL': ['SPEEDREL', 'SPEED_REL'], 'VEH_MAN': ['VEH_MAN', 'VEH_MOVEMENT', 'MAN_COLL'], 'TRAV_DIR': ['TRAV_DIR', 'TRAVEL_DIRECTION']}
VEHICLE_EXPANDED_CONCEPTS = {'ROLLOVER': ['ROLLOVER'], 'IMPACT': ['IMPACT1', 'IMPACT_POINT', 'PRIN_IMP_PT'], 'TOWED': ['TOW_VEH', 'TOWED', 'TOW_IND'], 'DEFORMED': ['DEFORMED', 'DAMAGE_IND'], 'UNDERRIDE': ['UNDER_RIDE', 'UNDER_RIDE_IND']}
NUMERIC_CONCEPTS = {'AGE', 'MONTH', 'DAY_WEEK', 'HOUR', 'TRAV_SP'}

def log(msg: str) -> None:
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def find_member(zf: zipfile.ZipFile, basename: str) -> str:
    target = basename.lower()
    hits = [n for n in zf.namelist() if Path(n).name.lower() == target]
    if not hits:
        raise FileNotFoundError(f'{basename} not found in ZIP')
    return hits[0]

def read_csv_zip(zf: zipfile.ZipFile, basename: str) -> pd.DataFrame:
    member = find_member(zf, basename)
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with dest.open('wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    log(f'Saved {dest.name}')

def first_existing(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None

def resolve_key(df: pd.DataFrame, aliases: list[str], what: str) -> str:
    c = first_existing(df, aliases)
    if c is None:
        raise KeyError(f'Required {what} key not found. Tried {aliases}')
    return c

def extract_concepts(df: pd.DataFrame, concepts: dict[str, list[str]], prefix: str='') -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for concept, aliases in concepts.items():
        c = first_existing(df, aliases)
        if c is not None:
            out[prefix + concept] = df[c]
    return out

def bac_gt0_state(alc_status: pd.Series, alc_res: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    status = pd.to_numeric(alc_status, errors='coerce')
    res = pd.to_numeric(alc_res, errors='coerce')
    A = (status == ALC_TEST_GIVEN).astype(int)
    numeric = (A == 1) & res.between(0, 940, inclusive='both')
    positive_unspecified = (A == 1) & (res == ALC_RES_POSITIVE_NO_VALUE)
    R = (numeric | positive_unspecified).astype(int)
    Z = (numeric & (res > 0) | positive_unspecified).astype(int)
    RZ = (R * Z).astype(int)
    return (A, R, RZ)

def prepare_year(year: int, force: bool=False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f'CRSS_ML_driver_{year}.pkl'
    if cache.exists() and (not force):
        return pd.read_pickle(cache)
    zpath = ZIP_DIR / f'CRSS{year}CSV.zip'
    download(NHTSA_STATIC.format(year=year), zpath)
    log(f'Preparing CRSS {year}')
    with zipfile.ZipFile(zpath) as zf:
        person = read_csv_zip(zf, 'person.csv')
        accident = read_csv_zip(zf, 'accident.csv')
        vehicle = read_csv_zip(zf, 'vehicle.csv')
    p_case = resolve_key(person, ['CASENUM', 'ST_CASE'], 'person case')
    p_veh = resolve_key(person, ['VEH_NO', 'VEHICLE_NO', 'UNIT_NUM'], 'person vehicle')
    per_typ = resolve_key(person, ['PER_TYP', 'PERSON_TYPE'], 'person type')
    inj = resolve_key(person, ['INJ_SEV', 'INJ_SEVERITY'], 'injury severity')
    alc_status = resolve_key(person, ['ALC_STATUS'], 'alcohol status')
    alc_res = resolve_key(person, ['ALC_RES'], 'alcohol result')
    person = person.copy()
    person[per_typ] = pd.to_numeric(person[per_typ], errors='coerce')
    person[inj] = pd.to_numeric(person[inj], errors='coerce')
    person = person[(person[per_typ] == PER_TYP_DRIVER) & (person[inj] != INJ_DIED_PRIOR)].copy()
    person = person[person[inj].isin([0, 1, 2, 3, 4])].copy()
    base = pd.DataFrame({'CASE_ID': str(year) + '_' + person[p_case].astype(str), 'VEH_ID': person[p_veh].astype(str), 'YEAR': year, 'Y_FATAL': (person[inj] == INJ_FATAL).astype(int)}, index=person.index)
    pcon = extract_concepts(person, PERSON_CONCEPTS, 'P_')
    A, R, RZ = bac_gt0_state(person[alc_status], person[alc_res])
    base['A_TEST'] = A.values
    base['R_RESOLVED'] = R.values
    base['RZ_POSITIVE'] = RZ.values
    if 'WEIGHT' in person.columns:
        base['SAMPLE_WEIGHT'] = pd.to_numeric(person['WEIGHT'], errors='coerce').values
    else:
        base['SAMPLE_WEIGHT'] = np.nan
    a_case = resolve_key(accident, ['CASENUM', 'ST_CASE'], 'accident case')
    acon = extract_concepts(accident, ACCIDENT_CONCEPTS, 'A_')
    acon['CASE_ID'] = (str(year) + '_' + accident[a_case].astype(str)).values
    if 'WEIGHT' in accident.columns:
        acon['ACC_WEIGHT'] = pd.to_numeric(accident['WEIGHT'], errors='coerce').values
    acon = acon.drop_duplicates('CASE_ID')
    v_case = resolve_key(vehicle, ['CASENUM', 'ST_CASE'], 'vehicle case')
    v_veh = resolve_key(vehicle, ['VEH_NO', 'VEHICLE_NO', 'UNIT_NUM'], 'vehicle number')
    vcore = extract_concepts(vehicle, VEHICLE_CORE_CONCEPTS, 'V_')
    vexp = extract_concepts(vehicle, VEHICLE_EXPANDED_CONCEPTS, 'VX_')
    vdf = pd.concat([vcore, vexp], axis=1)
    vdf['CASE_ID'] = (str(year) + '_' + vehicle[v_case].astype(str)).values
    vdf['VEH_ID'] = vehicle[v_veh].astype(str).values
    vdf = vdf.drop_duplicates(['CASE_ID', 'VEH_ID'])
    out = pd.concat([base.reset_index(drop=True), pcon.reset_index(drop=True)], axis=1)
    out = out.merge(acon, on='CASE_ID', how='left', validate='many_to_one')
    out = out.merge(vdf, on=['CASE_ID', 'VEH_ID'], how='left', validate='many_to_one')
    if out['SAMPLE_WEIGHT'].isna().all() and 'ACC_WEIGHT' in out.columns:
        out['SAMPLE_WEIGHT'] = out['ACC_WEIGHT']
    elif 'ACC_WEIGHT' in out.columns:
        out['SAMPLE_WEIGHT'] = out['SAMPLE_WEIGHT'].fillna(out['ACC_WEIGHT'])
    out = out.drop(columns=[c for c in ['ACC_WEIGHT'] if c in out.columns])
    out['SAMPLE_WEIGHT'] = pd.to_numeric(out['SAMPLE_WEIGHT'], errors='coerce').fillna(1.0).clip(lower=1e-12)
    known = out['R_RESOLVED'] == 1
    q_test = np.average(out.loc[known, 'RZ_POSITIVE'], weights=out.loc[known, 'SAMPLE_WEIGHT']) if known.any() else np.nan
    out.attrs['q_test'] = float(q_test)
    out.to_pickle(cache)
    return out

def feature_sets(df: pd.DataFrame, tier: str, arm: str) -> tuple[list[str], list[str]]:
    core = [c for c in df.columns if (c.startswith('P_') or c.startswith('A_') or c.startswith('V_')) and c not in {'A_TEST'}]
    exp = core + [c for c in df.columns if c.startswith('VX_')]
    cols = core if tier == 'core' else exp
    if arm in ('M1', 'M2'):
        cols = cols + ['A_TEST']
    if arm == 'M2':
        cols = cols + ['R_RESOLVED', 'RZ_POSITIVE']
    cols = [c for c in cols if c in df.columns]
    numeric = [c for c in cols if c.split('_', 1)[-1] in NUMERIC_CONCEPTS or c in {'A_TEST', 'R_RESOLVED', 'RZ_POSITIVE'}]
    categorical = [c for c in cols if c not in numeric]
    return (numeric, categorical)

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', min_frequency=20, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', min_frequency=20, sparse=True)

def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', make_ohe())])
    return ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')

def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    m = np.nanmean(w[w > 0]) if np.any(w > 0) else 1.0
    return np.where(np.isfinite(w) & (w > 0), w / m, 1.0)

def fit_model(family: str, X_train: pd.DataFrame, y_train: np.ndarray, w_train: np.ndarray, num_cols: list[str], cat_cols: list[str], seed: int):
    if family in ('LR', 'XGB'):
        prep = make_preprocessor(num_cols, cat_cols)
        if family == 'LR':
            clf = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=seed)
        else:
            if not HAVE_XGB:
                raise ImportError('xgboost is not installed')
            clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, min_child_weight=5, reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_jobs=-1, random_state=seed)
        pipe = Pipeline([('prep', prep), ('model', clf)])
        pipe.fit(X_train, y_train, model__sample_weight=normalize_weights(w_train))
        return pipe
    if family == 'CAT':
        if not HAVE_CAT:
            raise ImportError('catboost is not installed')
        cols = num_cols + cat_cols
        X = X_train[cols].copy()
        medians = {}
        for c in num_cols:
            medians[c] = pd.to_numeric(X[c], errors='coerce').median()
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(medians[c])
        for c in cat_cols:
            X[c] = X[c].astype('string').fillna('__MISSING__')
        cat_idx = [X.columns.get_loc(c) for c in cat_cols]
        model = CatBoostClassifier(iterations=500, depth=7, learning_rate=0.05, loss_function='Logloss', random_seed=seed, verbose=False, allow_writing_files=False, thread_count=-1)
        model.fit(X, y_train, sample_weight=normalize_weights(w_train), cat_features=cat_idx)
        return (model, cols, num_cols, cat_cols, medians)
    raise ValueError(family)

def predict_model(model, family: str, X: pd.DataFrame) -> np.ndarray:
    if family in ('LR', 'XGB'):
        return model.predict_proba(X)[:, 1]
    m, cols, num_cols, cat_cols, medians = model
    X2 = X[cols].copy()
    for c in num_cols:
        X2[c] = pd.to_numeric(X2[c], errors='coerce').fillna(medians.get(c, 0.0))
    for c in cat_cols:
        X2[c] = X2[c].astype('string').fillna('__MISSING__')
    return m.predict_proba(X2)[:, 1]

def weighted_ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = np.sum(w)
    if total <= 0:
        return np.nan
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])
        if not np.any(m):
            continue
        wm = w[m]
        obs = np.average(y[m], weights=wm)
        pred = np.average(p[m], weights=wm)
        ece += wm.sum() / total * abs(obs - pred)
    return float(ece)

def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-08, 1 - 1e-08)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    out = {'brier': brier_score_loss(y, p, sample_weight=w), 'pr_auc': average_precision_score(y, p, sample_weight=w), 'log_loss': log_loss(y, p, sample_weight=w, labels=[0, 1]), 'ece10': weighted_ece(y, p, w, 10)}
    try:
        out['roc_auc'] = roc_auc_score(y, p, sample_weight=w)
    except Exception:
        out['roc_auc'] = np.nan
    return out

def split_cases(df: pd.DataFrame, seed: int, test_size: float=0.2, val_size: float=0.1):
    case = df.groupby('CASE_ID', as_index=False)['Y_FATAL'].max().rename(columns={'Y_FATAL': 'CASE_FATAL'})
    trainval, test = train_test_split(case, test_size=test_size, random_state=seed, stratify=case['CASE_FATAL'] if case['CASE_FATAL'].nunique() > 1 else None)
    val_frac_of_trainval = val_size / (1 - test_size)
    train, val = train_test_split(trainval, test_size=val_frac_of_trainval, random_state=seed + 1000, stratify=trainval['CASE_FATAL'] if trainval['CASE_FATAL'].nunique() > 1 else None)
    return (set(train.CASE_ID), set(val.CASE_ID), set(test.CASE_ID))

def evaluate_fit(train_df: pd.DataFrame, eval_frames: dict[str, pd.DataFrame], tier: str, arm: str, family: str, seed: int, results: list[dict], predictions_dir: Path, experiment: str):
    num_cols, cat_cols = feature_sets(train_df, tier, arm)
    feat_cols = num_cols + cat_cols
    if len(feat_cols) == 0:
        raise RuntimeError(f'No features resolved for {tier}/{arm}')
    ytr = train_df['Y_FATAL'].to_numpy(int)
    wtr = train_df['SAMPLE_WEIGHT'].to_numpy(float)
    model = fit_model(family, train_df[feat_cols], ytr, wtr, num_cols, cat_cols, seed)
    for eval_name, edf in eval_frames.items():
        p = predict_model(model, family, edf[feat_cols])
        met = metrics(edf['Y_FATAL'].to_numpy(int), p, edf['SAMPLE_WEIGHT'].to_numpy(float))
        row = {'experiment': experiment, 'eval_set': eval_name, 'tier': tier, 'arm': arm, 'family': family, 'seed': seed, 'n_train': len(train_df), 'n_eval': len(edf), 'n_features_raw': len(feat_cols), **met}
        results.append(row)
        pred = pd.DataFrame({'CASE_ID': edf['CASE_ID'].astype(str).values, 'YEAR': edf['YEAR'].values, 'Y_FATAL': edf['Y_FATAL'].values, 'SAMPLE_WEIGHT': edf['SAMPLE_WEIGHT'].values, 'PRED': p, 'experiment': experiment, 'eval_set': eval_name, 'tier': tier, 'arm': arm, 'family': family, 'seed': seed})
        fn = f'pred_{experiment}_{eval_name}_{tier}_{arm}_{family}_seed{seed}.csv.gz'.replace('/', '_')
        pred.to_csv(predictions_dir / fn, index=False, compression='gzip')

def run_pooled_premise(all_df: pd.DataFrame, seeds: list[int], families: list[str], results: list[dict], pred_dir: Path):
    for seed in seeds:
        tr_cases, va_cases, te_cases = split_cases(all_df, seed)
        train = all_df[all_df.CASE_ID.isin(tr_cases)].copy()
        test = all_df[all_df.CASE_ID.isin(te_cases)].copy()
        for tier in ['core', 'expanded']:
            for arm in ['M0', 'M1', 'M2']:
                for fam in families:
                    log(f'Premise seed={seed} tier={tier} arm={arm} family={fam}')
                    evaluate_fit(train, {'pooled_test': test}, tier, arm, fam, seed, results, pred_dir, 'premise')

def run_temporal(all_df: pd.DataFrame, seeds: list[int], families: list[str], results: list[dict], pred_dir: Path):
    source = all_df[all_df.YEAR <= 2019].copy()
    forward = {str(y): all_df[all_df.YEAR == y].copy() for y in range(2020, 2025)}
    for seed in seeds:
        tr_cases, va_cases, te_cases = split_cases(source, seed)
        train = source[source.CASE_ID.isin(tr_cases)].copy()
        source_test = source[source.CASE_ID.isin(te_cases)].copy()
        eval_frames = {'source_test': source_test, **{f'year_{y}': df for y, df in [(int(k), v) for k, v in forward.items()]}}
        for tier in ['core', 'expanded']:
            for arm in ['M0', 'M1', 'M2']:
                for fam in families:
                    log(f'Temporal seed={seed} tier={tier} arm={arm} family={fam}')
                    evaluate_fit(train, eval_frames, tier, arm, fam, seed, results, pred_dir, 'temporal')

def run_annual(frames: dict[int, pd.DataFrame], seeds: list[int], families: list[str], results: list[dict], pred_dir: Path):
    for year, df in frames.items():
        for seed in seeds:
            tr_cases, va_cases, te_cases = split_cases(df, seed)
            train = df[df.CASE_ID.isin(tr_cases)].copy()
            test = df[df.CASE_ID.isin(te_cases)].copy()
            for arm in ['M0', 'M2']:
                for fam in families:
                    log(f'Annual {year} seed={seed} arm={arm} family={fam}')
                    evaluate_fit(train, {f'year_{year}_test': test}, 'core', arm, fam, seed, results, pred_dir, 'annual')

def summarize_deltas(metrics_df: pd.DataFrame, annual_q: pd.DataFrame) -> None:
    keys = ['experiment', 'eval_set', 'tier', 'family', 'seed']
    wide = metrics_df.pivot_table(index=keys, columns='arm', values=['brier', 'pr_auc', 'roc_auc', 'log_loss', 'ece10'], aggfunc='first')
    rows = []
    for idx, rr in wide.iterrows():
        meta = dict(zip(keys, idx if isinstance(idx, tuple) else (idx,)))
        for comp in ['M1', 'M2']:
            if ('brier', comp) not in rr.index or ('brier', 'M0') not in rr.index:
                continue
            rows.append({**meta, 'comparison': f'{comp}-M0', 'delta_brier_gain': rr['brier', 'M0'] - rr['brier', comp], 'delta_pr_auc': rr['pr_auc', comp] - rr['pr_auc', 'M0'], 'delta_roc_auc': rr['roc_auc', comp] - rr['roc_auc', 'M0'], 'delta_logloss_gain': rr['log_loss', 'M0'] - rr['log_loss', comp], 'delta_ece_gain': rr['ece10', 'M0'] - rr['ece10', comp]})
    delta = pd.DataFrame(rows)
    delta.to_csv(OUT / 'paired_arm_deltas.csv', index=False)
    if not delta.empty:
        group = [c for c in ['experiment', 'eval_set', 'tier', 'family', 'comparison'] if c in delta.columns]
        summary = delta.groupby(group).agg(n_seeds=('seed', 'nunique'), mean_delta_brier_gain=('delta_brier_gain', 'mean'), sd_delta_brier_gain=('delta_brier_gain', 'std'), mean_delta_pr_auc=('delta_pr_auc', 'mean'), sd_delta_pr_auc=('delta_pr_auc', 'std')).reset_index()
        summary.to_csv(OUT / 'paired_arm_delta_seed_summary.csv', index=False)
    annual = delta[(delta.experiment == 'annual') & (delta.comparison == 'M2-M0')].copy() if not delta.empty else pd.DataFrame()
    if not annual.empty:
        annual['year'] = annual['eval_set'].str.extract('year_(\\d+)_test', expand=False).astype(int)
        annual = annual.merge(annual_q, on='year', how='left')
        annual.to_csv(OUT / 'annual_apparent_value_vs_qtest.csv', index=False)
        annsum = annual.groupby(['year', 'family']).agg(q_test=('q_test', 'first'), mean_delta_brier_gain=('delta_brier_gain', 'mean'), mean_delta_pr_auc=('delta_pr_auc', 'mean'), sd_delta_brier_gain=('delta_brier_gain', 'std'), sd_delta_pr_auc=('delta_pr_auc', 'std')).reset_index()
        annsum.to_csv(OUT / 'annual_apparent_value_vs_qtest_summary.csv', index=False)
        corr_rows = []
        from scipy.stats import spearmanr
        for fam, g in annsum.groupby('family'):
            if len(g) >= 3:
                rb = spearmanr(g.q_test, g.mean_delta_brier_gain)
                rp = spearmanr(g.q_test, g.mean_delta_pr_auc)
                corr_rows.append({'family': fam, 'rho_qtest_brier_gain': rb.statistic, 'p_qtest_brier_gain': rb.pvalue, 'rho_qtest_pr_gain': rp.statistic, 'p_qtest_pr_gain': rp.pvalue, 'n_years': len(g)})
        pd.DataFrame(corr_rows).to_csv(OUT / 'annual_qtest_spearman_descriptive.csv', index=False)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='One seed, Logistic Regression only; skips annual experiment.')
    ap.add_argument('--force-prepare', action='store_true', help='Rebuild cached prepared year parquet files.')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pred_dir = OUT / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    seeds = [42] if args.quick else SEEDS_FULL
    families = ['LR'] if args.quick else ['LR'] + (['XGB'] if HAVE_XGB else []) + (['CAT'] if HAVE_CAT else [])
    if not args.quick and len(families) < 3:
        log(f'WARNING: available families={families}; install xgboost/catboost if missing.')
    frames = {}
    qrows = []
    schema_rows = []
    for year in YEARS:
        df = prepare_year(year, force=args.force_prepare)
        frames[year] = df
        known = df.R_RESOLVED == 1
        q = np.average(df.loc[known, 'RZ_POSITIVE'], weights=df.loc[known, 'SAMPLE_WEIGHT']) if known.any() else np.nan
        qrows.append({'year': year, 'q_test': q, 'n_drivers': len(df), 'fatal_raw': int(df.Y_FATAL.sum()), 'test_rate_weighted': np.average(df.A_TEST, weights=df.SAMPLE_WEIGHT), 'resolved_rate_weighted': np.average(df.R_RESOLVED, weights=df.SAMPLE_WEIGHT)})
        schema_rows.append({'year': year, 'columns': '|'.join(df.columns)})
    annual_q = pd.DataFrame(qrows)
    annual_q.to_csv(OUT / 'annual_prepared_support.csv', index=False)
    pd.DataFrame(schema_rows).to_csv(OUT / 'prepared_schema_by_year.csv', index=False)
    all_df = pd.concat(frames.values(), ignore_index=True, sort=False)
    feature_candidates = [c for c in all_df.columns if c.startswith(('P_', 'A_', 'V_', 'VX_')) and c not in {'A_TEST', 'R_RESOLVED', 'RZ_POSITIVE'}]
    present_by_year = {c: all((c in frames[y].columns for y in YEARS)) for c in feature_candidates}
    common = [c for c, ok in present_by_year.items() if ok]
    keep = ['CASE_ID', 'VEH_ID', 'YEAR', 'Y_FATAL', 'SAMPLE_WEIGHT', 'A_TEST', 'R_RESOLVED', 'RZ_POSITIVE'] + common
    all_df = all_df[keep].copy()
    frames = {y: all_df[all_df.YEAR == y].copy() for y in YEARS}
    (OUT / 'common_feature_list.json').write_text(json.dumps(common, indent=2), encoding='utf-8')
    results = []
    run_pooled_premise(all_df, seeds, families, results, pred_dir)
    run_temporal(all_df, seeds, families, results, pred_dir)
    if not args.quick:
        run_annual(frames, seeds, families, results, pred_dir)
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUT / 'all_model_metrics.csv', index=False)
    summarize_deltas(metrics_df, annual_q)
    notes = ['CRSS ML analysis notes', '======================', 'A_TEST = ALC_STATUS==2.', 'R_RESOLVED for BAC>0 = numeric BAC 0..940 or code 998 positive-unspecified.', 'RZ_POSITIVE = resolved BAC>0.', 'Outcome = exact INJ_SEV 0-4 driver records only; fatal is code 4.', 'CRSS sample weights are used in fitting and point metrics.', 'The script reports seed variability, not design-based CRSS survey confidence intervals.', 'M1-M0 is conditional predictive information, NOT proof of a causal Y->A effect.', f'Model families available/run: {families}', f'Seeds: {seeds}']
    (OUT / 'analysis_notes.txt').write_text('\n'.join(notes), encoding='utf-8')
    log(f'DONE. Outputs saved to {OUT}')
if __name__ == '__main__':
    main()
