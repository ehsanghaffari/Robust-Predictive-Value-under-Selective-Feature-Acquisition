#!/usr/bin/env python
"""
PennDOT administrative alcohol-flag diagnostic (supplementary)
==============================================================

A supplementary PennDOT experiment, not a replacement for the CRSS A/R/Z
selective-acquisition analysis. PennDOT Statewide files contain final
crash-level administrative flags such as ALCOHOL_RELATED and DRINKING_DRIVER,
but no CRSS-style test acquisition or result-resolution variables.

The experiment asks a narrower question at crash level:
  P0 = W
  P1 = W + ALCOHOL_RELATED
  P2 = W + ALCOHOL_RELATED + DRINKING_DRIVER

It quantifies how much final administrative alcohol flags add to retrospective
fatal-crash prediction and whether that apparent value is temporally unstable.
It is an administrative-flag shortcut diagnostic, not a causal
alcohol-test-acquisition analysis.

Data (not redistributed)
------------------------
--data-dir (default data/penndot) with Statewide_<year>/<TABLE>_<year>.csv for
2016-2025.

Outputs: --output-dir (default outputs/5_supplementary_audits/penndot_admin_flags).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "penndot"
OUT = REPO_ROOT / "outputs" / "5_supplementary_audits" / "penndot_admin_flags"
YEARS = list(range(2016, 2026))
SEEDS = [42, 1337, 2026, 31415, 271828]

# Deliberately exclude outcome-derived counts (FATAL_COUNT, injury counts,
# MAX_SEVERITY_LEVEL), response/transport times, geographic identifiers, and
# final severity flags.
CORE_FEATURES = [
    "COLLISION_TYPE", "CONS_ZONE_SPD_LIM", "CRASH_MONTH", "DAY_OF_WEEK",
    "HOUR_OF_DAY", "ILLUMINATION", "INTERSECTION_RELATED", "INTERSECT_TYPE",
    "LOCATION_TYPE", "RDWY_SURF_TYPE_CD", "RELATION_TO_ROAD", "ROAD_CONDITION",
    "SCH_BUS_IND", "SCH_ZONE_IND", "TCD_FUNC_CD", "TCD_TYPE", "URBAN_RURAL",
    "WEATHER1", "WEATHER2", "WORK_ZONE_IND", "WORK_ZONE_LOC", "WORK_ZONE_TYPE",
    "AUTOMOBILE_COUNT", "BICYCLE_COUNT", "BUS_COUNT", "COMM_VEH_COUNT",
    "HEAVY_TRUCK_COUNT", "MOTORCYCLE_COUNT", "PED_COUNT", "PERSON_COUNT",
    "TOTAL_UNITS", "VEHICLE_COUNT", "VULNERABLE_ROAD_USER_COUNT",
]
NUMERIC = {
    "CONS_ZONE_SPD_LIM", "HOUR_OF_DAY", "AUTOMOBILE_COUNT", "BICYCLE_COUNT",
    "BUS_COUNT", "COMM_VEH_COUNT", "HEAVY_TRUCK_COUNT", "MOTORCYCLE_COUNT",
    "PED_COUNT", "PERSON_COUNT", "TOTAL_UNITS", "VEHICLE_COUNT",
    "VULNERABLE_ROAD_USER_COUNT",
}


def log(msg: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def read_table(year: int, stem: str) -> pd.DataFrame:
    folder = DATASET / f"Statewide_{year}"
    candidates = [folder / f"{stem}_{year}.csv", folder / f"{stem}_{year}.CSV"]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            df.columns = [str(c).strip().upper() for c in df.columns]
            return df
    raise FileNotFoundError(f"Could not find {stem}_{year}.csv in {folder}")


def build_year(year: int) -> pd.DataFrame:
    crash = read_table(year, "CRASH")
    flags = read_table(year, "FLAGS")
    need = {"CRN", "FATAL_COUNT"}
    miss = need - set(crash.columns)
    if miss:
        raise KeyError(f"{year} CRASH missing {miss}")
    flag_need = {"CRN", "ALCOHOL_RELATED", "DRINKING_DRIVER"}
    missf = flag_need - set(flags.columns)
    if missf:
        raise KeyError(f"{year} FLAGS missing {missf}")
    cols = [c for c in CORE_FEATURES if c in crash.columns]
    out = crash[["CRN", "FATAL_COUNT"] + cols].copy()
    out["YEAR"] = year
    out["Y_FATAL_CRASH"] = (pd.to_numeric(out["FATAL_COUNT"], errors="coerce").fillna(0) > 0).astype(int)
    out = out.drop(columns=["FATAL_COUNT"])
    ff = flags[["CRN", "ALCOHOL_RELATED", "DRINKING_DRIVER"]].drop_duplicates("CRN").copy()
    for c in ["ALCOHOL_RELATED", "DRINKING_DRIVER"]:
        ff[c] = pd.to_numeric(ff[c], errors="coerce").fillna(0).astype(int)
    out = out.merge(ff, on="CRN", how="left", validate="one_to_one")
    out[["ALCOHOL_RELATED", "DRINKING_DRIVER"]] = out[["ALCOHOL_RELATED", "DRINKING_DRIVER"]].fillna(0)
    return out


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse=True)


def feature_cols(df: pd.DataFrame, arm: str):
    core = [c for c in CORE_FEATURES if c in df.columns]
    if arm in ("P1", "P2"):
        core += ["ALCOHOL_RELATED"]
    if arm == "P2":
        core += ["DRINKING_DRIVER"]
    num = [c for c in core if c in NUMERIC or c in {"ALCOHOL_RELATED", "DRINKING_DRIVER"}]
    cat = [c for c in core if c not in num]
    return num, cat


def prep(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]), cat),
    ])


def fit_predict(family, train, test, arm, seed):
    num, cat = feature_cols(train, arm)
    cols = num + cat
    if family in ("LR", "XGB"):
        if family == "LR":
            clf = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=seed)
        else:
            if not HAVE_XGB: raise ImportError("xgboost missing")
            clf = XGBClassifier(n_estimators=450, max_depth=6, learning_rate=.05, subsample=.85,
                                colsample_bytree=.85, tree_method="hist", eval_metric="logloss",
                                objective="binary:logistic", n_jobs=-1, random_state=seed)
        model = Pipeline([("prep", prep(num, cat)), ("model", clf)])
        model.fit(train[cols], train.Y_FATAL_CRASH)
        p = model.predict_proba(test[cols])[:, 1]
        return p, len(cols)
    if family == "CAT":
        if not HAVE_CAT: raise ImportError("catboost missing")
        X = train[cols].copy(); Xt = test[cols].copy()
        for c in num:
            med = pd.to_numeric(X[c], errors="coerce").median()
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)
            Xt[c] = pd.to_numeric(Xt[c], errors="coerce").fillna(med)
        for c in cat:
            X[c] = X[c].astype("string").fillna("__MISSING__")
            Xt[c] = Xt[c].astype("string").fillna("__MISSING__")
        idx = [X.columns.get_loc(c) for c in cat]
        m = CatBoostClassifier(iterations=450, depth=7, learning_rate=.05, loss_function="Logloss",
                               random_seed=seed, verbose=False, allow_writing_files=False)
        m.fit(X, train.Y_FATAL_CRASH, cat_features=idx)
        return m.predict_proba(Xt)[:, 1], len(cols)
    raise ValueError(family)


def metric_row(y, p):
    return {
        "brier": brier_score_loss(y, p),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-8, 1-1e-8)),
    }


def main():
    global DATASET, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="one seed, logistic regression only")
    ap.add_argument("--data-dir", type=Path, default=DATASET,
                    help="folder with Statewide_<year>/ sub-folders (default: data/penndot)")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    DATASET, OUT = args.data_dir, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [42] if args.quick else SEEDS
    families = ["LR"] if args.quick else ["LR"] + (["XGB"] if HAVE_XGB else []) + (["CAT"] if HAVE_CAT else [])

    frames = {y: build_year(y) for y in YEARS}
    all_df = pd.concat(frames.values(), ignore_index=True, sort=False)
    common = [c for c in CORE_FEATURES if all(c in frames[y].columns for y in YEARS)]
    keep = ["CRN", "YEAR", "Y_FATAL_CRASH", "ALCOHOL_RELATED", "DRINKING_DRIVER"] + common
    all_df = all_df[keep].copy()
    (OUT / "common_core_features.json").write_text(json.dumps(common, indent=2), encoding="utf-8")

    rows = []
    # Pooled retrospective premise diagnostic.
    for seed in seeds:
        tr, te = train_test_split(all_df, test_size=.20, random_state=seed, stratify=all_df.Y_FATAL_CRASH)
        for arm in ["P0", "P1", "P2"]:
            for fam in families:
                log(f"Pooled seed={seed} {arm} {fam}")
                p, nf = fit_predict(fam, tr, te, arm, seed)
                rows.append({"experiment":"pooled", "eval_set":"pooled_test", "seed":seed, "arm":arm, "family":fam,
                             "n_train":len(tr), "n_eval":len(te), "n_features":nf, **metric_row(te.Y_FATAL_CRASH.to_numpy(), p)})

    # Temporal source 2016-2019 -> 2020-2025.
    source = all_df[all_df.YEAR <= 2019].copy()
    for seed in seeds:
        tr, src_test = train_test_split(source, test_size=.20, random_state=seed, stratify=source.Y_FATAL_CRASH)
        evals = {"source_test": src_test, **{f"year_{y}": all_df[all_df.YEAR == y].copy() for y in range(2020, 2026)}}
        for arm in ["P0", "P1", "P2"]:
            for fam in families:
                # Fit once, then evaluate unchanged on all forward years.
                num, cat = feature_cols(tr, arm); cols = num + cat
                if fam in ("LR", "XGB"):
                    clf = LogisticRegression(C=1.0,max_iter=1000,solver="liblinear",random_state=seed) if fam=="LR" else XGBClassifier(n_estimators=450,max_depth=6,learning_rate=.05,subsample=.85,colsample_bytree=.85,tree_method="hist",eval_metric="logloss",objective="binary:logistic",n_jobs=-1,random_state=seed)
                    model = Pipeline([("prep", prep(num,cat)),("model",clf)]); model.fit(tr[cols],tr.Y_FATAL_CRASH)
                    pred_fn = lambda d: model.predict_proba(d[cols])[:,1]
                else:
                    X=tr[cols].copy(); meds={}
                    for c in num:
                        meds[c]=pd.to_numeric(X[c],errors="coerce").median(); X[c]=pd.to_numeric(X[c],errors="coerce").fillna(meds[c])
                    for c in cat: X[c]=X[c].astype("string").fillna("__MISSING__")
                    idx=[X.columns.get_loc(c) for c in cat]
                    model=CatBoostClassifier(iterations=450,depth=7,learning_rate=.05,loss_function="Logloss",random_seed=seed,verbose=False,allow_writing_files=False); model.fit(X,tr.Y_FATAL_CRASH,cat_features=idx)
                    def pred_fn(d):
                        Z=d[cols].copy()
                        for c in num: Z[c]=pd.to_numeric(Z[c],errors="coerce").fillna(meds[c])
                        for c in cat: Z[c]=Z[c].astype("string").fillna("__MISSING__")
                        return model.predict_proba(Z)[:,1]
                for name, edf in evals.items():
                    p=pred_fn(edf); rows.append({"experiment":"temporal","eval_set":name,"seed":seed,"arm":arm,"family":fam,"n_train":len(tr),"n_eval":len(edf),"n_features":len(cols),**metric_row(edf.Y_FATAL_CRASH.to_numpy(),p)})

    metrics=pd.DataFrame(rows); metrics.to_csv(OUT/"penndot_admin_flag_metrics.csv",index=False)
    # Paired deltas.
    keys=["experiment","eval_set","family","seed"]
    wide=metrics.pivot_table(index=keys,columns="arm",values=["brier","pr_auc","roc_auc","log_loss"],aggfunc="first")
    dr=[]
    for idx,r in wide.iterrows():
        meta=dict(zip(keys,idx))
        for arm in ["P1","P2"]:
            dr.append({**meta,"comparison":f"{arm}-P0","delta_brier_gain":r[("brier","P0")]-r[("brier",arm)],"delta_pr_auc":r[("pr_auc",arm)]-r[("pr_auc","P0")],"delta_roc_auc":r[("roc_auc",arm)]-r[("roc_auc","P0")]})
    pd.DataFrame(dr).to_csv(OUT/"penndot_admin_flag_paired_deltas.csv",index=False)

    note=(
        "This PennDOT analysis is supplementary only. ALCOHOL_RELATED and DRINKING_DRIVER are final crash-level administrative flags, not test-acquisition/result-resolution variables.\n"
        "P0/P1/P2 are not the CRSS M0/M1/M2 arms and should not be reported as such.\n"
        "2025 is exploratory because the CRSS analysis period is 2016-2024.\n"
    )
    (OUT/"SCOPE_WARNING.txt").write_text(note,encoding="utf-8")
    log(f"DONE: {OUT}")

if __name__=="__main__":
    main()
