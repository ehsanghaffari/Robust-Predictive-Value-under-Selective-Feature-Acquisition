#!/usr/bin/env python3
"""Run the five-seed LightGBM M0/M1/M2 premise experiment."""
from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = [42, 1337, 2026, 31415, 271828]

BASE_CAT = [
    "YEAR", "SEX", "CRASH_MONTH", "DAY_OF_WEEK", "HOUR_OF_DAY",
    "ILLUMINATION", "RELATION_TO_ROAD", "ROAD_CONDITION", "WEATHER1",
    "INTERSECTION_RELATED", "INTERSECT_TYPE", "WORK_ZONE_IND", "URBAN_RURAL",
    "COLLISION_TYPE", "BODY_TYPE", "VEH_MOVEMENT", "UNIT_TYPE", "RDWY_ALIGNMENT",
]

BASE_NUM = ["AGE", "TRAVEL_SPD"]


def build_baseline_matrix(driver: pd.DataFrame):
    xcols = []
    catidx = []

    for col in BASE_CAT:
        vals = driver[col].astype("string").fillna("MISSING")
        xcols.append(pd.Categorical(vals).codes.astype(np.int16))
        catidx.append(len(xcols) - 1)

    for col in BASE_NUM:
        arr = pd.to_numeric(driver[col], errors="coerce").to_numpy(dtype=np.float32)
        if col == "AGE":
            arr[arr >= 99] = np.nan
        if col == "TRAVEL_SPD":
            arr[(arr < 0) | (arr > 150)] = np.nan
        xcols.append(arr)

    X0 = np.column_stack(xcols).astype(np.float32)
    return X0, catidx


def run_ml(driver: pd.DataFrame, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)

    X0, catidx = build_baseline_matrix(driver)
    R = driver["R_RESOLVED"].to_numpy(dtype=np.float32).reshape(-1, 1)
    ZS = driver["Z_STATE"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y = driver["Y_FATAL"].to_numpy(dtype=np.int8)
    crn = driver["CRN"].to_numpy()

    case = (
        driver[["CRN", "Y_FATAL"]]
        .groupby("CRN", sort=False)["Y_FATAL"]
        .max()
        .reset_index(name="fatal_case")
    )

    results = []

    for seed in SEEDS:
        tr_ids, temp_ids = train_test_split(
            case.CRN,
            test_size=0.30,
            random_state=seed,
            stratify=case.fatal_case,
        )

        temp = case.loc[case.CRN.isin(temp_ids)]
        va_ids, te_ids = train_test_split(
            temp.CRN,
            test_size=2 / 3,
            random_state=seed,
            stratify=temp.fatal_case,
        )

        tr = set(tr_ids.tolist())
        va = set(va_ids.tolist())
        te = set(te_ids.tolist())

        itr = np.fromiter((x in tr for x in crn), dtype=bool, count=len(crn))
        iva = np.fromiter((x in va for x in crn), dtype=bool, count=len(crn))
        ite = np.fromiter((x in te for x in crn), dtype=bool, count=len(crn))

        for arm in ("M0", "M1", "M2"):
            if arm == "M0":
                X = X0
                cats = catidx
            elif arm == "M1":
                X = np.hstack([X0, R])
                cats = catidx + [X0.shape[1]]
            else:
                X = np.hstack([X0, ZS])
                cats = catidx + [X0.shape[1]]

            model = lgb.LGBMClassifier(
                n_estimators=600,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=50,
                colsample_bytree=0.9,
                subsample=0.9,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )

            model.fit(
                X[itr],
                y[itr],
                categorical_feature=cats,
                eval_set=[(X[iva], y[iva])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            p = model.predict_proba(X[ite])[:, 1]

            results.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "pr_auc": average_precision_score(y[ite], p),
                    "roc_auc": roc_auc_score(y[ite], p),
                    "brier": brier_score_loss(y[ite], p),
                    "logloss": log_loss(y[ite], p),
                    "best_iter": model.best_iteration_,
                }
            )

            if arm != "M0":
                del X
            del model, p
            gc.collect()

    res = pd.DataFrame(results)
    res.to_csv(out / "ml_seed_results.csv", index=False)

    summary = (
        res.groupby("arm")[["pr_auc", "roc_auc", "brier", "logloss"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(out / "ml_summary.csv")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    agg = (
        res.groupby("arm")["pr_auc"]
        .agg(["mean", "std"])
        .reindex(["M0", "M1", "M2"])
    )
    x = np.arange(3)
    ax.bar(x, agg["mean"], yerr=agg["std"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["M0: baseline", "M1: + resolution", "M2: + 3-state condition"]
    )
    ax.set_ylabel("Fatal-injury PR-AUC")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig6_ml_premise.png", dpi=300)
    plt.close(fig)

    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Prepared .pkl.gz file from 01_PennDOT_Data_Preparation.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    driver = pd.read_pickle(args.input_file, compression="gzip")
    run_ml(driver, args.output_dir)

    print(f"Saved ML outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
