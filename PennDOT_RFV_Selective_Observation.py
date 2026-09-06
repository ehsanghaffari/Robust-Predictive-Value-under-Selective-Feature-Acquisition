#!/usr/bin/env python3
"""
Reproducibility analysis for:
Robust Predictive Value under Selective Feature Observation:
Exact Identification Boundaries with an Application to Police-Reported Driver Condition

Data source (not redistributed):
Pennsylvania Department of Transportation (PennDOT), Pennsylvania Crash
Information Tool (PCIT), Public Crash Databases:
https://crashinfo.penndot.pa.gov/

Expected inputs:
Statewide_2016.zip, ..., Statewide_2025.zip

The script reproduces the annual selective-resolution audit, exact interval-overlap
classification, minimax Brier RFV calculations, assumption-frontier quantities,
figures, and the five-seed LightGBM premise audit.
"""
from __future__ import annotations
import argparse
import gc
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
import lightgbm as lgb

warnings.filterwarnings("ignore")
YEARS = list(range(2016, 2026))
SEEDS = [42, 1337, 2026, 31415, 271828]

CRASH_COLS = [
    "CRN", "CRASH_MONTH", "DAY_OF_WEEK", "HOUR_OF_DAY", "ILLUMINATION",
    "RELATION_TO_ROAD", "ROAD_CONDITION", "WEATHER1", "INTERSECTION_RELATED",
    "INTERSECT_TYPE", "WORK_ZONE_IND", "URBAN_RURAL", "COLLISION_TYPE",
]
VEH_COLS = [
    "CRN", "UNIT_NUM", "BODY_TYPE", "TRAVEL_SPD", "VEH_MOVEMENT",
    "UNIT_TYPE", "RDWY_ALIGNMENT",
]
PERS_COLS = [
    "CRN", "UNIT_NUM", "PERSON_TYPE", "INJ_SEVERITY", "DVR_PED_CONDITION",
    "AGE", "SEX",
]
BASE_CAT = [
    "YEAR", "SEX", "CRASH_MONTH", "DAY_OF_WEEK", "HOUR_OF_DAY",
    "ILLUMINATION", "RELATION_TO_ROAD", "ROAD_CONDITION", "WEATHER1",
    "INTERSECTION_RELATED", "INTERSECT_TYPE", "WORK_ZONE_IND", "URBAN_RURAL",
    "COLLISION_TYPE", "BODY_TYPE", "VEH_MOVEMENT", "UNIT_TYPE", "RDWY_ALIGNMENT",
]
BASE_NUM = ["AGE", "TRAVEL_SPD"]


def read_driver_data(data_dir: Path) -> pd.DataFrame:
    parts = []
    for year in YEARS:
        zpath = data_dir / f"Statewide_{year}.zip"
        if not zpath.exists():
            raise FileNotFoundError(f"Missing required input: {zpath}")
        with zipfile.ZipFile(zpath) as z:
            person = pd.read_csv(z.open(f"PERSON_{year}.csv"), usecols=PERS_COLS, low_memory=False)
            person = person.loc[person["PERSON_TYPE"] == 1].copy()  # PennDOT code 1 = Driver
            crash = pd.read_csv(z.open(f"CRASH_{year}.csv"), usecols=CRASH_COLS, low_memory=False)
            vehicle = pd.read_csv(z.open(f"VEHICLE_{year}.csv"), usecols=VEH_COLS, low_memory=False)
        d = person.merge(vehicle, on=["CRN", "UNIT_NUM"], how="left", validate="many_to_one")
        d = d.merge(crash, on="CRN", how="left", validate="many_to_one")
        d["YEAR"] = year
        d["Y_FATAL"] = (d["INJ_SEVERITY"] == 1).astype(np.int8)  # PennDOT code 1 = Killed
        # DVR_PED_CONDITION: 0-6 are explicit conditions; 9 and missing are unresolved.
        d["R_RESOLVED"] = d["DVR_PED_CONDITION"].isin([0, 1, 2, 3, 4, 5, 6]).astype(np.int8)
        # PennDOT code 1 = Had Been Drinking.
        d["Z_DRINK"] = (d["DVR_PED_CONDITION"] == 1).astype(np.int8)
        d["Z_STATE"] = np.where(
            d["R_RESOLVED"] == 0,
            0,  # unresolved
            np.where(d["Z_DRINK"] == 1, 2, 1),  # 2=drinking, 1=resolved non-drinking
        ).astype(np.int8)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def worst_brier(p0: float, p1: float, pi: float, sL: float, sU: float, fL: float, fU: float) -> float:
    risks = []
    for s in (sL, sU):
        for f in (fL, fU):
            risk = (
                pi * f * (1.0 - p1) ** 2
                + (1.0 - pi) * s * p1**2
                + pi * (1.0 - f) * (1.0 - p0) ** 2
                + (1.0 - pi) * (1.0 - s) * p0**2
            )
            risks.append(risk)
    return float(max(risks))


def minimax_rfv(pi: float, sL: float, sU: float, fL: float, fU: float):
    deletion = pi * (1.0 - pi)
    objective = lambda x: worst_brier(x[0], x[1], pi, sL, sU, fL, fU)
    res = minimize(
        objective,
        x0=np.array([pi, pi]),
        method="Nelder-Mead",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 10000},
    )
    rfv = max(0.0, deletion - float(res.fun))
    return float(res.x[0]), float(res.x[1]), float(res.fun), float(rfv)


def annual_identification(driver: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        d = driver.loc[driver["YEAR"] == year]
        n = len(d)
        fatal_n = int(d["Y_FATAL"].sum())
        row = {"year": year, "n_drivers": n, "fatal_n": fatal_n, "pi": fatal_n / n}
        for y, lab in ((0, "s"), (1, "f")):
            g = d.loc[d["Y_FATAL"] == y]
            a = float(g["R_RESOLVED"].mean())
            q = float(g.loc[g["R_RESOLVED"] == 1, "Z_DRINK"].mean())
            row[f"a_{lab}"] = a
            row[f"q_{lab}"] = q
        row["sL"] = row["a_s"] * row["q_s"]
        row["sU"] = row["a_s"] * row["q_s"] + (1.0 - row["a_s"])
        row["fL"] = row["a_f"] * row["q_f"]
        row["fU"] = row["a_f"] * row["q_f"] + (1.0 - row["a_f"])
        row["margin"] = row["fL"] - row["sU"]
        row["overlap"] = not (row["fL"] > row["sU"] or row["sL"] > row["fU"])
        p0, p1, wr, rfv = minimax_rfv(row["pi"], row["sL"], row["sU"], row["fL"], row["fU"])
        row.update({"p0_natural": p0, "p1_natural": p1, "worst_natural": wr, "RFV_natural": rfv})
        p0, p1, wr, rfv = minimax_rfv(0.5, row["sL"], row["sU"], row["fL"], row["fU"])
        row.update({"p0_balanced": p0, "p1_balanced": p1, "worst_balanced": wr, "RFV_balanced": rfv})
        row["gamma_crit_phi0"] = (row["fL"] - row["a_s"] * row["q_s"]) / (1.0 - row["a_s"])
        row["phi_crit_gamma1"] = (row["sU"] - row["fL"]) / (1.0 - row["a_f"])
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(a: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(a.year, a.a_s * 100, marker="o", label="Surviving drivers")
    ax.plot(a.year, a.a_f * 100, marker="s", label="Fatal drivers")
    ax.set(xlabel="Crash year", ylabel="Driver-condition resolved (%)", ylim=(0, 100))
    ax.set_xticks(a.year); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig1_resolution_rates.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(a.year, a.q_s * 100, marker="o", label="Surviving drivers")
    ax.plot(a.year, a.q_f * 100, marker="s", label="Fatal drivers")
    ax.set(xlabel="Crash year", ylabel='"Had Been Drinking" among resolved (%)')
    ax.set_xticks(a.year); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig2_resolved_drinking.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(a))
    for i, r in a.iterrows():
        ax.plot([i - .12, i - .12], [r.sL, r.sU], marker="_", markersize=10, linewidth=2)
        ax.plot([i + .12, i + .12], [r.fL, r.fU], marker="_", markersize=10, linewidth=2)
    ax.scatter(x - .12, a.sL, s=12, label="Survivor interval")
    ax.scatter(x + .12, a.fL, s=12, label="Fatal interval")
    ax.set_xticks(x); ax.set_xticklabels(a.year, rotation=45)
    ax.set(xlabel="Crash year", ylabel="Admissible prevalence of the drinking-state feature")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig3_prevalence_intervals.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0, linewidth=1); ax.plot(a.year, a.margin, marker="o")
    for y, v in zip(a.year, a.margin):
        ax.annotate(f"{v:+.3f}", (y, v), xytext=(0, 7 if v >= 0 else -13), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel="Crash year", ylabel="Separation margin f_L - s_U")
    ax.set_xticks(a.year); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "fig4_boundary_margin.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    phis = np.linspace(0, .05, 101)
    for year in [2021, 2023, 2025]:
        r = a.loc[a.year == year].iloc[0]
        gamma = (r.a_f * r.q_f + (1 - r.a_f) * phis - r.a_s * r.q_s) / (1 - r.a_s)
        ax.plot(phis, gamma, label=str(year))
    ax.axhline(1, linestyle="--", linewidth=1, label="Unrestricted survivor upper bound")
    ax.set(xlabel="Lower bound phi_L among unresolved fatal drivers", ylabel="Critical survivor upper bound gamma_crit", xlim=(0, .05), ylim=(.75, 1.6))
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig5_assumption_frontier.png", dpi=300); plt.close(fig)


def run_ml(driver: pd.DataFrame, out: Path) -> pd.DataFrame:
    xcols, catidx = [], []
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
    del xcols
    R = driver["R_RESOLVED"].to_numpy(dtype=np.float32).reshape(-1, 1)
    ZS = driver["Z_STATE"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y = driver["Y_FATAL"].to_numpy(dtype=np.int8)
    crn = driver["CRN"].to_numpy()
    case = driver[["CRN", "Y_FATAL"]].groupby("CRN", sort=False)["Y_FATAL"].max().reset_index(name="fatal_case")
    results = []
    for seed in SEEDS:
        tr_ids, temp_ids = train_test_split(case.CRN, test_size=.30, random_state=seed, stratify=case.fatal_case)
        temp = case.loc[case.CRN.isin(temp_ids)]
        va_ids, te_ids = train_test_split(temp.CRN, test_size=2/3, random_state=seed, stratify=temp.fatal_case)
        tr, va, te = set(tr_ids.tolist()), set(va_ids.tolist()), set(te_ids.tolist())
        itr = np.fromiter((x in tr for x in crn), dtype=bool, count=len(crn))
        iva = np.fromiter((x in va for x in crn), dtype=bool, count=len(crn))
        ite = np.fromiter((x in te for x in crn), dtype=bool, count=len(crn))
        for arm in ("M0", "M1", "M2"):
            if arm == "M0":
                X, cats = X0, catidx
            elif arm == "M1":
                X, cats = np.hstack([X0, R]), catidx + [X0.shape[1]]
            else:
                X, cats = np.hstack([X0, ZS]), catidx + [X0.shape[1]]
            model = lgb.LGBMClassifier(
                n_estimators=600, learning_rate=.04, num_leaves=31, min_child_samples=50,
                colsample_bytree=.9, subsample=.9, reg_lambda=1.0, random_state=seed,
                n_jobs=-1, verbosity=-1,
            )
            model.fit(
                X[itr], y[itr], categorical_feature=cats, eval_set=[(X[iva], y[iva])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            p = model.predict_proba(X[ite])[:, 1]
            results.append({
                "seed": seed, "arm": arm,
                "pr_auc": average_precision_score(y[ite], p),
                "roc_auc": roc_auc_score(y[ite], p),
                "brier": brier_score_loss(y[ite], p),
                "logloss": log_loss(y[ite], p),
                "best_iter": model.best_iteration_,
            })
            if arm != "M0":
                del X
            del model, p
            gc.collect()
    res = pd.DataFrame(results)
    res.to_csv(out / "ml_seed_results.csv", index=False)
    summary = res.groupby("arm")[["pr_auc", "roc_auc", "brier", "logloss"]].agg(["mean", "std"])
    summary.to_csv(out / "ml_summary.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    agg = res.groupby("arm")["pr_auc"].agg(["mean", "std"]).reindex(["M0", "M1", "M2"])
    x = np.arange(3)
    ax.bar(x, agg["mean"], yerr=agg["std"], capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(["M0: baseline", "M1: + resolution", "M2: + 3-state condition"])
    ax.set_ylabel("Fatal-injury PR-AUC"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "fig6_ml_premise.png", dpi=300); plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory containing Statewide_2016.zip ... Statewide_2025.zip")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--skip-ml", action="store_true", help="Run only the deterministic identification/RFV analysis")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    driver = read_driver_data(args.data_dir)
    annual = annual_identification(driver)
    annual.to_csv(args.output_dir / "annual_identification.csv", index=False)
    make_figures(annual, args.output_dir)
    if not args.skip_ml:
        run_ml(driver, args.output_dir)
    print(f"Complete. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
