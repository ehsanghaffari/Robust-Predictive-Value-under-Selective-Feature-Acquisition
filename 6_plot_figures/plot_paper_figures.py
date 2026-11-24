#!/usr/bin/env python3
"""Plot article Figures 2, 3, and 5–7 from saved CRSS analysis CSVs.

These are plotting/reconstruction scripts, not analyses. Required input results
are deliberately NOT shipped with the repository. Incomplete model/annual
coverage is rejected unless --allow-partial is explicitly passed; partial
charts are visibly labeled and never receive publication figure filenames.

Fig 2: survey-weighted survivor/fatal testing rates by year.
Fig 3: BAC>0 positivity among tested surviving drivers with a resolved Z state.
Fig 5: pooled PR-AUC by M0/M1/M2 and model family, seed SD bars, both W tiers.
Fig 6: M1-M0 PR-AUC across source and 2020–2024 holdouts; family/seed mean.
Fig 7: annual q_ML versus annual M2-M0 PR-AUC by model family.

Example: python 6_plot_figures/plot_paper_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "outputs/1_identification_gate/crss_identification_gate.csv"
ML = ROOT / "outputs/3_ml_premise/all_model_metrics.csv"
ANNUAL = ROOT / "outputs/3_ml_premise/annual_apparent_value_vs_qtest_summary.csv"
OUT = ROOT / "outputs/6_plot_figures"
YEARS = tuple(range(2016, 2025))
SEEDS = (42, 1337, 2026, 31415, 271828)
TIERS = ("core", "expanded")
FAMS = ("LR", "XGB", "CAT")
ARMS = ("M0", "M1", "M2")
FAM_NAMES = {"LR": "Logistic regression", "XGB": "XGBoost", "CAT": "CatBoost"}
FIGURES = {
    2: "Fig2_annual_testing_rates", 3: "Fig3_tested_survivor_positivity",
    5: "Fig5_pooled_PR_AUC", 6: "Fig6_temporal_acquisition_gain",
    7: "Fig7_annual_selectivity_vs_apparent_value",
}


def load(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis output absent: {path}\nRun the producer listed in the README first.")
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    if df.empty or missing:
        raise ValueError(f"Invalid file {path}: empty={df.empty}; missing columns={sorted(missing)}")
    return df


def assert_coverage(observed: set, expected: set, label: str, allow_partial: bool) -> bool:
    if observed == expected:
        return False
    msg = f"{label}: {len(expected - observed)} required records missing; {len(observed - expected)} unexpected."
    if not allow_partial:
        raise ValueError(msg + " Use --allow-partial only for draft diagnostics.")
    print("WARNING: " + msg, file=sys.stderr)
    return True


def save(fig: plt.Figure, fig_id: int, dest: Path, partial: bool, dpi: int) -> Path:
    if partial:
        fig.text(.5, .5, "DRAFT / INCOMPLETE DATA", ha="center", va="center", rotation=25,
                 fontsize=20, weight="bold", alpha=.17)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / (FIGURES[fig_id] + ("_PARTIAL" if partial else "") + ".png")
    fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def gate_data(path: Path, allow_partial: bool) -> tuple[pd.DataFrame, bool]:
    required = ["year", "survivor_test_rate_weighted_all_survivors",
                "fatal_test_rate_weighted_all_fatal", "q_test_BAC_gt0_weighted"]
    data = load(path, required)
    data["year"] = pd.to_numeric(data["year"], errors="raise").astype(int)
    if data.year.duplicated().any():
        raise ValueError("Duplicate years in identification-gate output.")
    partial = assert_coverage(set(data.year), set(YEARS), "2016–2024 gate years", allow_partial)
    for column in required[1:]:
        value = pd.to_numeric(data[column], errors="raise")
        if not value.between(0, 1).all():
            raise ValueError(f"{column} must be a fraction in [0,1], not percentage points")
        data[column] = value
    return data.sort_values("year"), partial


def plot2(data: pd.DataFrame, output: Path, partial: bool, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(data.year, 100 * data.survivor_test_rate_weighted_all_survivors,
            marker="o", label="Surviving drivers", linewidth=1.8)
    ax.plot(data.year, 100 * data.fatal_test_rate_weighted_all_fatal,
            marker="s", label="Fatal-injury drivers", linewidth=1.8)
    ax.set(xlabel="Year", ylabel="Alcohol testing rate (%)")
    ax.set_xticks(data.year)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.23)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, 2, output, partial, dpi)


def plot3(data: pd.DataFrame, output: Path, partial: bool, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(data.year, 100 * data.q_test_BAC_gt0_weighted, marker="o", linewidth=1.8)
    ax.axvline(2019.5, linestyle="--", alpha=.5, linewidth=1)
    ax.set(xlabel="Year", ylabel="BAC > 0 among tested, resolved surviving drivers (%)")
    ax.set_xticks(data.year)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=.23)
    fig.tight_layout()
    return save(fig, 3, output, partial, dpi)


def metrics_data(path: Path) -> pd.DataFrame:
    columns = ["experiment", "eval_set", "tier", "arm", "family", "seed", "pr_auc"]
    data = load(path, columns)
    data["seed"] = pd.to_numeric(data["seed"], errors="raise").astype(int)
    data["pr_auc"] = pd.to_numeric(data["pr_auc"], errors="raise")
    if not np.isfinite(data.pr_auc).all() or not data.pr_auc.between(0, 1).all():
        raise ValueError("ML CSV has invalid PR-AUC values")
    if data[columns[:-1]].duplicated().any():
        raise ValueError("ML CSV has duplicate experiment/eval/tier/arm/family/seed rows")
    return data


def metric_subset(data: pd.DataFrame, experiment: str, evals: list[str], arms: tuple[str, ...],
                  allow_partial: bool) -> tuple[pd.DataFrame, bool]:
    sub = data[data.experiment.eq(experiment) & data.eval_set.isin(evals) & data.arm.isin(arms)]
    if sub.empty:
        raise ValueError(f"No {experiment} ML rows found. Full ML run required.")
    observed = set(sub[["eval_set", "tier", "arm", "family", "seed"]].itertuples(index=False, name=None))
    expected = {(e, t, a, f, s) for e in evals for t in TIERS for a in arms for f in FAMS for s in SEEDS}
    partial = assert_coverage(observed, expected, f"{experiment} model coverage", allow_partial)
    return sub, partial


def plot5(data: pd.DataFrame, output: Path, partial: bool, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), sharey=True)
    for ax, tier in zip(axes, TIERS):
        for fam in FAMS:
            sub = data[data.tier.eq(tier) & data.family.eq(fam)]
            values = sub.groupby("arm").pr_auc.agg(["mean", "std"]).reindex(ARMS)
            if values["mean"].notna().any():
                ax.errorbar(range(3), values["mean"], yerr=values["std"].fillna(0),
                            marker="o", linewidth=1.6, capsize=3, label=FAM_NAMES[fam])
        ax.set_xticks(range(3), ARMS)
        ax.set(xlabel="Feature set", title=tier.capitalize() + " covariates")
        ax.grid(axis="y", alpha=.23)
    axes[0].set_ylabel("Fatal-injury PR-AUC (mean ± seed SD)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.03), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, .92])
    return save(fig, 5, output, partial, dpi)


def plot6(data: pd.DataFrame, output: Path, partial: bool, dpi: int) -> Path:
    # The paired difference MUST be formed within the same seed/model/tier/period.
    wide = data.pivot(index=["eval_set", "tier", "family", "seed"], columns="arm", values="pr_auc")
    wide = wide.reindex(columns=["M0", "M1"])
    if wide.isna().any(axis=None):
        if not partial:
            raise ValueError("Temporal rows lack an M0/M1 paired value")
        wide = wide.dropna(subset=["M0", "M1"])
    if wide.empty:
        raise ValueError("No complete temporal M0/M1 pairs to plot")
    gain = (wide.M1 - wide.M0).rename("gain").reset_index()
    periods = ["source_test"] + [f"year_{year}" for year in range(2020, 2025)]
    labels = ["2016–19\nsource"] + [str(year) for year in range(2020, 2025)]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for tier in TIERS:
        # Equal weight to each family and each seed, not survey-inference intervals.
        vals = gain[gain.tier.eq(tier)].groupby("eval_set").gain.mean().reindex(periods)
        ax.plot(range(len(periods)), vals, "o-", linewidth=1.7, label=tier.capitalize())
    ax.axhline(0, linewidth=.8, alpha=.7)
    ax.set_xticks(range(len(periods)), labels)
    ax.set(xlabel="Held-out regime", ylabel="M1 − M0 fatal-injury PR-AUC")
    ax.grid(axis="y", alpha=.23)
    ax.legend(title="Covariate tier", frameon=False)
    fig.tight_layout()
    return save(fig, 6, output, partial, dpi)


def plot7(path: Path, output: Path, allow_partial: bool, dpi: int) -> Path:
    data = load(path, ["year", "family", "q_test", "mean_delta_pr_auc"])
    data["year"] = pd.to_numeric(data["year"], errors="raise").astype(int)
    for col in ["q_test", "mean_delta_pr_auc"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"Nonfinite values for {col} in annual summary")
    if not data.q_test.between(0, 1).all():
        raise ValueError("q_test is expected as a fraction, not a percent")
    if data[["year", "family"]].duplicated().any():
        raise ValueError("Duplicate year/family rows in annual summary")
    observed = set(data[["year", "family"]].itertuples(index=False, name=None))
    expected = {(y, f) for y in YEARS for f in FAMS}
    partial = assert_coverage(observed, expected, "annual q_test coverage", allow_partial)
    # q_test is the aggregate ML resolved-driver statistic, NOT MAS q_s.
    if data.groupby("year").q_test.agg(lambda s: s.max()-s.min()).max() > 1e-9:
        raise ValueError("q_test differs across families for the same year")
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for fam in FAMS:
        sub = data[data.family.eq(fam)].sort_values("year")
        if sub.empty:
            continue
        ax.scatter(sub.q_test, sub.mean_delta_pr_auc, s=35, label=FAM_NAMES[fam])
        for point in sub.itertuples(index=False):
            ax.annotate(str(point.year)[-2:], (point.q_test, point.mean_delta_pr_auc),
                        xytext=(3, 2), textcoords="offset points", fontsize=7)
    ax.axhline(0, linewidth=.8, alpha=.7)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set(xlabel="BAC > 0 among all resolved ML drivers (q_ML)",
           ylabel="Annual held-out M2 − M0 fatal-injury PR-AUC")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, 7, output, partial, dpi)


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", nargs="+", type=int, choices=sorted(FIGURES), default=sorted(FIGURES))
    ap.add_argument("--gate-csv", type=Path, default=GATE)
    ap.add_argument("--ml-csv", type=Path, default=ML)
    ap.add_argument("--annual-csv", type=Path, default=ANNUAL)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--allow-partial", action="store_true", help="Draft ONLY: label incomplete figures")
    args = ap.parse_args(argv)
    wanted = set(args.figures)
    created: list[Path] = []
    if wanted & {2, 3}:
        gate, incomplete = gate_data(args.gate_csv, args.allow_partial)
        if 2 in wanted:
            created.append(plot2(gate, args.output_dir, incomplete, args.dpi))
        if 3 in wanted:
            created.append(plot3(gate, args.output_dir, incomplete, args.dpi))
    if wanted & {5, 6}:
        all_ml = metrics_data(args.ml_csv)
        if 5 in wanted:
            pooled, incomplete = metric_subset(all_ml, "premise", ["pooled_test"], ARMS, args.allow_partial)
            created.append(plot5(pooled, args.output_dir, incomplete, args.dpi))
        if 6 in wanted:
            temporal, incomplete = metric_subset(all_ml, "temporal", ["source_test"] +
                [f"year_{y}" for y in range(2020, 2025)], ("M0", "M1"), args.allow_partial)
            created.append(plot6(temporal, args.output_dir, incomplete, args.dpi))
    if 7 in wanted:
        created.append(plot7(args.annual_csv, args.output_dir, args.allow_partial, args.dpi))
    for path in created:
        print(f"Saved: {path}")
    return created


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
