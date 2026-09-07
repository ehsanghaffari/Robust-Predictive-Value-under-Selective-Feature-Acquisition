#!/usr/bin/env python3
"""Run interval identification, minimax RFV, assumption-frontier, and figure calculations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

YEARS = list(range(2016, 2026))


def worst_brier(
    p0: float,
    p1: float,
    pi: float,
    sL: float,
    sU: float,
    fL: float,
    fU: float,
) -> float:
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


def minimax_rfv(
    pi: float,
    sL: float,
    sU: float,
    fL: float,
    fU: float,
):
    deletion = pi * (1.0 - pi)

    def objective(x):
        return worst_brier(x[0], x[1], pi, sL, sU, fL, fU)

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

        row = {
            "year": year,
            "n_drivers": n,
            "fatal_n": fatal_n,
            "pi": fatal_n / n,
        }

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
        row["overlap"] = not (
            row["fL"] > row["sU"] or row["sL"] > row["fU"]
        )

        p0, p1, wr, rfv = minimax_rfv(
            row["pi"], row["sL"], row["sU"], row["fL"], row["fU"]
        )
        row.update(
            {
                "p0_natural": p0,
                "p1_natural": p1,
                "worst_natural": wr,
                "RFV_natural": rfv,
            }
        )

        p0, p1, wr, rfv = minimax_rfv(
            0.5, row["sL"], row["sU"], row["fL"], row["fU"]
        )
        row.update(
            {
                "p0_balanced": p0,
                "p1_balanced": p1,
                "worst_balanced": wr,
                "RFV_balanced": rfv,
            }
        )

        row["gamma_crit_phi0"] = (
            row["fL"] - row["a_s"] * row["q_s"]
        ) / (1.0 - row["a_s"])
        row["phi_crit_gamma1"] = (
            row["sU"] - row["fL"]
        ) / (1.0 - row["a_f"])

        rows.append(row)

    return pd.DataFrame(rows)


def make_figures(a: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(a.year, a.a_s * 100, marker="o", label="Surviving drivers")
    ax.plot(a.year, a.a_f * 100, marker="s", label="Fatal drivers")
    ax.set(
        xlabel="Crash year",
        ylabel="Driver-condition resolved (%)",
        ylim=(0, 100),
    )
    ax.set_xticks(a.year)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig1_resolution_rates.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(a.year, a.q_s * 100, marker="o", label="Surviving drivers")
    ax.plot(a.year, a.q_f * 100, marker="s", label="Fatal drivers")
    ax.set(
        xlabel="Crash year",
        ylabel='"Had Been Drinking" among resolved (%)',
    )
    ax.set_xticks(a.year)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_resolved_drinking.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(a))
    for i, r in a.iterrows():
        ax.plot(
            [i - 0.12, i - 0.12],
            [r.sL, r.sU],
            marker="_",
            markersize=10,
            linewidth=2,
        )
        ax.plot(
            [i + 0.12, i + 0.12],
            [r.fL, r.fU],
            marker="_",
            markersize=10,
            linewidth=2,
        )
    ax.scatter(x - 0.12, a.sL, s=12, label="Survivor interval")
    ax.scatter(x + 0.12, a.fL, s=12, label="Fatal interval")
    ax.set_xticks(x)
    ax.set_xticklabels(a.year, rotation=45)
    ax.set(
        xlabel="Crash year",
        ylabel="Admissible prevalence of the drinking-state feature",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig3_prevalence_intervals.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0, linewidth=1)
    ax.plot(a.year, a.margin, marker="o")
    for y, v in zip(a.year, a.margin):
        ax.annotate(
            f"{v:+.3f}",
            (y, v),
            xytext=(0, 7 if v >= 0 else -13),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set(
        xlabel="Crash year",
        ylabel="Separation margin f_L - s_U",
    )
    ax.set_xticks(a.year)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig4_boundary_margin.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    phis = np.linspace(0, 0.05, 101)
    for year in [2021, 2023, 2025]:
        r = a.loc[a.year == year].iloc[0]
        gamma = (
            r.a_f * r.q_f
            + (1.0 - r.a_f) * phis
            - r.a_s * r.q_s
        ) / (1.0 - r.a_s)
        ax.plot(phis, gamma, label=str(year))
    ax.axhline(
        1,
        linestyle="--",
        linewidth=1,
        label="Unrestricted survivor upper bound",
    )
    ax.set(
        xlabel="Lower bound phi_L among unresolved fatal drivers",
        ylabel="Critical survivor upper bound gamma_crit",
        xlim=(0, 0.05),
        ylim=(0.75, 1.6),
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig5_assumption_frontier.png", dpi=300)
    plt.close(fig)


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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    driver = pd.read_pickle(args.input_file, compression="gzip")

    annual = annual_identification(driver)
    annual.to_csv(args.output_dir / "annual_identification.csv", index=False)
    make_figures(annual, args.output_dir)

    print(f"Saved identification outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
