#!/usr/bin/env python
"""
MAS bounds, boundary distances, robust value and external-information frontier
==============================================================================

Reads the identification gate CSV (1_identification_gate) and computes the
deterministic quantities of the analysis without training any model:

  * per-year alpha_s, alpha_f, q_s, q_f for BAC > 0
  * MAS intervals for Gamma = 2, 4, 8, 22
  * positive (f_L - s_U) and reverse (s_L - f_U) boundary margins
  * Brier risk intervals for the natural and 50/50 populations
  * zero-RFV classification by interval overlap
  * the joint external-information frontier c_crit(phi_L)
  * OracleGain(gamma_true, phi_true) surfaces and curves
  * a fatal/survivor accounting audit from the gate masses
  * robust feature value at selected external assumptions (direct minimax)

Usage
-----
    python bounds_frontier_oracle.py
    python bounds_frontier_oracle.py --gate-csv path/to/crss_identification_gate.csv --output-dir out/

No ML training occurs.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from scipy.optimize import minimize
except Exception as exc:
    raise ImportError("scipy is required for the selected external-assumption RFV optimization") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_CSV = REPO_ROOT / "outputs" / "1_identification_gate" / "crss_identification_gate.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "2_bounds_and_frontier"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

PRIMARY_GAMMA = 4.0
GAMMAS = [2.0, 4.0, 8.0, 22.0]
PHI_L_GRID = np.array([0.00, 0.05, 0.10, 0.20, 0.30], dtype=float)
ORACLE_GAMMA_GRID = np.linspace(0.0, 0.25, 251)
ORACLE_PHI_GRID = np.linspace(0.0, 1.0, 201)
SELECTED_PHI_L = [0.00, 0.05, 0.10, 0.20, 0.30]
SELECTED_GAMMA_CAP = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
FIG_DPI = 600                       # JDSA minimum for combination artwork


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def find_input(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Gate CSV not found: {path}\n"
        "Run 1_identification_gate/crss_identification_gate.py --all-years first, "
        "or pass --gate-csv."
    )


def safe_div(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def oracle_gain(pi: float, s: float, f: float) -> float:
    """Bayes Brier improvement from fully observed binary Z."""
    pi = float(pi); s = float(s); f = float(f)
    r_delete = pi * (1.0 - pi)
    m1 = pi * f + (1.0 - pi) * s
    m0 = 1.0 - m1
    risk = 0.0
    if m1 > 0:
        r1 = pi * f / m1
        risk += m1 * r1 * (1.0 - r1)
    if m0 > 0:
        r0 = pi * (1.0 - f) / m0
        risk += m0 * r0 * (1.0 - r0)
    return float(max(0.0, r_delete - risk))


def brier_risk(pi: float, s: float, f: float, p0: float, p1: float) -> float:
    m1 = pi * f + (1 - pi) * s
    m0 = 1.0 - m1
    y1z1 = pi * f
    y1z0 = pi * (1.0 - f)
    return float(m1 * p1**2 + m0 * p0**2 + pi - 2.0 * (y1z1 * p1 + y1z0 * p0))


def minimax_rfv(pi: float, sL: float, sU: float, fL: float, fU: float) -> dict:
    """Minimize max Brier risk over a rectangular (s,f) set.

    When the prevalence intervals overlap, the robust value is exactly zero,
    so RFV=0 is returned analytically rather than asking a numerical optimizer
    to solve a nonsmooth equality case.
    """
    r_delete = pi * (1.0 - pi)
    overlap = not (fL > sU or fU < sL)
    if overlap:
        return {
            "pi": pi, "p0_star": pi, "p1_star": pi,
            "R_delete": r_delete, "R_robust": r_delete, "RFV": 0.0,
            "optimizer_success": True, "solution_method": "INTERVAL_OVERLAP",
        }

    corners = [(sL, fL), (sL, fU), (sU, fL), (sU, fU)]

    def objective(x):
        p0, p1 = float(x[0]), float(x[1])
        return max(brier_risk(pi, s, f, p0, p1) for s, f in corners)

    starts = [
        np.array([pi, pi]),
        np.array([max(0, pi - 0.10), min(1, pi + 0.10)]),
        np.array([min(1, pi + 0.10), max(0, pi - 0.10)]),
        np.array([0.25, 0.75]),
        np.array([0.75, 0.25]),
    ]
    best = None
    for x0 in starts:
        res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=[(0, 1), (0, 1)], options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    p0, p1 = map(float, best.x)
    r_robust = float(best.fun)
    rfv = max(0.0, r_delete - r_robust)
    if abs(rfv) < 1e-10:
        rfv = 0.0
    return {
        "pi": pi,
        "p0_star": p0,
        "p1_star": p1,
        "R_delete": r_delete,
        "R_robust": r_robust,
        "RFV": rfv,
        "optimizer_success": bool(best.success),
        "solution_method": "NUMERICAL_MINIMAX",
    }


def build_year_parameters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.sort_values("year").iterrows():
        alpha_s = float(r["gt0_survivor_Z_resolved_share_all_survivors"])
        alpha_f = float(r["gt0_fatal_Z_resolved_share_all_fatal"])
        q_s = float(r["q_test_BAC_gt0_weighted"])
        a1 = float(r["gt0_a1_weighted_mass"])
        a0 = float(r["gt0_a0_weighted_mass"])
        q_f = safe_div(a1, a1 + a0)
        fatal_w = float(r["fatal_driver_weighted_count"])
        surv_w = float(r["survivor_driver_weighted_count"])
        pi_nat = safe_div(fatal_w, fatal_w + surv_w)
        rows.append({
            "year": int(r["year"]),
            "alpha_s": alpha_s,
            "alpha_f": alpha_f,
            "q_s": q_s,
            "q_f": q_f,
            "pi_natural": pi_nat,
            "pi_balanced": 0.5,
            "survivor_test_rate": float(r["survivor_test_rate_weighted_all_survivors"]),
            "fatal_test_rate": float(r["fatal_test_rate_weighted_all_fatal"]),
            "v_rate_direct": float(r["v_rate_weighted_1_minus_fatal_test_rate"]),
        })
    return pd.DataFrame(rows)


def bounds_for(row: pd.Series, Gamma: float, phi_L: float = 0.0, phi_U: float = 1.0, gamma_cap: float | None = None) -> dict:
    a_s, a_f, q_s, q_f = row["alpha_s"], row["alpha_f"], row["q_s"], row["q_f"]
    gamma_L = q_s / Gamma
    gamma_U = q_s if gamma_cap is None else min(q_s, gamma_cap)
    sL = a_s * q_s + (1 - a_s) * gamma_L
    sU = a_s * q_s + (1 - a_s) * gamma_U
    fL = a_f * q_f + (1 - a_f) * phi_L
    fU = a_f * q_f + (1 - a_f) * phi_U
    return {"gamma_L": gamma_L, "gamma_U": gamma_U, "s_L": sL, "s_U": sU, "f_L": fL, "f_U": fU}


def risk_interval(pi: float, sL: float, sU: float, fL: float, fU: float) -> tuple[float, float, float, float]:
    L1 = safe_div(pi * fL, pi * fL + (1 - pi) * sU)
    U1 = safe_div(pi * fU, pi * fU + (1 - pi) * sL)
    L0 = safe_div(pi * (1 - fU), pi * (1 - fU) + (1 - pi) * (1 - sL))
    U0 = safe_div(pi * (1 - fL), pi * (1 - fL) + (1 - pi) * (1 - sU))
    return L1, U1, L0, U0


def make_accounting_audit(raw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in raw.sort_values("year").iterrows():
        year = int(r["year"])
        for zname in ["gt0", "ge008"]:
            fatal = {
                "resolved_Z1": float(r[f"{zname}_a1_weighted_mass"]),
                "resolved_Z0": float(r[f"{zname}_a0_weighted_mass"]),
                "tested_Z_unresolved": float(r[f"{zname}_fatal_tested_Z_unresolved_weighted_mass"]),
                "explicit_A0": float(r[f"{zname}_fatal_A0_weighted_mass"]),
                "unknown_A": float(r[f"{zname}_fatal_A_unknown_weighted_mass"]),
            }
            survivor = {
                "resolved_Z1": float(r[f"{zname}_b1_weighted_mass"]),
                "resolved_Z0": float(r[f"{zname}_b0_weighted_mass"]),
                "tested_Z_unresolved": float(r[f"{zname}_survivor_tested_Z_unresolved_weighted_mass"]),
                "explicit_A0": float(r[f"{zname}_u_optimistic_weighted_mass"]),
                "unknown_A": float(r[f"{zname}_u_conservative_weighted_mass"] - r[f"{zname}_u_optimistic_weighted_mass"]),
            }
            for outcome, d in [("fatal", fatal), ("survivor", survivor)]:
                total = sum(d.values())
                for cat, mass in d.items():
                    out.append({
                        "year": year,
                        "z_definition": zname,
                        "outcome": outcome,
                        "category": cat,
                        "population_mass": mass,
                        "conditional_share_within_outcome": safe_div(mass, total),
                    })
                out.append({
                    "year": year,
                    "z_definition": zname,
                    "outcome": outcome,
                    "category": "OUTCOME_PARTITION_SUM",
                    "population_mass": total,
                    "conditional_share_within_outcome": 1.0,
                })
            out.append({
                "year": year,
                "z_definition": zname,
                "outcome": "all",
                "category": "FULL_PARTITION_RESIDUAL_FROM_V0_6",
                "population_mass": float(r[f"{zname}_full_partition_mass_residual"]),
                "conditional_share_within_outcome": np.nan,
            })
    return pd.DataFrame(out)


def plot_boundary(boundary_df: pd.DataFrame, outpath: Path) -> None:
    """Fig. 1: survivor ceiling s_U and fatal floor f_L (phi_L = 0) by year, with the
    separation margin d_sep = f_L - s_U labelled next to each year's survivor point."""
    d = boundary_df.sort_values("year")
    yrs = d["year"].to_numpy(int)
    fL, sU = d["f_L"].to_numpy(float), d["s_U"].to_numpy(float)
    navy, rust = "#1f3a6e", "#a63603"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9.645, 5.295))
    ax.fill_between(yrs, fL, sU, color="#d9d9d9", lw=0, label=r"Gap $f_L < s_U$: no positive separation", zorder=0)
    for x, lo, hi in zip(yrs, fL, sU):                       # the gap measured by d_sep
        ax.plot([x, x], [lo, hi], color="#8c8c8c", lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.plot(yrs, sU, color=navy, lw=2.4, marker="o", ms=7, zorder=3,
            label=r"$s_U = q_s$  (survivor ceiling)")
    ax.plot(yrs, fL, color=rust, lw=2.4, marker="s", ms=7, zorder=3,
            label=r"$f_L = \alpha_f q_f$  (fatal floor, $\varphi_L = 0$)")
    for i, (x, lo, hi) in enumerate(zip(yrs, fL, sU)):
        # label just under the survivor point, on the side away from a steep neighbouring segment
        right_drop = hi - sU[i + 1] if i + 1 < len(sU) else 0.0
        left_drop = hi - sU[i - 1] if i > 0 else 0.0
        last = i == len(sU) - 1
        side = -1 if (last or right_drop > max(left_drop, 0.15)) else 1   # keep labels inside the band
        ax.annotate(f"{lo - hi:.2f}".replace("-", "−"), xy=(x, hi), xytext=(side * 7, -9),
                    textcoords="offset points", ha="left" if side > 0 else "right", va="top",
                    fontsize=10, color="#262626", zorder=4)
    ax.set_xlim(yrs.min() - 0.4, yrs.max() + 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(yrs)
    ax.set_xlabel("CRSS data year")
    ax.set_ylabel("Admissible alcohol prevalence")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_testing_rates(params: pd.DataFrame, outpath: Path) -> None:
    """Fig. 2: survey-weighted alcohol testing rates of fatal and surviving drivers (log scale)."""
    plt.rcdefaults()
    plt.rcParams["font.family"] = "DejaVu Sans"
    d = params.sort_values("year")
    fig, ax = plt.subplots(figsize=(8.0, 5.21))
    ax.plot(d["year"], 100 * d["fatal_test_rate"], marker="o", ms=8, lw=2, label="Fatal drivers")
    ax.plot(d["year"], 100 * d["survivor_test_rate"], marker="s", ms=8, lw=2, label="Surviving drivers")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 40])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(1, 50)
    ax.set_xticks(d["year"])
    ax.set_xlabel("CRSS data year")
    ax.set_ylabel("Alcohol test rate (%, log scale)")
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_survivor_positivity(params: pd.DataFrame, outpath: Path, negatives: tuple[int, int] | None = None) -> None:
    """Fig. 3: BAC > 0 positivity q_s among tested, resolved surviving drivers.
    negatives: raw tested-negative survivor counts in 2019 and 2020, shown in the annotation."""
    plt.rcdefaults()
    plt.rcParams["font.family"] = "DejaVu Sans"
    d = params.sort_values("year")
    fig, ax = plt.subplots(figsize=(8.0, 5.20))
    ax.axvspan(2019.6, 2020.4, color="C0", alpha=0.12, lw=0)
    ax.plot(d["year"], 100 * d["q_s"], marker="o", ms=8, lw=2, color="C0")
    y20 = 100 * float(d.loc[d["year"] == 2020, "q_s"].iloc[0])
    text = "testing broadens" + (f";\ntested negatives: {negatives[0]} to {negatives[1]}" if negatives else "")
    ax.annotate(text, xy=(2020, y20), xytext=(2021.2, y20 + 20), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.set_xticks(d["year"])
    ax.set_xlabel("CRSS data year")
    ax.set_ylabel("BAC > 0 among tested survivors (%)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_frontiers(frontier_all: pd.DataFrame, outpath: Path) -> None:
    """Fig. 8: joint assumption frontier c_crit(phi_L), envelope across years and the extreme years."""
    plt.rcdefaults()
    plt.rcParams["font.family"] = "DejaVu Sans"
    wide = frontier_all.pivot(index="phi_L", columns="year", values="c_crit_gamma_cap").sort_index()
    at0 = wide.iloc[0]
    hi, lo = int(at0.idxmax()), int(at0.idxmin())
    fig, ax = plt.subplots(figsize=(8.0, 5.04))
    ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1), color="C0", alpha=0.18, lw=0,
                    label=f"Frontier envelope across {wide.columns.min()}–{wide.columns.max()}")
    ax.plot(wide.index, wide[hi], color="C0", lw=2, label=rf"{hi} (least restrictive at $\varphi_L = 0$)")
    ax.plot(wide.index, wide[lo], color="C1", lw=2, label=rf"{lo} (most restrictive at $\varphi_L = 0$)")
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.40)
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"Assumed lower bound $\varphi_L$ among unresolved fatal drivers")
    ax.set_ylabel(r"Critical survivor upper bound $\gamma_{cap}$ for separation")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_oracle_2024(params: pd.DataFrame, outpath: Path, year: int = 2024) -> None:
    """Fig. 4: OracleGain sensitivity for one year on the standardized 50/50 population."""
    plt.rcdefaults()
    plt.rcParams["font.family"] = "DejaVu Sans"
    r = params.set_index("year").loc[year]
    fig, ax = plt.subplots(figsize=(8.0, 5.02))
    for phi_true in [0.05, 0.10, 0.25, 0.50]:
        f = r["alpha_f"] * r["q_f"] + (1 - r["alpha_f"]) * phi_true
        vals = [oracle_gain(0.5, r["alpha_s"] * r["q_s"] + (1 - r["alpha_s"]) * g, f) for g in ORACLE_GAMMA_GRID]
        ax.plot(ORACLE_GAMMA_GRID, vals, lw=2, label=rf"assumed $\varphi_{{true}}$ = {phi_true:.2f}")
    ax.axhline(0.010, linestyle="--", linewidth=1.0, color="C0", label="0.010 reporting floor")
    ax.set_xlabel(r"Assumed $\gamma_{true}$ among unresolved survivors")
    ax.set_ylabel("OracleGain (absolute Brier reduction)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_oracle_curves(params: pd.DataFrame, outdir: Path) -> None:
    """OracleGain sensitivity curves for every year and both populations (supplementary diagnostics)."""
    for _, r in params.iterrows():
        year = int(r["year"])
        for pop_name, pi in [("natural", r["pi_natural"]), ("balanced", 0.5)]:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for phi_true in [0.05, 0.10, 0.25, 0.50]:
                vals = []
                for gamma_true in ORACLE_GAMMA_GRID:
                    s = r["alpha_s"] * r["q_s"] + (1 - r["alpha_s"]) * gamma_true
                    f = r["alpha_f"] * r["q_f"] + (1 - r["alpha_f"]) * phi_true
                    vals.append(oracle_gain(pi, s, f))
                ax.plot(ORACLE_GAMMA_GRID, vals, label=rf"$\varphi_{{true}}$ = {phi_true:.2f}")
            ax.axhline(0.010, linestyle="--", linewidth=1.0, label="0.010 reporting floor")
            ax.set_xlabel(r"assumed $\gamma_{true}$ among unresolved survivors")
            ax.set_ylabel("OracleGain (absolute Brier reduction)")
            ax.set_title(f"OracleGain sensitivity, {year}, {pop_name} prevalence")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"OracleGain_{year}_{pop_name}.png", dpi=220)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-csv", type=Path, default=DEFAULT_GATE_CSV,
                    help="Gate CSV from 1_identification_gate (default: outputs/1_identification_gate/crss_identification_gate.csv)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Output directory (default: outputs/2_bounds_and_frontier)")
    return ap.parse_args()


def main() -> None:
    global OUTPUT_DIR
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_csv = find_input(args.gate_csv)
    log(f"Reading {input_csv}")
    raw = pd.read_csv(input_csv)
    params = build_year_parameters(raw)
    params.to_csv(OUTPUT_DIR / "year_parameters_gt0.csv", index=False)

    bounds_rows = []
    risk_rows = []
    boundary_rows = []
    frontier_rows = []
    external_rfv_rows = []

    for _, r in params.iterrows():
        year = int(r["year"])
        for Gamma in GAMMAS:
            b = bounds_for(r, Gamma=Gamma, phi_L=0.0, phi_U=1.0)
            overlap = not (b["f_L"] > b["s_U"] or b["f_U"] < b["s_L"])
            bounds_rows.append({"year": year, "Gamma": Gamma, "phi_L": 0.0, "phi_U": 1.0, **b,
                                "positive_margin_fL_minus_sU": b["f_L"] - b["s_U"],
                                "reverse_margin_sL_minus_fU": b["s_L"] - b["f_U"],
                                "interval_overlap": overlap,
                                "RFV_zero_by_overlap": overlap})
            for pop_name, pi in [("natural", r["pi_natural"]), ("balanced", 0.5)]:
                L1, U1, L0, U0 = risk_interval(pi, b["s_L"], b["s_U"], b["f_L"], b["f_U"])
                mm = minimax_rfv(pi, b["s_L"], b["s_U"], b["f_L"], b["f_U"])
                risk_rows.append({"year": year, "Gamma": Gamma, "population": pop_name, "pi": pi,
                                  "L1": L1, "U1": U1, "L0": L0, "U0": U0,
                                  "pi_in_Z1_interval": bool(L1 <= pi <= U1),
                                  "pi_in_Z0_interval": bool(L0 <= pi <= U0),
                                  **mm})

        b0 = bounds_for(r, Gamma=PRIMARY_GAMMA, phi_L=0.0, phi_U=1.0)
        boundary_rows.append({
            "year": year,
            "q_f": r["q_f"], "alpha_f": r["alpha_f"], "f_L": b0["f_L"],
            "s_U": b0["s_U"], "q_s": r["q_s"],
            "f_L_minus_s_U": b0["f_L"] - b0["s_U"],
            "s_L": b0["s_L"], "f_U": b0["f_U"],
            "s_L_minus_f_U": b0["s_L"] - b0["f_U"],
        })

        # Analytic joint assumption frontier across a dense phi_L grid.
        for phi_L in np.linspace(0, 0.35, 351):
            numerator = r["alpha_f"] * r["q_f"] + (1 - r["alpha_f"]) * phi_L - r["alpha_s"] * r["q_s"]
            ccrit = safe_div(numerator, 1 - r["alpha_s"])
            frontier_rows.append({"year": year, "phi_L": phi_L, "c_crit_gamma_cap": ccrit})

        # Selected external points: compute actual minimax RFV, not only separation status.
        for phi_L in SELECTED_PHI_L:
            for gamma_cap in SELECTED_GAMMA_CAP:
                b_ext = bounds_for(r, Gamma=PRIMARY_GAMMA, phi_L=phi_L, phi_U=1.0, gamma_cap=gamma_cap)
                for pop_name, pi in [("natural", r["pi_natural"]), ("balanced", 0.5)]:
                    mm = minimax_rfv(pi, b_ext["s_L"], b_ext["s_U"], b_ext["f_L"], b_ext["f_U"])
                    external_rfv_rows.append({
                        "year": year, "population": pop_name, "Gamma": PRIMARY_GAMMA,
                        "phi_L": phi_L, "gamma_cap": gamma_cap,
                        **b_ext,
                        "positive_separation": b_ext["f_L"] > b_ext["s_U"],
                        **mm,
                    })

    bounds_df = pd.DataFrame(bounds_rows)
    risk_df = pd.DataFrame(risk_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    frontier_df = pd.DataFrame(frontier_rows)
    ext_df = pd.DataFrame(external_rfv_rows)

    bounds_df.to_csv(OUTPUT_DIR / "all_year_MAS_bounds.csv", index=False)
    risk_df.to_csv(OUTPUT_DIR / "all_year_risk_intervals_and_RFV.csv", index=False)
    boundary_df.to_csv(OUTPUT_DIR / "all_year_boundary_distances.csv", index=False)
    frontier_df.to_csv(OUTPUT_DIR / "joint_assumption_frontier_dense.csv", index=False)
    ext_df.to_csv(OUTPUT_DIR / "selected_external_assumption_RFV.csv", index=False)

    # c_crit table at selected phi_L values.
    ccrit_rows = []
    for _, r in params.iterrows():
        row = {"year": int(r["year"])}
        for phi_L in PHI_L_GRID:
            numerator = r["alpha_f"] * r["q_f"] + (1 - r["alpha_f"]) * phi_L - r["alpha_s"] * r["q_s"]
            row[f"phi_L_{phi_L:.2f}"] = safe_div(numerator, 1 - r["alpha_s"])
        ccrit_rows.append(row)
    pd.DataFrame(ccrit_rows).to_csv(OUTPUT_DIR / "joint_assumption_frontier_table.csv", index=False)

    accounting = make_accounting_audit(raw)
    accounting.to_csv(OUTPUT_DIR / "full_accounting_audit.csv", index=False)

    # OracleGain surface data for all years and both populations.
    oracle_rows = []
    for _, r in params.iterrows():
        year = int(r["year"])
        for pop_name, pi in [("natural", r["pi_natural"]), ("balanced", 0.5)]:
            for gamma_true in ORACLE_GAMMA_GRID:
                s = r["alpha_s"] * r["q_s"] + (1 - r["alpha_s"]) * gamma_true
                for phi_true in ORACLE_PHI_GRID:
                    f = r["alpha_f"] * r["q_f"] + (1 - r["alpha_f"]) * phi_true
                    oracle_rows.append({"year": year, "population": pop_name, "pi": pi,
                                        "gamma_true": gamma_true, "phi_true": phi_true,
                                        "s_true": s, "f_true": f,
                                        "OracleGain": oracle_gain(pi, s, f)})
    pd.DataFrame(oracle_rows).to_csv(OUTPUT_DIR / "OracleGain_surface_all_years.csv.gz", index=False, compression="gzip")

    plot_boundary(boundary_df, OUTPUT_DIR / "Fig1_boundary_distances.png")
    negs = raw.set_index("year")["gt0_b0_raw_n"] if "gt0_b0_raw_n" in raw.columns else None
    plot_testing_rates(params, OUTPUT_DIR / "Fig2_testing_rates.png")
    plot_survivor_positivity(params, OUTPUT_DIR / "Fig3_survivor_positivity.png",
                             None if negs is None else (int(negs.loc[2019]), int(negs.loc[2020])))
    plot_oracle_2024(params, OUTPUT_DIR / "Fig4_oraclegain_2024.png")
    plot_frontiers(frontier_df, OUTPUT_DIR / "Fig8_joint_assumption_frontier.png")
    plot_oracle_curves(params, OUTPUT_DIR)

    # Compact console summary.
    print("\n=== PRIMARY BROAD-SET BOUNDARY DISTANCES (Gamma=4, phi=[0,1]) ===")
    print(boundary_df[["year", "q_f", "alpha_f", "f_L", "s_U", "f_L_minus_s_U", "s_L_minus_f_U"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== RFV ZERO CHECK: Gamma=4 ===")
    chk = risk_df[risk_df["Gamma"] == PRIMARY_GAMMA][["year", "population", "pi", "L1", "U1", "L0", "U0", "RFV"]]
    print(chk.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print(f"\nSaved all outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
