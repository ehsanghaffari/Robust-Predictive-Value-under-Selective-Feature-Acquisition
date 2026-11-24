#!/usr/bin/env python3
"""
Value identity, robust skill and analyses from the annual summary quantities
============================================================================

Analyses that need only the annual summary quantities (testing rates, resolved
shares, q_s, q_f and natural prevalence; manuscript Tables 2, 5 and 6):

  1. input check: reproduces the risk intervals (Table 5), boundary distances
     (Table 6) and c_crit(0) (Table 10);
  2. numerical verification of the value identity RFV = min_U OracleGain on
     random rectangles and convex polygons;
  3. 2024 OracleGain sensitivity points with MAS-compatibility labels and the
     scale-free skill criterion (Table 4);
  4. exact robust value beyond the separation boundary (Table 11) and the
     0.04 robust-skill frontier;
  5. decomposition of the MAS survivor ceiling into untested and
     tested-unresolved survivors;
  6. conditional versus marginal RFV on two small examples.

Inputs are the rounded annual values reported in the manuscript, so results
agree with the full pipeline (1_identification_gate, bounds_frontier_oracle.py,
4_design_based_and_attribution) up to rounding.

Usage
-----
    python value_identity_and_skill.py [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--output-dir", type=Path,
                 default=REPO_ROOT / "outputs" / "2_bounds_and_frontier" / "value_identity_and_skill",
                 help="output folder (default: outputs/2_bounds_and_frontier/value_identity_and_skill)")
OUT = _ap.parse_args().output_dir
OUT.mkdir(parents=True, exist_ok=True)
GAMMA = 4.0

# ---------------------------------------------------------------- inputs
# Table 2 (percent), Table 6 (q_f), Table 5 (natural pi)
rows = [
    # year, test_s, test_f, alpha_s, alpha_f, q_s,   q_f,   pi_nat
    (2016, 1.29, 31.89, 0.58, 8.31, 85.93, 0.578, 0.001850),
    (2017, 1.36, 28.63, 0.67, 11.31, 84.96, 0.612, 0.001888),
    (2018, 1.22, 26.40, 0.60, 5.69, 81.96, 0.518, 0.001868),
    (2019, 1.22, 27.43, 0.64, 6.72, 76.99, 0.580, 0.001603),
    (2020, 2.41, 31.46, 1.61, 13.18, 38.55, 0.503, 0.002466),
    (2021, 2.46, 31.13, 1.66, 8.93, 32.28, 0.544, 0.002545),
    (2022, 2.13, 29.94, 1.35, 8.59, 34.53, 0.519, 0.002523),
    (2023, 1.89, 35.00, 1.18, 11.02, 45.40, 0.373, 0.002403),
    (2024, 1.73, 33.66, 1.11, 11.92, 51.36, 0.412, 0.002182),
]
inp = pd.DataFrame(rows, columns=["year", "test_s", "test_f", "alpha_s", "alpha_f", "q_s", "q_f", "pi_nat"])
for c in ["test_s", "test_f", "alpha_s", "alpha_f", "q_s"]:
    inp[c] = inp[c] / 100.0
inp = inp.set_index("year")


# ---------------------------------------------------------------- core functions
def oracle_gain(s, f, pi):
    """Brier OracleGain = Var(P(Y=1|Z)) = pi^2(1-pi)^2(f-s)^2 / (M1 M0)."""
    s = np.asarray(s, float)
    f = np.asarray(f, float)
    m1 = pi * f + (1 - pi) * s
    m0 = 1 - m1
    with np.errstate(divide="ignore", invalid="ignore"):
        og = (pi * (1 - pi)) ** 2 * (f - s) ** 2 / (m1 * m0)
    return np.where((m1 <= 0) | (m0 <= 0), 0.0, og)


def bayes_risk(s, f, pi):
    return pi * (1 - pi) - oracle_gain(s, f, pi)


def rfv_rect(sL, sU, fL, fU, pi):
    """Exact RFV on a rectangle via the value identity (min OracleGain)."""
    if fL > sU:
        return float(oracle_gain(sU, fL, pi)), (sU, fL)
    if fU < sL:
        return float(oracle_gain(sL, fU, pi)), (sL, fU)
    return 0.0, None


def brier(p0, p1, s, f, pi):
    return (pi * f * (1 - p1) ** 2 + (1 - pi) * s * p1 ** 2
            + pi * (1 - f) * (1 - p0) ** 2 + (1 - pi) * (1 - s) * p0 ** 2)


def rfv_numeric(vertices, pi, starts=8, rng=None):
    """Direct numerical min over (p0,p1) of max over polytope vertices (risk is
    linear in the completion, so the inner max is attained at a vertex)."""
    rng = rng or np.random.default_rng(0)
    obj = lambda x: max(brier(x[0], x[1], s, f, pi) for s, f in vertices)
    best = np.inf
    x0s = [np.array([pi, pi])] + [rng.uniform(0, 1, 2) for _ in range(starts)]
    for x0 in x0s:
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"xatol": 1e-13, "fatol": 1e-16, "maxiter": 40000})
        best = min(best, r.fun)
    return pi * (1 - pi) - best


def mas_intervals(r, gamma=GAMMA, phiL=0.0, phiU=1.0):
    sL = r.q_s * (r.alpha_s + (1 - r.alpha_s) / gamma)
    sU = r.q_s
    fL = r.alpha_f * r.q_f + (1 - r.alpha_f) * phiL
    fU = r.alpha_f * r.q_f + (1 - r.alpha_f) * phiU
    return sL, sU, fL, fU


def alt_intervals(r, gcap, phiL):
    """External-information family of Sect. 3.8: gamma in [0,gcap], phi in [phiL,1]."""
    sL = r.alpha_s * r.q_s
    sU = r.alpha_s * r.q_s + (1 - r.alpha_s) * gcap
    fL = r.alpha_f * r.q_f + (1 - r.alpha_f) * phiL
    fU = r.alpha_f * r.q_f + (1 - r.alpha_f)
    return sL, sU, fL, fU


def risk_intervals(sL, sU, fL, fU, pi):
    L1 = pi * fL / (pi * fL + (1 - pi) * sU)
    U1 = pi * fU / (pi * fU + (1 - pi) * sL)
    L0 = pi * (1 - fU) / (pi * (1 - fU) + (1 - pi) * (1 - sL))
    U0 = pi * (1 - fL) / (pi * (1 - fL) + (1 - pi) * (1 - sU))
    return L1, U1, L0, U0


report = {}

# ---------------------------------------------------------------- 1. input check
risk_ms = {  # Table 5 reported values, natural/balanced [L1,U1,L0,U0]
    2016: [(0.000104, 0.008115, 0.000083, 0.012380), (0.052938, 0.815330, 0.042993, 0.871210)],
    2021: [(0.000384, 0.028085, 0.000114, 0.003572), (0.130815, 0.918865, 0.042624, 0.584174)],
    2024: [(0.000209, 0.015094, 0.000177, 0.004257), (0.087195, 0.875137, 0.074831, 0.661587)],
}
chk = []
for y, r in inp.iterrows():
    sL, sU, fL, fU = mas_intervals(r)
    for lab, pi in [("natural", r.pi_nat), ("balanced", 0.5)]:
        L1, U1, L0, U0 = risk_intervals(sL, sU, fL, fU, pi)
        rfv, _ = rfv_rect(sL, sU, fL, fU, pi)
        row = dict(year=y, pop=lab, pi=pi, L1=L1, U1=U1, L0=L0, U0=U0, RFV=rfv,
                   pi_in_both=(L1 <= pi <= U1) and (L0 <= pi <= U0))
        if y in risk_ms:
            ms = risk_ms[y][0 if lab == "natural" else 1]
            row["max_rel_dev_vs_reported"] = max(abs(a - b) / b for a, b in zip((L1, U1, L0, U0), ms))
        chk.append(row)
chk = pd.DataFrame(chk)
chk.to_csv(OUT / "check_risk_intervals.csv", index=False)
report["risk_intervals_max_rel_dev"] = float(chk["max_rel_dev_vs_reported"].max())
report["risk_intervals_all_rfv_zero"] = bool((chk.RFV == 0).all() and chk.pi_in_both.all())

t5 = []
for y, r in inp.iterrows():
    sL, sU, fL, fU = mas_intervals(r)
    t5.append(dict(year=y, f_L=fL, s_U=sU, d_sep=fL - sU, rev_margin=sL - fU,
                   c_crit_0=(r.alpha_f * r.q_f - r.alpha_s * r.q_s) / (1 - r.alpha_s)))
t5 = pd.DataFrame(t5).set_index("year")
t5.to_csv(OUT / "check_boundary_distances.csv")

# ---------------------------------------------------------------- 2. value identity check
rng = np.random.default_rng(20261123)
dev = []
for _ in range(400):
    pi = float(rng.choice([0.002, 0.05, 0.3, 0.5, 0.8]))
    a, b = np.sort(rng.uniform(0, 1, 2))
    c, d = np.sort(rng.uniform(0, 1, 2))
    verts = [(a, c), (a, d), (b, c), (b, d)]
    exact, _ = rfv_rect(a, b, c, d, pi)
    num = rfv_numeric(verts, pi, starts=4, rng=rng)
    dev.append(dict(pi=pi, exact=exact, numeric=num, absdiff=abs(exact - num),
                    reldiff=abs(exact - num) / max(pi * (1 - pi), 1e-12)))
# random convex polygons (not rectangles): compare numeric minimax with min OG
for _ in range(100):
    pi = float(rng.choice([0.05, 0.3, 0.5]))
    pts = rng.uniform(0, 1, (5, 2))
    num = rfv_numeric([tuple(p) for p in pts], pi, starts=4, rng=rng)
    # min OG over convex hull: minimize over barycentric weights
    def og_w(w):
        w = np.abs(w) / np.abs(w).sum()
        s, f = w @ pts
        return float(oracle_gain(s, f, pi))
    best = min(minimize(og_w, rng.uniform(0.1, 1, 5), method="Nelder-Mead",
                        options={"xatol": 1e-12, "fatol": 1e-16, "maxiter": 20000}).fun for _ in range(6))
    dev.append(dict(pi=pi, exact=best, numeric=num, absdiff=abs(best - num),
                    reldiff=abs(best - num) / (pi * (1 - pi)), polygon=True))
dev = pd.DataFrame(dev)
dev.to_csv(OUT / "check_value_identity.csv", index=False)
report["value_identity_max_reldiff_rect"] = float(dev[dev.get("polygon").isna()].reldiff.max())
report["value_identity_max_reldiff_polygon"] = float(dev[dev.get("polygon").fillna(False)].reldiff.max())

# also check the CRSS 2024 alternative-family point numerically
r24 = inp.loc[2024]
sL, sU, fL, fU = alt_intervals(r24, 0.02, 0.20)
ex, _ = rfv_rect(sL, sU, fL, fU, 0.5)
nu = rfv_numeric([(sL, fL), (sL, fU), (sU, fL), (sU, fU)], 0.5)
report["crss2024_alt_point_exact_vs_numeric"] = [ex, nu]

# ---------------------------------------------------------------- 3. 2024 OracleGain sensitivity points
r = inp.loc[2024]
sL24, sU24, _, _ = mas_intervals(r)
g_lo = r.q_s / GAMMA
t3 = []
grid = [(0.01, 0.25), (0.04, 0.25), (0.08, 0.25), (0.15, 0.25),
        (round(g_lo, 4), 0.25), (0.30, 0.25), (round(r.q_s, 4), 0.25),
        (0.15, 0.50), (0.30, 0.50), (round(r.q_s, 4), 1.00)]
for g, ph in grid:
    s = r.alpha_s * r.q_s + (1 - r.alpha_s) * g
    f = r.alpha_f * r.q_f + (1 - r.alpha_f) * ph
    ogb = float(oracle_gain(s, f, 0.5))
    ogn = float(oracle_gain(s, f, r.pi_nat))
    t3.append(dict(gamma_true=g, phi_true=ph, in_primary_MAS=(g_lo - 1e-9 <= g <= r.q_s + 1e-9),
                   Gamma_min=max(1.0, r.q_s / g), OG_5050=ogb, BSS_5050=ogb / 0.25,
                   floor_0010_5050=ogb >= 0.010, OG_nat=ogn, BSS_nat=ogn / (r.pi_nat * (1 - r.pi_nat)),
                   rel_criterion_5050=ogb / 0.25 >= 0.04,
                   rel_criterion_nat=ogn / (r.pi_nat * (1 - r.pi_nat)) >= 0.04))
t3 = pd.DataFrame(t3)
t3.to_csv(OUT / "oraclegain_sensitivity_2024.csv", index=False)
report["mas_2024_gamma_interval"] = [g_lo, r.q_s]

# natural-prevalence ceiling of OracleGain = pi(1-pi)
ceil = pd.DataFrame({"pi_nat": inp.pi_nat, "max_possible_OG": inp.pi_nat * (1 - inp.pi_nat)})
ceil.to_csv(OUT / "natural_oraclegain_ceiling.csv")
report["natural_OG_ceiling_range"] = [float(ceil.max_possible_OG.min()), float(ceil.max_possible_OG.max())]

# maximum OG over primary MAS set at natural prevalence, and over gamma in [0,.25]
g = np.linspace(0, 0.25, 501)
ph = np.linspace(0, 1, 501)
G, P = np.meshgrid(g, ph)
S = r.alpha_s * r.q_s + (1 - r.alpha_s) * G
F = r.alpha_f * r.q_f + (1 - r.alpha_f) * P
ogn = oracle_gain(S, F, r.pi_nat)
report["nat2024_max_OG_grid"] = float(ogn.max())
report["nat2024_max_BSS_grid"] = float(ogn.max() / (r.pi_nat * (1 - r.pi_nat)))
# over the primary MAS set (gamma in [q_s/4, q_s])
g2 = np.linspace(g_lo, r.q_s, 401)
G2, P2 = np.meshgrid(g2, ph)
S2 = r.alpha_s * r.q_s + (1 - r.alpha_s) * G2
F2 = r.alpha_f * r.q_f + (1 - r.alpha_f) * P2
for lab, pi in [("nat", r.pi_nat), ("5050", 0.5)]:
    o = oracle_gain(S2, F2, pi)
    report[f"primaryMAS_2024_OG_range_{lab}"] = [float(o.min()), float(o.max())]
    report[f"primaryMAS_2024_BSS_max_{lab}"] = float(o.max() / (pi * (1 - pi)))

# ---------------------------------------------------------------- 4. robust value beyond the boundary
pts = [(0.10, 0.02), (0.10, 0.05), (0.20, 0.02), (0.20, 0.05), (0.20, 0.10), (0.30, 0.05), (0.30, 0.10), (0.30, 0.20)]
beyond = []
for y, rr in inp.iterrows():
    for phL, gc in pts:
        sL, sU, fL, fU = alt_intervals(rr, gc, phL)
        for lab, pi in [("natural", rr.pi_nat), ("balanced", 0.5)]:
            v, corner = rfv_rect(sL, sU, fL, fU, pi)
            beyond.append(dict(year=y, phi_L=phL, gamma_cap=gc, pop=lab, pi=pi,
                               c_crit=(rr.alpha_f * rr.q_f + (1 - rr.alpha_f) * phL - rr.alpha_s * rr.q_s) / (1 - rr.alpha_s),
                               RFV=v, robust_BSS=v / (pi * (1 - pi)), separated=corner is not None))
beyond = pd.DataFrame(beyond)
beyond.to_csv(OUT / "rfv_beyond_boundary.csv", index=False)

# MAS ceiling retained, fatal floor only: minimal phi_L for separation
mas_floor = []
for y, rr in inp.iterrows():
    phi_star = (rr.q_s - rr.alpha_f * rr.q_f) / (1 - rr.alpha_f)
    rec = dict(year=y, phi_L_star=phi_star)
    for phL in [0.5, 0.9]:
        sL, sU, fL, fU = mas_intervals(rr, phiL=phL)
        v, _ = rfv_rect(sL, sU, fL, fU, 0.5)
        rec[f"BSS_5050_phiL{phL}"] = v / 0.25
    mas_floor.append(rec)
mas_floor = pd.DataFrame(mas_floor).set_index("year")
mas_floor.to_csv(OUT / "mas_fatal_floor.csv")

# 0.04 robust-skill frontier at 50/50: gamma_cap such that robust BSS = 0.04
def gcap_for_bss(rr, phL, pi, target=0.04):
    lo, hi = 0.0, (rr.alpha_f * rr.q_f + (1 - rr.alpha_f) * phL - rr.alpha_s * rr.q_s) / (1 - rr.alpha_s)
    if hi <= 0:
        return np.nan
    fL = rr.alpha_f * rr.q_f + (1 - rr.alpha_f) * phL
    bss = lambda gc: float(oracle_gain(rr.alpha_s * rr.q_s + (1 - rr.alpha_s) * gc, fL, pi)) / (pi * (1 - pi))
    if bss(0.0) < target:
        return np.nan
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bss(mid) >= target else (lo, mid)
    return lo

fr = []
for y, rr in inp.iterrows():
    for phL in [0.0, 0.05, 0.10, 0.20, 0.30]:
        fr.append(dict(year=y, phi_L=phL,
                       c_crit=(rr.alpha_f * rr.q_f + (1 - rr.alpha_f) * phL - rr.alpha_s * rr.q_s) / (1 - rr.alpha_s),
                       gcap_bss004_5050=gcap_for_bss(rr, phL, 0.5),
                       gcap_bss004_nat=gcap_for_bss(rr, phL, rr.pi_nat)))
fr = pd.DataFrame(fr)
fr.to_csv(OUT / "robust_skill_004_frontier.csv", index=False)

# ---------------------------------------------------------------- 5. MAS ceiling decomposition
dec = []
for y, rr in inp.iterrows():
    tu = rr.test_s - rr.alpha_s                       # tested-unresolved survivors (share of all survivors)
    w_tu = tu / (1 - rr.alpha_s)                     # share of unresolved survivors who were tested
    d = t5.loc[y, "d_sep"]
    dec.append(dict(year=y, tested_unresolved_share=tu, w_TU=w_tu,
                    d_sep_primary=d,
                    d_sep_TU_unrestricted=d - tu * (1 - rr.q_s),   # gamma_TU in [0,1]
                    d_sep_TU_all_negative=d + tu * rr.q_s,         # gamma_TU = 0 (most favourable to separation)
                    fatal_tested_unresolved_share=rr.test_f - rr.alpha_f))
dec = pd.DataFrame(dec).set_index("year")
dec.to_csv(OUT / "mas_ceiling_decomposition.csv")

# ---------------------------------------------------------------- 6. conditional vs marginal examples
def marg_from_cells(cells):
    """cells: list of (P(w|Y=0), P(w|Y=1), sL, sU, fL, fU); product rectangles."""
    sL = sum(c[0] * c[2] for c in cells); sU = sum(c[0] * c[3] for c in cells)
    fL = sum(c[1] * c[4] for c in cells); fU = sum(c[1] * c[5] for c in cells)
    return sL, sU, fL, fU

def cond_rfv(cells, pi):
    tot = 0.0
    for p0, p1, sL, sU, fL, fU in cells:
        pw = pi * p1 + (1 - pi) * p0
        piw = pi * p1 / pw
        tot += pw * rfv_rect(sL, sU, fL, fU, piw)[0]
    return tot

pi = 0.5
exA = [(0.5, 0.5, 0.10, 0.20, 0.50, 0.60), (0.5, 0.5, 0.0, 1.0, 0.0, 1.0)]
exB = [(0.1, 0.9, 0.80, 0.80, 0.80, 0.80), (0.9, 0.1, 0.10, 0.10, 0.10, 0.10)]
report["example_A_marginal_RFV"] = rfv_rect(*marg_from_cells(exA), pi)[0]
report["example_A_conditional_RFV"] = cond_rfv(exA, pi)
report["example_B_marginal_RFV"] = rfv_rect(*marg_from_cells(exB), pi)[0]
report["example_B_conditional_RFV"] = cond_rfv(exB, pi)

# Fig. 9 is drawn by fig9_robust_skill.py.

# ---------------------------------------------------------------- print
pd.set_option("display.width", 200, "display.max_columns", 30)
print(json.dumps(report, indent=1, default=float))
print("\nBoundary distances and c_crit(0)\n", t5.round(4))
print("\n2024 OracleGain sensitivity points\n", t3.round(5))
print("\nNatural OG ceiling\n", ceil.round(6))
print("\nMAS decomposition\n", dec.round(4))
print("\nMAS + fatal floor\n", mas_floor.round(4))
print("\nBSS 0.04 frontier\n", fr.round(4).to_string())
print("\nBeyond boundary (2024 and range)\n", beyond[beyond.year == 2024].round(5).to_string())
summ = beyond.groupby(["phi_L", "gamma_cap", "pop"]).agg(sep_years=("separated", "sum"),
        bss_min=("robust_BSS", "min"), bss_max=("robust_BSS", "max"), rfv_min=("RFV", "min"), rfv_max=("RFV", "max"))
print(summ.round(5).to_string())
summ.to_csv(OUT / "rfv_beyond_boundary_summary.csv")
(OUT / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
