#!/usr/bin/env python
"""
CRSS OTFA Theory / Accounting / Oracle Finalizer
================================================

Reads the completed v0.6 CRSS gate CSV and produces the remaining deterministic
manuscript calculations without retraining any model:
  * all-year alpha_s, alpha_f, q_s, q_f
  * MAS intervals for Gamma = 2, 4, 8, 22
  * exact positive/reverse boundary distances
  * risk intervals for natural and 50/50 populations
  * zero-RFV classification via interval overlap
  * joint external-information frontier c_crit(phi_L)
  * OracleGain(gamma_true, phi_true) surfaces/curves
  * direct fatal/survivor accounting audit from saved v0.6 masses
  * selected external-assumption RFV values by direct minimax optimization

All outputs are written under:
  C:\\Users\\ehsanghaffari\\Downloads\\penndot Data\\Output\\New Output\\CRSS_Theory_Final

The script searches the user's previous Output folders for the v0.6 gate CSV.
No ML training occurs.
"""
from __future__ import annotations
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from scipy.optimize import minimize
except Exception as exc:
    raise ImportError('scipy is required for the selected external-assumption RFV optimization') from exc
BASE = Path('C:\\Users\\ehsanghaffari\\Downloads\\penndot Data')
OUTPUT_DIR = BASE / 'Output' / 'New Output' / 'CRSS_Theory_Final'
INPUT_CANDIDATES = [BASE / 'Output' / 'New Output' / 'CRSS_OTFA_gate' / 'CRSS_OTFA_alcohol_gate_v0_6.csv', BASE / 'Output' / 'CRSS_OTFA_gate' / 'CRSS_OTFA_alcohol_gate_v0_6.csv', BASE / 'Output' / 'CRSS_OTFA_alcohol_gate_v0_6.csv']
PRIMARY_GAMMA = 4.0
GAMMAS = [2.0, 4.0, 8.0, 22.0]
PHI_L_GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3], dtype=float)
ORACLE_GAMMA_GRID = np.linspace(0.0, 0.25, 251)
ORACLE_PHI_GRID = np.linspace(0.0, 1.0, 201)
SELECTED_PHI_L = [0.0, 0.05, 0.1, 0.2, 0.3]
SELECTED_GAMMA_CAP = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

def log(msg: str) -> None:
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)

def find_input() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    tried = '\n'.join((str(p) for p in INPUT_CANDIDATES))
    raise FileNotFoundError('Could not find CRSS_OTFA_alcohol_gate_v0_6.csv. Tried:\n' + tried + '\nRun 01_CRSS_OTFA_Gate_v0_6_NEW_OUTPUT.py --all-years first, or copy the existing CSV to one of those locations.')

def safe_div(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and (b != 0) else np.nan

def oracle_gain(pi: float, s: float, f: float) -> float:
    """Bayes Brier improvement from fully observed binary Z."""
    pi = float(pi)
    s = float(s)
    f = float(f)
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
    return float(m1 * p1 ** 2 + m0 * p0 ** 2 + pi - 2.0 * (y1z1 * p1 + y1z0 * p0))

def minimax_rfv(pi: float, sL: float, sU: float, fL: float, fU: float) -> dict:
    """Minimize max Brier risk over a rectangular (s,f) set.

    When the prevalence intervals overlap, Proposition 1 applies exactly and
    we return RFV=0 analytically rather than asking a numerical optimizer to
    solve a nonsmooth equality case.
    """
    r_delete = pi * (1.0 - pi)
    overlap = not (fL > sU or fU < sL)
    if overlap:
        return {'pi': pi, 'p0_star': pi, 'p1_star': pi, 'R_delete': r_delete, 'R_robust': r_delete, 'RFV': 0.0, 'optimizer_success': True, 'solution_method': 'THEOREM_OVERLAP'}
    corners = [(sL, fL), (sL, fU), (sU, fL), (sU, fU)]

    def objective(x):
        p0, p1 = (float(x[0]), float(x[1]))
        return max((brier_risk(pi, s, f, p0, p1) for s, f in corners))
    starts = [np.array([pi, pi]), np.array([max(0, pi - 0.1), min(1, pi + 0.1)]), np.array([min(1, pi + 0.1), max(0, pi - 0.1)]), np.array([0.25, 0.75]), np.array([0.75, 0.25])]
    best = None
    for x0 in starts:
        res = minimize(objective, x0=x0, method='L-BFGS-B', bounds=[(0, 1), (0, 1)], options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 3000})
        if best is None or res.fun < best.fun:
            best = res
    p0, p1 = map(float, best.x)
    r_robust = float(best.fun)
    rfv = max(0.0, r_delete - r_robust)
    if abs(rfv) < 1e-10:
        rfv = 0.0
    return {'pi': pi, 'p0_star': p0, 'p1_star': p1, 'R_delete': r_delete, 'R_robust': r_robust, 'RFV': rfv, 'optimizer_success': bool(best.success), 'solution_method': 'NUMERICAL_MINIMAX'}

def build_year_parameters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.sort_values('year').iterrows():
        alpha_s = float(r['gt0_survivor_Z_resolved_share_all_survivors'])
        alpha_f = float(r['gt0_fatal_Z_resolved_share_all_fatal'])
        q_s = float(r['q_test_BAC_gt0_weighted'])
        a1 = float(r['gt0_a1_weighted_mass'])
        a0 = float(r['gt0_a0_weighted_mass'])
        q_f = safe_div(a1, a1 + a0)
        fatal_w = float(r['fatal_driver_weighted_count'])
        surv_w = float(r['survivor_driver_weighted_count'])
        pi_nat = safe_div(fatal_w, fatal_w + surv_w)
        rows.append({'year': int(r['year']), 'alpha_s': alpha_s, 'alpha_f': alpha_f, 'q_s': q_s, 'q_f': q_f, 'pi_natural': pi_nat, 'pi_balanced': 0.5, 'survivor_test_rate': float(r['survivor_test_rate_weighted_all_survivors']), 'fatal_test_rate': float(r['fatal_test_rate_weighted_all_fatal']), 'v_rate_direct': float(r['v_rate_weighted_1_minus_fatal_test_rate'])})
    return pd.DataFrame(rows)

def bounds_for(row: pd.Series, Gamma: float, phi_L: float=0.0, phi_U: float=1.0, gamma_cap: float | None=None) -> dict:
    a_s, a_f, q_s, q_f = (row['alpha_s'], row['alpha_f'], row['q_s'], row['q_f'])
    gamma_L = q_s / Gamma
    gamma_U = q_s if gamma_cap is None else min(q_s, gamma_cap)
    sL = a_s * q_s + (1 - a_s) * gamma_L
    sU = a_s * q_s + (1 - a_s) * gamma_U
    fL = a_f * q_f + (1 - a_f) * phi_L
    fU = a_f * q_f + (1 - a_f) * phi_U
    return {'gamma_L': gamma_L, 'gamma_U': gamma_U, 's_L': sL, 's_U': sU, 'f_L': fL, 'f_U': fU}

def risk_interval(pi: float, sL: float, sU: float, fL: float, fU: float) -> tuple[float, float, float, float]:
    L1 = safe_div(pi * fL, pi * fL + (1 - pi) * sU)
    U1 = safe_div(pi * fU, pi * fU + (1 - pi) * sL)
    L0 = safe_div(pi * (1 - fU), pi * (1 - fU) + (1 - pi) * (1 - sL))
    U0 = safe_div(pi * (1 - fL), pi * (1 - fL) + (1 - pi) * (1 - sU))
    return (L1, U1, L0, U0)

def make_accounting_audit(raw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in raw.sort_values('year').iterrows():
        year = int(r['year'])
        for zname in ['gt0', 'ge008']:
            fatal = {'resolved_Z1': float(r[f'{zname}_a1_weighted_mass']), 'resolved_Z0': float(r[f'{zname}_a0_weighted_mass']), 'tested_Z_unresolved': float(r[f'{zname}_fatal_tested_Z_unresolved_weighted_mass']), 'explicit_A0': float(r[f'{zname}_fatal_A0_weighted_mass']), 'unknown_A': float(r[f'{zname}_fatal_A_unknown_weighted_mass'])}
            survivor = {'resolved_Z1': float(r[f'{zname}_b1_weighted_mass']), 'resolved_Z0': float(r[f'{zname}_b0_weighted_mass']), 'tested_Z_unresolved': float(r[f'{zname}_survivor_tested_Z_unresolved_weighted_mass']), 'explicit_A0': float(r[f'{zname}_u_optimistic_weighted_mass']), 'unknown_A': float(r[f'{zname}_u_conservative_weighted_mass'] - r[f'{zname}_u_optimistic_weighted_mass'])}
            for outcome, d in [('fatal', fatal), ('survivor', survivor)]:
                total = sum(d.values())
                for cat, mass in d.items():
                    out.append({'year': year, 'z_definition': zname, 'outcome': outcome, 'category': cat, 'population_mass': mass, 'conditional_share_within_outcome': safe_div(mass, total)})
                out.append({'year': year, 'z_definition': zname, 'outcome': outcome, 'category': 'OUTCOME_PARTITION_SUM', 'population_mass': total, 'conditional_share_within_outcome': 1.0})
            out.append({'year': year, 'z_definition': zname, 'outcome': 'all', 'category': 'FULL_PARTITION_RESIDUAL_FROM_V0_6', 'population_mass': float(r[f'{zname}_full_partition_mass_residual']), 'conditional_share_within_outcome': np.nan})
    return pd.DataFrame(out)

def plot_boundary(boundary_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(boundary_df['year'], boundary_df['s_U'], marker='o', label='s_U = q_test')
    ax.plot(boundary_df['year'], boundary_df['f_L'], marker='s', label='f_L (phi_L=0)')
    ax.fill_between(boundary_df['year'], boundary_df['f_L'], boundary_df['s_U'], alpha=0.15, label='overlap: zero-RFV guarantee')
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel('CRSS year')
    ax.set_ylabel('Admissible alcohol prevalence')
    ax.set_title('Positive-direction separation boundary, 2016-2024')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def plot_frontiers(frontier_all: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for year, g in frontier_all.groupby('year'):
        ax.plot(g['phi_L'], g['c_crit_gamma_cap'], alpha=0.45, label=str(year))
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel('phi_L: lower bound among unresolved fatal drivers')
    ax.set_ylabel('critical gamma_cap among unresolved survivors')
    ax.set_title('Joint external-information frontier')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def plot_oracle_curves(params: pd.DataFrame, outdir: Path) -> None:
    for _, r in params.iterrows():
        year = int(r['year'])
        for pop_name, pi in [('natural', r['pi_natural']), ('balanced', 0.5)]:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for phi_true in [0.05, 0.1, 0.25, 0.5]:
                vals = []
                for gamma_true in ORACLE_GAMMA_GRID:
                    s = r['alpha_s'] * r['q_s'] + (1 - r['alpha_s']) * gamma_true
                    f = r['alpha_f'] * r['q_f'] + (1 - r['alpha_f']) * phi_true
                    vals.append(oracle_gain(pi, s, f))
                ax.plot(ORACLE_GAMMA_GRID, vals, label=f'phi_true={phi_true:.2f}')
            ax.axhline(0.01, linestyle='--', linewidth=1.0, label='0.010 reporting floor')
            ax.set_xlabel('assumed gamma_true among unresolved survivors')
            ax.set_ylabel('OracleGain (absolute Brier reduction)')
            ax.set_title(f'OracleGain sensitivity - {year} - {pop_name}')
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f'OracleGain_{year}_{pop_name}.png', dpi=220)
            plt.close(fig)

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_csv = find_input()
    log(f'Reading {input_csv}')
    raw = pd.read_csv(input_csv)
    params = build_year_parameters(raw)
    params.to_csv(OUTPUT_DIR / 'year_parameters_gt0.csv', index=False)
    bounds_rows = []
    risk_rows = []
    boundary_rows = []
    frontier_rows = []
    external_rfv_rows = []
    for _, r in params.iterrows():
        year = int(r['year'])
        for Gamma in GAMMAS:
            b = bounds_for(r, Gamma=Gamma, phi_L=0.0, phi_U=1.0)
            overlap = not (b['f_L'] > b['s_U'] or b['f_U'] < b['s_L'])
            bounds_rows.append({'year': year, 'Gamma': Gamma, 'phi_L': 0.0, 'phi_U': 1.0, **b, 'positive_margin_fL_minus_sU': b['f_L'] - b['s_U'], 'reverse_margin_sL_minus_fU': b['s_L'] - b['f_U'], 'interval_overlap': overlap, 'RFV_zero_by_theorem': overlap})
            for pop_name, pi in [('natural', r['pi_natural']), ('balanced', 0.5)]:
                L1, U1, L0, U0 = risk_interval(pi, b['s_L'], b['s_U'], b['f_L'], b['f_U'])
                mm = minimax_rfv(pi, b['s_L'], b['s_U'], b['f_L'], b['f_U'])
                risk_rows.append({'year': year, 'Gamma': Gamma, 'population': pop_name, 'pi': pi, 'L1': L1, 'U1': U1, 'L0': L0, 'U0': U0, 'pi_in_Z1_interval': bool(L1 <= pi <= U1), 'pi_in_Z0_interval': bool(L0 <= pi <= U0), **mm})
        b0 = bounds_for(r, Gamma=PRIMARY_GAMMA, phi_L=0.0, phi_U=1.0)
        boundary_rows.append({'year': year, 'q_f': r['q_f'], 'alpha_f': r['alpha_f'], 'f_L': b0['f_L'], 's_U': b0['s_U'], 'q_s': r['q_s'], 'f_L_minus_s_U': b0['f_L'] - b0['s_U'], 's_L': b0['s_L'], 'f_U': b0['f_U'], 's_L_minus_f_U': b0['s_L'] - b0['f_U']})
        for phi_L in np.linspace(0, 0.35, 351):
            numerator = r['alpha_f'] * r['q_f'] + (1 - r['alpha_f']) * phi_L - r['alpha_s'] * r['q_s']
            ccrit = safe_div(numerator, 1 - r['alpha_s'])
            frontier_rows.append({'year': year, 'phi_L': phi_L, 'c_crit_gamma_cap': ccrit})
        for phi_L in SELECTED_PHI_L:
            for gamma_cap in SELECTED_GAMMA_CAP:
                b_ext = bounds_for(r, Gamma=PRIMARY_GAMMA, phi_L=phi_L, phi_U=1.0, gamma_cap=gamma_cap)
                for pop_name, pi in [('natural', r['pi_natural']), ('balanced', 0.5)]:
                    mm = minimax_rfv(pi, b_ext['s_L'], b_ext['s_U'], b_ext['f_L'], b_ext['f_U'])
                    external_rfv_rows.append({'year': year, 'population': pop_name, 'Gamma': PRIMARY_GAMMA, 'phi_L': phi_L, 'gamma_cap': gamma_cap, **b_ext, 'positive_separation': b_ext['f_L'] > b_ext['s_U'], **mm})
    bounds_df = pd.DataFrame(bounds_rows)
    risk_df = pd.DataFrame(risk_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    frontier_df = pd.DataFrame(frontier_rows)
    ext_df = pd.DataFrame(external_rfv_rows)
    bounds_df.to_csv(OUTPUT_DIR / 'all_year_MAS_bounds.csv', index=False)
    risk_df.to_csv(OUTPUT_DIR / 'all_year_risk_intervals_and_RFV.csv', index=False)
    boundary_df.to_csv(OUTPUT_DIR / 'all_year_boundary_distances.csv', index=False)
    frontier_df.to_csv(OUTPUT_DIR / 'joint_assumption_frontier_dense.csv', index=False)
    ext_df.to_csv(OUTPUT_DIR / 'selected_external_assumption_RFV.csv', index=False)
    ccrit_rows = []
    for _, r in params.iterrows():
        row = {'year': int(r['year'])}
        for phi_L in PHI_L_GRID:
            numerator = r['alpha_f'] * r['q_f'] + (1 - r['alpha_f']) * phi_L - r['alpha_s'] * r['q_s']
            row[f'phi_L_{phi_L:.2f}'] = safe_div(numerator, 1 - r['alpha_s'])
        ccrit_rows.append(row)
    pd.DataFrame(ccrit_rows).to_csv(OUTPUT_DIR / 'joint_assumption_frontier_table.csv', index=False)
    accounting = make_accounting_audit(raw)
    accounting.to_csv(OUTPUT_DIR / 'full_accounting_audit.csv', index=False)
    oracle_rows = []
    for _, r in params.iterrows():
        year = int(r['year'])
        for pop_name, pi in [('natural', r['pi_natural']), ('balanced', 0.5)]:
            for gamma_true in ORACLE_GAMMA_GRID:
                s = r['alpha_s'] * r['q_s'] + (1 - r['alpha_s']) * gamma_true
                for phi_true in ORACLE_PHI_GRID:
                    f = r['alpha_f'] * r['q_f'] + (1 - r['alpha_f']) * phi_true
                    oracle_rows.append({'year': year, 'population': pop_name, 'pi': pi, 'gamma_true': gamma_true, 'phi_true': phi_true, 's_true': s, 'f_true': f, 'OracleGain': oracle_gain(pi, s, f)})
    pd.DataFrame(oracle_rows).to_csv(OUTPUT_DIR / 'OracleGain_surface_all_years.csv.gz', index=False, compression='gzip')
    plot_boundary(boundary_df, OUTPUT_DIR / 'Figure_boundary_distances.png')
    plot_frontiers(frontier_df, OUTPUT_DIR / 'Figure_joint_assumption_frontier.png')
    plot_oracle_curves(params, OUTPUT_DIR)
    print('\n=== PRIMARY BROAD-SET BOUNDARY DISTANCES (Gamma=4, phi=[0,1]) ===')
    print(boundary_df[['year', 'q_f', 'alpha_f', 'f_L', 's_U', 'f_L_minus_s_U', 's_L_minus_f_U']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n=== RFV ZERO CHECK: Gamma=4 ===')
    chk = risk_df[risk_df['Gamma'] == PRIMARY_GAMMA][['year', 'population', 'pi', 'L1', 'U1', 'L0', 'U0', 'RFV']]
    print(chk.to_string(index=False, float_format=lambda x: f'{x:.8f}'))
    print(f'\nSaved all outputs to: {OUTPUT_DIR}')
if __name__ == '__main__':
    main()
