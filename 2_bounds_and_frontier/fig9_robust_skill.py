"""Fig. 9: exact robust Brier skill beyond the assumption frontier, CRSS 2024.

Inputs are the 2024 quantities alpha_s, q_s, alpha_f, q_f and the natural fatal
prevalence, read unrounded from year_parameters_gt0.csv (bounds_frontier_oracle.py)
when it exists, otherwise the rounded values reported in the manuscript. Usage:
    python fig9_robust_skill.py [--year-params CSV] [--output PATH]
Default output: outputs/2_bounds_and_frontier/fig9_robust_skill_2024.png
"""
import argparse, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
REPO_ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--output", type=Path,
                 default=REPO_ROOT / "outputs" / "2_bounds_and_frontier" / "fig9_robust_skill_2024.png")
_ap.add_argument("--year-params", type=Path,
                 default=REPO_ROOT / "outputs" / "2_bounds_and_frontier" / "year_parameters_gt0.csv")
_args = _ap.parse_args()
out = _args.output
out.parent.mkdir(parents=True, exist_ok=True)
a_s, q_s, a_f, q_f, pi_nat = 0.0111, 0.5136, 0.1192, 0.412, 0.002182   # rounded fallback
if _args.year_params.exists():
    import pandas as pd
    _r = pd.read_csv(_args.year_params).set_index("year").loc[2024]
    a_s, q_s, a_f, q_f, pi_nat = _r.alpha_s, _r.q_s, _r.alpha_f, _r.q_f, _r.pi_natural
    print(f"inputs: unrounded 2024 quantities ({_args.year_params.name})")
else:
    print("inputs: rounded 2024 values reported in the manuscript")
def og(s, f, pi):
    m1 = pi*f + (1-pi)*s
    return (pi*(1-pi))**2*(f-s)**2/(m1*(1-m1))
ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
phs = np.linspace(0, 0.5, 501); gcs = np.linspace(0, 0.30, 501)
PH, GC = np.meshgrid(phs, gcs)
S = a_s*q_s + (1-a_s)*GC; F = a_f*q_f + (1-a_f)*PH
crit = (a_f*q_f + (1-a_f)*phs - a_s*q_s)/(1-a_s)
panels = [("(a) Standardized 50/50 population", 0.5, [0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20]),
          ("(b) Natural 2024 prevalence", pi_nat, [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006])]
for ax, (lab, pi, lev) in zip(axes, panels):
    Z = np.where(F > S, og(S, F, pi)/(pi*(1-pi)), np.nan)
    cmap = ListedColormap(ramp); norm = BoundaryNorm(lev, cmap.N, extend="max")
    cmap.set_over("#0d366b")
    cf = ax.contourf(PH, GC, Z, levels=lev, cmap=cmap, norm=norm, extend="max")
    ax.fill_between(phs, crit, 0.30, color="#f0efec", zorder=0)
    ax.plot(phs, crit, color="#1f1f1f", lw=1.4)
    if pi == 0.5:
        cs = ax.contour(PH, GC, Z, levels=[0.04], colors="#1f1f1f", linewidths=1.0, linestyles="--")
        ax.annotate("robust skill = 0.04", xy=(0.43, 0.225), xytext=(0.30, 0.275), fontsize=8,
                    color="#1f1f1f", arrowprops=dict(arrowstyle="-", color="#1f1f1f", lw=0.7))
    ax.text(0.015, 0.275, "RFV = 0", fontsize=8.5, color="#3d3d3a")
    ax.annotate("frontier $\\gamma_{cap}=c_{crit}(\\varphi_L)$", xy=(0.12, 0.155), xytext=(0.015, 0.215),
                fontsize=8, color="#1f1f1f", arrowprops=dict(arrowstyle="-", color="#1f1f1f", lw=0.7))
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.30)
    ax.set_xlabel(r"Fatal-side lower bound $\varphi_L$"); ax.set_title(lab, fontsize=9.5)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    cb = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.92)
    cb.set_label("Robust Brier skill, RFV / $R_{delete}$", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
axes[0].set_ylabel(r"Survivor-side cap $\gamma_{cap}$")
fig.tight_layout()
fig.savefig(out, dpi=300, bbox_inches="tight")
