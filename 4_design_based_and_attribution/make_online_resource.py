"""
Online Resource 1 as a Word document
====================================

Builds online_resource_1.docx (and refreshes supplementary_tables.xlsx / .md)
from the CSV outputs of design_based_and_attribution.py, without re-running
any analysis. Every table has a numbered caption.

Usage
-----
    python make_online_resource.py [--out outputs/4_design_based_and_attribution]

Requires python-docx.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "4_design_based_and_attribution"
TITLE = ("Robust Predictive Value under Selective Feature Acquisition: "
         "Exact Identification Boundaries in Crash Alcohol Testing")
AUTHOR = "Ehsan Ghaffari"
AFFILIATION = "Department of Civil Engineering, Montana State University, Bozeman, MT 59717, USA"

BAC = {"bac0": "BAC > 0", "bac08": "BAC ≥ 0.08"}
COHORT = {"cohort_id": "Identification", "cohort_ml": "ML"}
TIER = {"core": "Core", "expanded": "Expanded"}
PARTITION = {"night": "Night vs day", "age_group": "Age group", "night_x_age": "Night × age"}
CONTRAST = {"M1-M0": "M_{1} − M_{0}", "M1_3-M0": "M_{1}^{(3)} − M_{0}", "M2-M0": "M_{2} − M_{0}",
            "M2-M_AR": "M_{2} − M_{AR}", "M1_M0": "M_{1} − M_{0}", "M2_M0": "M_{2} − M_{0}",
            "A_first_share": "Acquisition-first share"}
METRIC = {"pr_auc": "PR-AUC", "brier": "Brier score", "roc_auc": "ROC-AUC", "log_loss": "Log loss"}
MINUS = "−"

NOTE_INTERVAL = ("Point estimates with t-based 95% delete-one-PSU jackknife intervals in brackets "
                 "(strata PSUSTRAT, clusters PSU_VAR, weight WEIGHT; Sect. 5.8). Intervals are the estimate plus or "
                 "minus t times the jackknife standard error, without truncation, so where the design effective sample "
                 "size is small a limit can fall below 0 or above 1; such limits indicate imprecision, not admissible "
                 "values.")


# ----------------------------------------------------------------------------- number formats
def _neg(s: str) -> str:
    return s.replace("-", MINUS)


def _nz(s: str) -> str:
    """'-0.000' -> '0.000' (a value that rounds to zero carries no sign)."""
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def f4(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else _neg(_nz(f"{v:.4f}"))


def f3(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else _neg(_nz(f"{v:.3f}"))


def signed4(v) -> str:
    return "n/a" if pd.isna(v) else _neg(f"{v:+.4f}").replace("+0.0000", "0.0000").replace(MINUS + "0.0000", "0.0000")


def sci(v, digits: int = 2) -> str:
    """0 -> '0'; small values as 'm × 10^{e}' (markup for a superscript)."""
    if pd.isna(v):
        return "n/a"
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    if -3 <= e <= 2:
        return _neg(f"{v:.{max(digits + 1 - e - 1, 0) + 1}f}") if e < 0 else _neg(f"{v:.{digits}f}")
    m = v / 10 ** e
    return _neg(f"{m:.{digits}f}") + " × 10^{" + _neg(str(e)) + "}"


def est_ci(cell: str) -> str:
    """'0.8593 [0.8047, 0.9138]' -> two lines; a bare estimate stays one line."""
    if not isinstance(cell, str):
        return f4(cell)
    return _neg(cell.replace(" [", "\n["))


def mean_sd(m, s, fmt=f3) -> str:
    return f"{fmt(m)} ({fmt(s)})" if not pd.isna(s) else fmt(m)


# ----------------------------------------------------------------------------- table specs
def _spec_identification(t: pd.DataFrame, cols: dict[str, str]) -> tuple[list[str], list[list[str]]]:
    t = t.reset_index()
    header = ["Year"] + [cols[c] for c in cols if c in t]
    rows = [[str(int(r["year"]))] + [est_ci(r[c]) for c in cols if c in t] for _, r in t.iterrows()]
    return header, rows


S1_COLS = {"pi": "π", "test_s": "Tested, survivors", "test_f": "Tested, fatalities",
           "alpha_s": "α_{s}", "alpha_f": "α_{f}", "q_s": "q_{s}", "q_f": "q_{f}"}
S2_COLS = {"s_L": "s_{L}", "s_U": "s_{U}", "f_L": "f_{L}", "f_U": "f_{U}", "d_sep": "d_{sep} = f_{L} − s_{U}",
           "rev_margin": "s_{L} − f_{U}"}
S3_COLS = {f"c_crit_{p:.2f}": f"φ_{{L}} = {p:g}" for p in (0.0, 0.05, 0.10, 0.20, 0.30)}
S4_COLS = {"unknown_s": "Unknown status, survivors", "unknown_f": "Unknown status, fatalities",
           "test_s_known": "Tested, survivors with known status", "test_f_known": "Tested, fatalities with known status",
           "share_TU_s": "Tested-but-unresolved survivors", "d_sep_TU_free": "d_{sep}, TU unrestricted",
           "d_sep_TU_unknown_free": "d_{sep}, TU and unknown unrestricted",
           "d_sep_TU_unknown_negative": "d_{sep}, TU and unknown negative"}
S5_COLS = {"pi": "Δπ", "alpha_s": "Δα_{s}", "q_s": "Δq_{s}", "alpha_f": "Δα_{f}",
           "q_f": "Δq_{f}", "d_sep": "Δd_{sep}", "rev_margin": "Δ(s_{L} − f_{U})"}
S6_COLS = {"survivors": "Kish ESS, survivors", "fatal": "Kish ESS, fatalities",
           "resolved_survivors": "Kish ESS, resolved survivors", "resolved_fatal": "Kish ESS, resolved fatalities",
           "design_ess_alpha_s": "Design ESS, α_{s}", "design_ess_alpha_f": "Design ESS, α_{f}",
           "design_ess_q_s": "Design ESS, q_{s}", "design_ess_q_f": "Design ESS, q_{f}"}
SHORT = {1: "Prevalence, testing, and resolution", 2: "MAS intervals and overlap margins",
         3: "External-information frontier c_{crit}(\u03c6_{L})", 4: "Testing-status components and margin sensitivity",
         5: "Cohort harmonization of the identification audit", 6: "Effective sample sizes",
         7: "Conditional RFV given coarse partitions of W", 8: "Design robustness of the zero-RFV classification",
         9: "PR-AUC by specification, tier, and model family", 10: "Feature-block order, PR-AUC",
         11: "Feature-block order, Brier score", 12: "Evaluation-stage design-based uncertainty",
         13: "Refitting delete-a-group jackknife", 14: "Reproduction check, identification quantities",
         15: "Reproduction check, machine-learning fits", 16: "Annual selectivity analysis"}
ML_MODELS = {"M0": "M_{0}", "M1": "M_{1}", "M1_3": "M_{1}^{(3)}", "M_R": "M_{R}", "M_AR": "M_{AR}", "M2": "M_{2}",
             "M2_3": "M_{2}^{(3)}"}


def spec(name: str, t: pd.DataFrame) -> dict:
    """Caption, header, rows and options for one table."""
    m = re.match(r"S(\d+)_.*?(bac0|bac08)?$", name)
    num = int(m.group(1))
    thr = m.group(2)
    part = "" if num == 16 else {"bac0": "a", "bac08": "b"}.get(thr, "")   # S16 exists for BAC >= 0.08 only
    label = f"Table S{num}{part}"
    bac = BAC.get(thr, "")
    landscape = True
    if num == 1:
        header, rows = _spec_identification(t, S1_COLS)
        cap = (f"Annual fatal share, testing rates, resolved shares, and alcohol-positive shares among resolved drivers, "
               f"{bac}, identification cohort. π, weighted share of fatalities; tested, share of drivers given an "
               f"alcohol test; α, share whose alcohol state Z is resolved; q, alcohol-positive share among resolved "
               f"drivers; subscripts s and f, survivors and fatalities. {NOTE_INTERVAL}")
    elif num == 2:
        header, rows = _spec_identification(t, S2_COLS)
        cap = (f"MAS interval endpoints and overlap margins, {bac} (Γ = 4, φ ∈ [0, 1]). Survivor alcohol "
               f"prevalence lies in [s_{{L}}, s_{{U}}] and fatal prevalence in [f_{{L}}, f_{{U}}]. RFV = 0 when both "
               f"d_{{sep}} and s_{{L}} − f_{{U}} are nonpositive, that is, when the two intervals overlap. {NOTE_INTERVAL}")
    elif num == 3:
        header, rows = _spec_identification(t, S3_COLS)
        cap = (f"Critical survivor-side cap c_{{crit}}(φ_{{L}}) of the external-information frontier (Sect. 3.8), "
               f"{bac}. Positive separation requires γ_{{cap}} < c_{{crit}}(φ_{{L}}) for the stated fatal-side "
               f"floor φ_{{L}}. {NOTE_INTERVAL}")
    elif num == 4:
        header, rows = _spec_identification(t, S4_COLS)
        point_only = not any("[" in str(v) for v in t.get("d_sep_TU_unknown_negative", pd.Series(dtype=str)))
        cap = (f"Testing-status components of the unresolved mass and their effect on the separation margin, {bac}. "
               f"Unknown status, share of drivers whose testing status is unknown or not reported; tested with known "
               f"status, testing rate among drivers whose status is known; TU, tested-but-unresolved survivors as a "
               f"share of all survivors. The last three columns give d_{{sep}} when the alcohol prevalence of the TU "
               f"component, and of the TU and unknown-status components, is left unrestricted, and when both components "
               f"are alcohol-negative (the case most favorable to separation). {NOTE_INTERVAL}"
               + (" The last column is a point estimate computed from the other columns." if point_only else ""))
    elif num == 5:
        tt = t.reset_index()
        header = ["Year"] + [S5_COLS[c] for c in S5_COLS if c in tt]
        rows = [[str(int(r["year"]))] + [signed4(r[c]) for c in S5_COLS if c in tt] for _, r in tt.iterrows()]
        cap = (f"Cohort harmonization of the identification audit, {bac}: each quantity on the machine-learning cohort "
               f"(injury severity codes 0 to 4) minus the same quantity on the identification cohort (all drivers except "
               f"those who died before the crash). Notation as in Tables S1 and S2.")
    elif num == 6:
        tt = t.reset_index()
        header = ["Year"] + [S6_COLS[c] for c in S6_COLS if c in tt]
        rows = [[str(int(r["year"]))] + [f"{r[c]:,.1f}" if not pd.isna(r[c]) else "n/a" for c in S6_COLS if c in tt]
                for _, r in tt.iterrows()]
        cap = (f"Annual effective sample sizes, {bac}, identification cohort. Kish ESS, (Σw)^{{2}}/Σw^{{2}} "
               f"within each outcome and resolution cell; design ESS, p̂(1 − p̂)/Var_{{JK}}(p̂) for "
               f"the stated proportion.")
    elif num == 7:
        tt = t.reset_index()
        header = ["Year", "Partition of W", "Cells", "Separated cells", "Cells without resolved survivors",
                  "Conditional RFV", "Conditional RFV / [π(1 − π)]"]
        rows = [[str(int(r["year"])), PARTITION.get(r["partition"], r["partition"]), str(int(r["cells"])),
                 str(int(r["separated_cells"])), str(int(r["cells_no_resolved_survivors"])),
                 sci(r["conditional_RFV"]), f4(r["conditional_RFV_over_Rdel"])] for _, r in tt.iterrows()]
        cap = (f"Conditional RFV given coarse partitions of the covariates W (Sect. 3.9 and 6.11), {bac}, identification "
               f"cohort, Γ = 4. Conditional RFV is the probability-weighted sum of the cell-specific exact RFV "
               f"(Eq. 20); in cells without resolved survivors the survivor prevalence is left unrestricted. The last "
               f"column divides conditional RFV by the marginal R_{{delete}} = π(1 − π). A cell can be separated on the "
               f"strength of very few resolved drivers; the separated 2023 cells are described in Sect. 6.11 "
               f"(conditional_rfv_cell_detail.py).")
    elif num == 8:
        tt = t.reset_index()
        header = ["Cohort", "Definition", "Year", "Upper 95% limit, d_{sep}", "Upper 95% limit, s_{L} − f_{U}",
                  "Largest replicate, d_{sep}", "Largest replicate, s_{L} − f_{U}", "Design-robust overlap",
                  "Overlap in every replicate"]
        yn = lambda v: "Yes" if bool(v) else "No"
        rows = [[COHORT.get(r["cohort"], r["cohort"]), BAC.get(r["threshold"], r["threshold"]), str(int(r["year"])),
                 f4(r["hi_d_sep"]), f4(r["hi_rev_margin"]), f4(r["rep_max_d_sep"]), f4(r["rep_max_rev_margin"]),
                 yn(r["overlap_robust_95"]), yn(r["overlap_all_replicates"])] for _, r in tt.iterrows()]
        cap = ("Design robustness of the zero-RFV classification by cohort, BAC definition, and year. The classification "
               "is design-robust when the upper 95% jackknife limits of both overlap margins are negative; the largest "
               "replicate value is the maximum over all PSU-deleted jackknife replicates.")
    elif num == 9:
        tt = t.reset_index()
        models = [mm for mm in ML_MODELS if mm in tt]
        header = ["Cohort", "Tier", "Family"] + [ML_MODELS[mm] for mm in models]
        rows = [[COHORT.get(r["cohort"], r["cohort"]), TIER.get(r["tier"], r["tier"]), r["family"]]
                + [f4(r[mm]) for mm in models] for _, r in tt.iterrows()]
        cap = ("Pooled five-seed mean fatal PR-AUC by specification (Sect. 5.4), covariate tier, and model family"
               + (f", {bac}" if bac else "") + ". "
               "M_{0} = W; M_{1} = W + A; M_{1}^{(3)}, M_{1} with three-level testing status; M_{R} = W + R + R × Z; "
               "M_{AR} = W + A + R; M_{2} = W + A + R + R × Z. "
               + ("M_{0} and M_{1} do not use alcohol results and are the BAC > 0 fits; M_{R}, M_{AR}, and M_{2} are "
                  "refitted with R and R × Z defined for BAC ≥ 0.08, on the same case splits, seeds, and "
                  "hyperparameters. " if thr == "bac08" else
                  "On the machine-learning cohort, M_{0}, M_{1}, and M_{2} reproduce Table 7 (Table S15); on the "
                  "identification cohort only M_{0} and M_{1} are fitted. ")
               + "LR, logistic regression; XGB, XGBoost; CAT, CatBoost; n/a, not fitted.")
        landscape = False
    elif num in (10, 11):
        tt = t.reset_index()
        brier = num == 11
        tg = (lambda mm, ss: mean_sd(mm * 1e4, ss * 1e4, lambda v: f"{v:.2f}")) if brier else (lambda mm, ss: mean_sd(mm, ss, f4))
        cols = [("total_gain", "Total reduction, B(M_{0}) − B(M_{2}) (×10^{−4})" if brier else "Total gain, M_{2} − M_{0}"),
                ("A_first", "Acquisition-first share"), ("A_last", "Acquisition-last share"),
                ("A_shapley2", "Two-block Shapley share"), ("value_only", "Value-only share")]
        cols = [c for c in cols if f"{c[0]}_mean" in tt]
        header = ["Tier", "Family"] + [h for _, h in cols]
        rows = []
        for _, r in tt.iterrows():
            row = [TIER.get(r["tier"], r["tier"]), r["family"]]
            for c, _ in cols:
                row.append(tg(r[f"{c}_mean"], r[f"{c}_std"]) if c == "total_gain" else mean_sd(r[f"{c}_mean"], r[f"{c}_std"]))
            rows.append(row)
        what = "Brier-score reduction" if brier else "PR-AUC gain"
        if brier:   # Brier: every difference is a reduction, baseline minus model
            forms = ("Brier score B(M) of each specification; every difference is a reduction, baseline minus model. "
                     "Acquisition-first share, (B(M_{0}) − B(M_{1}))/(B(M_{0}) − B(M_{2})); acquisition-last share, "
                     "(B(M_{R}) − B(M_{2}))/(B(M_{0}) − B(M_{2})); two-block Shapley share, their average; value-only "
                     "share, (B(M_{AR}) − B(M_{2}))/(B(M_{0}) − B(M_{2})).")
        else:
            forms = ("Acquisition-first share, (M_{1} − M_{0})/(M_{2} − M_{0}); acquisition-last share, "
                     "(M_{2} − M_{R})/(M_{2} − M_{0}); two-block Shapley share, their average; value-only share, "
                     "(M_{2} − M_{AR})/(M_{2} − M_{0}), with each M denoting fatal PR-AUC.")
        cap = (f"Feature-block order: share of the M_{{0}} → M_{{2}} {what} attributed to the acquisition block, "
               f"machine-learning cohort{', ' + bac if bac else ''}. Entries are means (standard deviations) over the five seeds of the per-seed "
               f"values; Table 7 and Sect. 6.7 report the corresponding ratios of five-seed means. " + forms)
        landscape = True
    elif num == 12:
        tt = t.reset_index()
        header = ["Cohort", "Tier", "Family", "Contrast", "Estimate", "Evaluation-stage jackknife SE", "Seed SD"]
        rows = [[COHORT.get(r["cohort"], r["cohort"]), TIER.get(r["tier"], r["tier"]), r["family"],
                 CONTRAST.get(r["contrast"], r["contrast"]), f4(r["est"]), f4(r["se_eval_jk"]), f4(r["seed_sd"])]
                for _, r in tt.iterrows()]
        cap = ("Evaluation-stage design-based uncertainty for the machine-learning contrasts (PR-AUC)"
               + (f", {bac}" if bac else "") + ". Each fitted model "
               "is held fixed and the weighted test-set PR-AUC is recomputed under the delete-one-PSU replicate weights. "
               "Estimate and jackknife SE are means over the five seeds; seed SD is the standard deviation of the "
               "estimate across seeds."
               + (" For BAC ≥ 0.08 only the value-only contrast M_{2} − M_{AR} is shown, because M_{0} was not "
                  "refitted (its BAC > 0 fit is threshold-free) and its test predictions were not stored." if thr == "bac08" else ""))
        landscape = False
    elif num == 13:
        tt = t.reset_index()
        G = int(tt["groups"].iloc[0])
        fams = sorted(set(tt["family"]))
        header = ["Tier", "Family", "Statistic", "Estimate", "Jackknife SE", "95% interval"]
        rows = [[TIER.get(r["tier"], r["tier"]), r["family"], CONTRAST.get(r["stat"], r["stat"]), f4(r["est"]),
                 f4(r["se_dagjk"]), f"[{f4(r['lo'])}, {f4(r['hi'])}]"] for _, r in tt.iterrows()]
        fam_txt = " and ".join({"LR": "logistic regression", "XGB": "XGBoost", "CAT": "CatBoost"}.get(f, f) for f in fams)
        if thr == "bac08":
            cap = (f"Refitting delete-a-group jackknife for the machine-learning contrasts in PR-AUC, {bac} ({G} PSU "
                   f"groups, one seed split, {fam_txt}). M_{{0}} and M_{{2}} are refitted within each replicate, with the "
                   f"same PSU groups as Table S13a; M_{{1}} − M_{{0}} is taken from the BAC > 0 replicates, because it does "
                   f"not use alcohol results. Intervals are t-based with {G - 1} degrees of freedom.")
        else:
            cap = (f"Refitting delete-a-group jackknife for the machine-learning contrasts in PR-AUC"
                   f"{', ' + bac if bac else ''} ({G} PSU groups, one "
                   f"seed split, {fam_txt}). M_{{0}}, M_{{1}}, and M_{{2}} are refitted within each replicate, so sampling "
                   f"variability is propagated through model fitting. Intervals are t-based with {G - 1} degrees of freedom.")
        landscape = False
    elif num == 14:
        tt = t.reset_index()
        header = ["Statistic", "Gate output column", "Maximum absolute difference"]
        stat = {**{k: v for k, v in S1_COLS.items()}, "unknown_s": "Unknown status, survivors"}
        rows = [[stat.get(r["statistic"], str(r["statistic"])), str(r["gate_column"]), sci(r["max_abs_diff"])]
                for _, r in tt.iterrows()]
        cap = ("Reproduction check of the identification quantities (BAC > 0): maximum absolute difference over "
               "2016–2024 between this analysis and the identification-gate output used for Tables 2 to 6.")
        landscape = False
    elif num == 15:
        tt = t.reset_index()
        header = ["Metric", "Fits compared", "Maximum absolute difference"]
        rows = [[METRIC.get(r["metric"], r["metric"]), str(int(r["n_compared"])), sci(r["max_abs_diff"])]
                for _, r in tt.iterrows()]
        cap = ("Reproduction check of the machine-learning fits: maximum absolute difference between M_{0}, M_{1}, and "
               "M_{2} on the machine-learning cohort in this analysis and the pooled premise results behind Table 7, "
               "over seeds, model families, and covariate tiers.")
        landscape = False
    elif num == 16:
        tt = t.reset_index()
        header = ["Family", "Years", "ρ(q_{ML}, PR-AUC gain)", "p", "ρ(q_{ML}, Brier gain)", "p"]
        sg = lambda v: "n/a" if pd.isna(v) else _neg(_nz(f"{v:+.3f}")).replace("+0.000", "0.000")
        rows = [[r["family"], str(int(r["n_years"])), sg(r["rho_q_pr_gain"]), f3(r["p_q_pr_gain"]),
                 sg(r["rho_q_brier_gain"]), f3(r["p_q_brier_gain"])] for _, r in tt.iterrows()]
        cap = (f"Annual selectivity analysis (Sect. 5.6), {bac}: Spearman correlations between annual q_{{ML}} and the "
               f"held-out M_{{2}} − M_{{0}} gain in fatal PR-AUC and in Brier score (reduction), core tier, five-seed means, "
               f"as in Table 9. M_{{2}} and q_{{ML}} use BAC ≥ 0.08; M_{{0}} is the BAC > 0 fit, which does not use "
               f"alcohol results. With nine annual points the analysis is descriptive.")
        landscape = False
    else:
        raise KeyError(name)
    return dict(label=label, caption=cap, header=header, rows=rows, landscape=landscape, num=num, part=part)


# ----------------------------------------------------------------------------- docx writing
def _rich(par, text: str, size=None, bold=False, italic=False):
    """Add text to a paragraph; `_{x}` is a subscript, `^{x}` a superscript, newline a line break."""
    for line_i, line in enumerate(str(text).split("\n")):
        if line_i:
            par.add_run().add_break()
        for tok in re.split(r"(_\{[^}]*\}|\^\{[^}]*\})", line):
            if not tok:
                continue
            run = par.add_run(tok[2:-1] if tok[:2] in ("_{", "^{") else tok)
            run.bold, run.italic = bold, italic
            if size:
                run.font.size = size
            if tok.startswith("_{"):
                run.font.subscript = True
            elif tok.startswith("^{"):
                run.font.superscript = True


def _border(cell, edge: str, sz: int = 8):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")
    borders.append(el)


def _no_table_borders(table):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        b.append(el)
    after = [qn(f"w:{t}") for t in ("shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription")]
    nxt = next((c for c in tblPr if c.tag in after), None)   # schema order: tblBorders precedes these
    if nxt is None:
        tblPr.append(b)
    else:
        nxt.addprevious(b)


def _set_section(section, landscape: bool):
    from docx.enum.section import WD_ORIENT
    from docx.shared import Inches
    section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    section.page_width, section.page_height = (Inches(11), Inches(8.5)) if landscape else (Inches(8.5), Inches(11))
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(0.8))


def write_docx(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    specs = [spec(k, v) for k, v in tables.items()]
    specs.sort(key=lambda s: (s["num"], s["part"]))
    doc = Document()
    cp = doc.core_properties                      # no generator stamp in the file properties
    cp.author, cp.last_modified_by, cp.comments = AUTHOR, AUTHOR, ""
    cp.title, cp.subject, cp.keywords = "Online Resource 1", "", ""
    cp.created = cp.modified = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(6)
    _set_section(doc.sections[0], landscape=False)

    p = doc.add_paragraph()
    _rich(p, "Online Resource 1", size=Pt(14), bold=True)
    p = doc.add_paragraph()
    _rich(p, TITLE, size=Pt(12), bold=True)
    p = doc.add_paragraph()
    _rich(p, f"{AUTHOR}\n{AFFILIATION}")
    p = doc.add_paragraph()
    _rich(p, "Supplementary tables", size=Pt(12), bold=True)
    notes = [
        "All quantities are weighted with the CRSS case weights. Unless a table states otherwise, results refer to the "
        "identification cohort (all drivers except those who died before the crash), the primary MAS set with Γ = 4, "
        "and CRSS data years 2016 to 2024. Tables S1 to S7 and S9 to S13 are given for the primary "
        "alcohol definition BAC > 0 (a) and for BAC ≥ 0.08 (b).",
        "Design-based intervals use the delete-one-PSU jackknife recommended for CRSS (strata PSUSTRAT, clusters "
        "PSU_VAR, weight WEIGHT) with t-based 95% intervals (Sect. 5.8). The machine-learning cohort keeps drivers with "
        "injury severity codes 0 to 4 and is the cohort of Table 7.",
        "The tables are generated by make_online_resource.py from the outputs of design_based_and_attribution.py "
        "and bac08_result_block.py "
        "in the analysis code, version 0.7 (release tag v0.7, "
        "https://github.com/ehsanghaffari/Robust-Predictive-Value-under-Selective-Feature-Acquisition/releases/tag/v0.7).",
    ]
    for n in notes:
        p = doc.add_paragraph()
        _rich(p, n)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if "https://" in n else WD_ALIGN_PARAGRAPH.JUSTIFY
    p = doc.add_paragraph()
    _rich(p, "Contents", bold=True)
    for s in specs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        _rich(p, f"{s['label']}. {SHORT[s['num']]}" + (f", {BAC['bac0' if s['part'] == 'a' else 'bac08']}" if s["part"] else ""),
              size=Pt(10))

    for s in specs:
        sec = doc.add_section(WD_SECTION.NEW_PAGE)
        _set_section(sec, s["landscape"])
        width = Inches(11 - 1.6) if s["landscape"] else Inches(8.5 - 1.6)
        cap = doc.add_paragraph()
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_after = Pt(4)
        _rich(cap, s["label"] + " ", size=Pt(10), bold=True)
        _rich(cap, s["caption"], size=Pt(10))
        cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        ncol = len(s["header"])
        table = doc.add_table(rows=1 + len(s["rows"]), cols=ncol)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        _no_table_borders(table)
        def _plain(v):
            return re.sub(r"[_^]\{([^}]*)\}", r"\1", str(v))
        lens = [max([len(_plain(s["header"][j])) / 2.2, max(len(w) for w in _plain(s["header"][j]).split())]
                    + [max(len(x) for x in _plain(r[j]).split("\n")) for r in s["rows"]])
                for j in range(ncol)]
        natural = [Inches(0.068 * l + 0.2) for l in lens]   # text width at 8.5-9 pt plus cell padding
        fixed = [j for j in range(ncol) if lens[j] <= 8]    # short columns (Year, Tier, Family) keep their width
        free = [j for j in range(ncol) if j not in fixed]
        if not free:                                       # every column is short: share the width among all
            fixed, free = [], list(range(ncol))
        scale = (width - sum(natural[j] for j in fixed)) / sum(natural[j] for j in free)
        widths = [int(natural[j] * (scale if j in free else 1)) for j in range(ncol)]
        for j, w in enumerate(widths):                        # grid widths (LibreOffice and Word both honor these)
            table.columns[j].width = w
        compact = len(s["rows"]) > (30 if s["landscape"] else 44)   # long tables stay on one page
        fs = Pt(8) if compact else (Pt(8.5) if ncol >= 8 else Pt(9))
        for i, row in enumerate([s["header"]] + s["rows"]):
            tr = table.rows[i]
            trPr = tr._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            trPr.append(cant)
            if i == 0:
                hdr = OxmlElement("w:tblHeader")
                trPr.append(hdr)
            for j, val in enumerate(row):
                cell = tr.cells[j]
                cell.width = widths[j]
                par = cell.paragraphs[0]
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.space_before = Pt(0 if compact else 1)
                par.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 or (i > 0 and not re.match(r"^[\d−+\[(n]", str(val))) else WD_ALIGN_PARAGRAPH.CENTER
                _rich(par, val, size=fs, bold=(i == 0))
                if i == 0:
                    _border(cell, "top", 10)
                    _border(cell, "bottom", 6)
                if i == len(s["rows"]):
                    _border(cell, "bottom", 10)
    doc.save(path)
    return path


def load_design_module():
    path = Path(__file__).resolve().parent / "design_based_and_attribution.py"
    sp = importlib.util.spec_from_file_location("design_based_and_attribution", path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="outputs folder of design_based_and_attribution.py (default: outputs/4_design_based_and_attribution)")
    a = ap.parse_args()
    d = load_design_module()
    tables = d.build_tables(a.out)
    d.write_resource(a.out, tables)
    path = write_docx(tables, a.out / "online_resource_1.docx")
    print(f"wrote {path} and supplementary_tables.xlsx / .md ({len(tables)} tables)")


if __name__ == "__main__":
    main()
