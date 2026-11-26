#!/usr/bin/env python3
"""
Machine-learning result-block analyses for the secondary alcohol definition BAC >= 0.08
=====================================================================================

The machine-learning analyses in design_based_and_attribution.py and
3_ml_premise/ml_premise_temporal_annual.py use BAC > 0. Specifications that use only
the covariates W and the acquisition indicator A (M0 and M1) do not depend on the BAC
definition, so this script does not refit them: their results are read from the
BAC > 0 outputs. It refits only the specifications that use result resolution R or
the resolved alcohol value R x Z, now defined for BAC >= 0.08, with the same case
splits, seeds, model families, covariate tiers and hyperparameters:

  * pooled block order on the machine-learning cohort (Table 7 split, five seeds,
    LR / XGB / CAT, core and expanded tiers): M_R, M_AR and M2, the acquisition-first,
    acquisition-last and two-block Shapley shares, and the value-only contrast
    M2 - M_AR with its evaluation-stage PSU jackknife;
  * refitting delete-a-group jackknife (15 PSU groups, seed 42, LR and XGB): M2 is
    refitted in every replicate; M0 is refitted too because the BAC > 0 run saved
    replicate differences, not levels; M1 - M0 is taken from the BAC > 0 replicates;
  * annual within-year analysis (core tier, five seeds, three families): M2 per year,
    q_ML for BAC >= 0.08 and the Spearman correlations of Table 9.

Definitions (as in the identification analysis): resolved R = alcohol test given and a
numeric ALC_RES 0-940; R x Z = resolved and ALC_RES >= 80 (BAC >= 0.08). The BAC > 0
code 998 (positive, no value) does not resolve BAC >= 0.08.

Before fitting anything new, the script checks that it rebuilds the BAC > 0 features
and fits exactly (bac08_alignment_check.csv). Every block of work is checkpointed, so
an interrupted run resumes where it stopped.

Inputs
------
  --zipdir DIR        CRSS{year}CSV.zip files (default data/crss_zips)
  --out DIR           outputs/4_design_based_and_attribution (holds the BAC > 0 results:
                      ml_block_order_results.csv or ml_fits_cohort_ml.csv, and
                      ml_dagjk_replicates.csv, directly or in a bac0_reference sub-folder);
                      the BAC >= 0.08 results are written there
  --ml-metrics-csv    outputs/3_ml_premise/all_model_metrics.csv (annual M0 and the
                      pooled premise results)
If a needed BAC > 0 result is missing, that model is refitted and the log says so.

Usage
-----
  python 4_design_based_and_attribution/bac08_result_block.py --quick   (smoke test, minutes)
  python 4_design_based_and_attribution/bac08_result_block.py           (full run, hours)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
THR = "bac08"
RESULT_MODELS = ["M_R", "M_AR", "M2"]


def _load(path: Path, name: str):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


D4 = _load(HERE / "design_based_and_attribution.py", "design_based_and_attribution")


# ----------------------------------------------------------------------------- data
def build_cohort_ml(m03, zipdir: Path, years, log) -> pd.DataFrame:
    """Pooled machine-learning cohort, built exactly as build_ml_frames in design_based_and_attribution.py."""
    fr = {}
    for y in years:
        per, acc, veh = m03.read_year_tables(D4.zip_path(zipdir, y))
        fr[y] = m03.build_driver_frame(y, per, acc, veh, design=True)
    d, common = m03.pool_years(fr, extra=D4.DESIGN_EXTRA)
    st = d["ALC_STATUS_RAW"]
    d["S3_TESTED"] = st.isin(D4.CODEMAP["ALC_STATUS_tested"]).astype(int)
    d["S3_UNKNOWN"] = (~st.isin(D4.CODEMAP["ALC_STATUS_tested"] + D4.CODEMAP["ALC_STATUS_not_tested"])).astype(int)
    d["Y"] = d["Y_FATAL"].astype(int)
    d["w"] = d["SAMPLE_WEIGHT"].astype(float)
    d["strat"] = d["DSGN_STRAT"].astype(str)
    d["psu"] = d["DSGN_STRAT"].astype(str) + ":" + d["DSGN_PSU"].astype(str)
    d["case"] = d["CASE_ID"]
    log(f"ML cohort: {len(d):,} drivers, {d.CASE_ID.nunique():,} cases, {len(common)} covariate concepts")
    return d, common


def result_columns(d: pd.DataFrame, thr: str) -> tuple[np.ndarray, np.ndarray]:
    """R and R x Z from the raw CRSS codes for BAC > 0 ('bac0') or BAC >= 0.08 ('bac08')."""
    A = d["A_TEST"].to_numpy(int) == 1
    res = pd.to_numeric(d["ALC_RES_RAW"], errors="coerce").to_numpy(float)
    numeric = A & (res >= 0) & (res <= D4.CODEMAP["ALC_RES_numeric_max"])
    if thr == "bac0":
        posnov = A & np.isin(res, D4.CODEMAP["ALC_RES_positive_no_value"])
        R = numeric | posnov
        RZ = (numeric & (res > 0)) | posnov
    else:
        R = numeric
        RZ = numeric & (res >= D4.CODEMAP["BAC08_threshold"])
    return R.astype(int), RZ.astype(int)


def bac0_view(d8: pd.DataFrame) -> pd.DataFrame:
    """The same rows with the BAC > 0 result features restored."""
    return d8.assign(R_RESOLVED=d8["R0_"].astype(int), RZ_POSITIVE=d8["RZ0_"].astype(int))


def frame_bac08(d: pd.DataFrame, log) -> pd.DataFrame:
    R0, RZ0 = result_columns(d, "bac0")
    bad = int((R0 != d["R_RESOLVED"].to_numpy(int)).sum() + (RZ0 != d["RZ_POSITIVE"].to_numpy(int)).sum())
    log(f"BAC > 0 result features rebuilt from raw codes: {bad} mismatches with 3_ml_premise (must be 0)")
    if bad:
        raise RuntimeError("raw ALC_RES coding does not reproduce the BAC > 0 features; stop and check CODEMAP")
    R8, RZ8 = result_columns(d, "bac08")
    d8 = d   # modified in place to save memory; the BAC > 0 columns are kept as R0_ / RZ0_
    d8["R0_"], d8["RZ0_"] = R0.astype("int8"), RZ0.astype("int8")
    d8["R_RESOLVED"], d8["RZ_POSITIVE"] = R8, RZ8
    log(f"BAC >= 0.08: resolved {R8.sum():,} drivers (BAC > 0: {R0.sum():,}); "
        f"positive {RZ8.sum():,} (BAC > 0: {RZ0.sum():,})")
    return d8


# ----------------------------------------------------------------------------- BAC > 0 reference results
class Reference:
    """M0 / M1 metrics and DAGJK replicate differences from the BAC > 0 runs."""

    def __init__(self, out: Path, ml_csv: Path | None, log):
        self.pooled = pd.DataFrame()
        dirs = [out, out / "bac0_reference"]            # the BAC > 0 outputs, or a copy of them in bac0_reference/
        for f in ["ml_block_order_results.csv", "ml_fits_cohort_ml.csv"]:
            p = next((dd / f for dd in dirs if (dd / f).exists()), out / f)
            if p.exists():
                x = pd.read_csv(p)
                x = x[(x.get("cohort", "cohort_ml") == "cohort_ml") & x.model.isin(["M0", "M1", "M2"])]
                if "threshold" in x:
                    x = x[x.threshold == "bac0"]
                self.pooled = x.drop_duplicates(["tier", "seed", "family", "model"])
                log(f"BAC > 0 pooled reference: {p.name} ({len(self.pooled)} fits)")
                break
        self.premise, self.annual = pd.DataFrame(), pd.DataFrame()
        if ml_csv is not None and not ml_csv.exists() and (out / "bac0_reference" / ml_csv.name).exists():
            ml_csv = out / "bac0_reference" / ml_csv.name
        if ml_csv is not None and ml_csv.exists():
            m = pd.read_csv(ml_csv).rename(columns={"arm": "model"})
            self.premise = m[(m.experiment == "premise") & (m.eval_set == "pooled_test")]
            self.annual = m[m.experiment == "annual"]
            log(f"3_ml_premise reference: {ml_csv.name} ({len(self.premise)} premise, {len(self.annual)} annual fits)")
        if self.pooled.empty and not self.premise.empty:
            self.pooled = self.premise
        p = next((dd / "ml_dagjk_replicates.csv" for dd in dirs if (dd / "ml_dagjk_replicates.csv").exists()),
                 out / "ml_dagjk_replicates.csv")
        self.dagjk = pd.read_csv(p) if p.exists() else pd.DataFrame()
        if len(self.dagjk):
            log(f"BAC > 0 DAGJK replicates: {p.name} ({len(self.dagjk)} rows)")

    @staticmethod
    def _row(x: pd.DataFrame, **key):
        for k, v in key.items():
            x = x[x[k] == v]
        return None if x.empty else x.iloc[0]

    def pooled_metrics(self, tier, seed, fam, model):
        r = self._row(self.pooled, tier=tier, seed=seed, family=fam, model=model) if len(self.pooled) else None
        return None if r is None else {k: float(r[k]) for k in ["brier", "pr_auc", "roc_auc", "log_loss"] if k in r}

    def annual_metrics(self, year, seed, fam, model):
        if self.annual.empty:
            return None
        r = self._row(self.annual, eval_set=f"year_{year}_test", tier="core", seed=seed, family=fam, model=model)
        return None if r is None else {k: float(r[k]) for k in ["brier", "pr_auc", "roc_auc", "log_loss"] if k in r}

    def dagjk_m1m0(self, tier, fam, rep):
        if self.dagjk.empty:
            return None
        r = self._row(self.dagjk, tier=tier, family=fam, rep=rep)
        return None if r is None else float(r["M1_M0"])


# ----------------------------------------------------------------------------- checkpoint helpers
def _ck_load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _ck_append(path: Path, rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def _done(ck: pd.DataFrame, **key) -> bool:
    if ck.empty:
        return False
    x = ck
    for k, v in key.items():
        x = x[x[k] == v]
    return not x.empty


# ----------------------------------------------------------------------------- analyses
def alignment_check(m03, d8, ref: Reference, out: Path, log) -> None:
    """Refit BAC > 0 M2 for one pooled and one annual fit (logistic regression) and compare with the references."""
    rows = []
    d = bac0_view(d8)
    r = ref.pooled_metrics("core", 42, "LR", "M2")
    if r is not None:
        tr_c, _, te_c = m03.split_cases(d, 42)
        _, m = D4.fit_eval(m03, "LR", d[d.CASE_ID.isin(tr_c)], d[d.CASE_ID.isin(te_c)], "core", "M2", 42)
        rows.append(dict(check="pooled M2, core, seed 42, LR", pr_auc_ref=r["pr_auc"], pr_auc_refit=m["pr_auc"]))
    y0 = int(sorted(d.YEAR.unique())[0])
    r = ref.annual_metrics(y0, 42, "LR", "M2")
    if r is not None:
        dy = d[d.YEAR == y0]
        tr_c, _, te_c = m03.split_cases(dy, 42)
        _, m = D4.fit_eval(m03, "LR", dy[dy.CASE_ID.isin(tr_c)], dy[dy.CASE_ID.isin(te_c)], "core", "M2", 42)
        rows.append(dict(check=f"annual M2, {y0}, seed 42, LR", pr_auc_ref=r["pr_auc"], pr_auc_refit=m["pr_auc"]))
    if not rows:
        log("alignment check skipped: no BAC > 0 reference fits found")
        return
    t = pd.DataFrame(rows)
    t["abs_diff"] = (t.pr_auc_ref - t.pr_auc_refit).abs()
    t.to_csv(out / "bac08_alignment_check.csv", index=False)
    log("Alignment check (BAC > 0 refit vs reference):\n" + t.to_string(index=False))
    if (t.abs_diff > 1e-6).any():
        raise RuntimeError("BAC > 0 refits do not reproduce the reference results; stop and check the inputs")


def pooled_block_order(m03, d8, ref: Reference, tiers, families, seeds, out: Path, log) -> None:
    ck = out / "ml_fits_cohort_ml_bac08.csv"
    ckj = out / "ml_eval_jackknife_cohort_ml_bac08.csv"
    done = _ck_load(ck)
    todo = [(t, s, f) for s in seeds for t in tiers for f in families
            if not _done(done, tier=t, seed=s, family=f, model="M2")]
    log(f"pooled BAC >= 0.08: {len(todo)} of {len(tiers) * len(seeds) * len(families)} groups to fit")
    t_start = time.time()
    for i, (tier, seed, fam) in enumerate(todo, 1):
        tr_c, _, te_c = m03.split_cases(d8, seed)
        tr, te = d8[d8.CASE_ID.isin(tr_c)], d8[d8.CASE_ID.isin(te_c)]
        base = dict(pipeline=D4.ML_PIPELINE, cohort="cohort_ml", threshold=THR, tier=tier, seed=seed, family=fam)
        mets, preds, recs = {}, {}, []
        for name in ["M0", "M1"]:
            r = ref.pooled_metrics(tier, seed, fam, name)
            if r is None:
                log(f"  {name} {tier} seed={seed} {fam}: no BAC > 0 result found, refitting")
                preds[name], r = D4.fit_eval(m03, fam, tr, te, tier, name, seed)
                src = "refit"
            else:
                src = "bac0_result"
            mets[name] = r
            recs.append(dict(**base, model=name, source=src, **r))
        for name in RESULT_MODELS:
            t0 = time.time()
            preds[name], mets[name] = D4.fit_eval(m03, fam, tr, te, tier, name, seed)
            recs.append(dict(**base, model=name, source="fit_bac08", **mets[name], n_train=len(tr), n_test=len(te),
                             secs=time.time() - t0))
            log(f"  {tier} seed={seed} {fam} {name} BAC>=0.08 PR-AUC={mets[name]['pr_auc']:.4f} ({time.time() - t0:.0f}s)")
        for key in ["pr_auc", "brier"]:
            sh = D4.shares(mets, key)
            if sh:
                recs.append(dict(**base, model=f"shares_{key}", source="computed", **sh))
        pairs = [p for p in [("M2", "M_AR"), ("M2", "M0")] if p[0] in preds and p[1] in preds]
        jk = [dict(**base, contrast=k, est=e, se_eval_jk=s, jk_df=df_)
              for k, (e, s, df_) in D4.jk_eval(te, preds, pairs).items()]
        _ck_append(ck, recs)
        if jk:
            _ck_append(ckj, jk)
        el = time.time() - t_start
        log(f"  pooled group {i}/{len(todo)} done; about {el / i * (len(todo) - i) / 60:.0f} min left")


def dagjk_bac08(m03, d8, ref: Reference, tiers, families, G, seed, out: Path, log) -> None:
    """Delete-a-group jackknife, PSU groups assigned exactly as run_dagjk in design_based_and_attribution.py."""
    ck = out / "ml_dagjk_replicates_bac08.csv"
    done = _ck_load(ck)
    for tier in tiers:
        d = d8.copy()
        psus = d[["strat", "psu"]].drop_duplicates().sample(frac=1, random_state=seed).sort_values("strat", kind="stable")
        psus["g"] = np.arange(len(psus)) % G
        d["g"] = d.psu.map(dict(zip(psus.psu, psus.g))).values
        tr_c, _, te_c = m03.split_cases(d, seed)
        is_tr, is_te = d.CASE_ID.isin(tr_c).values, d.CASE_ID.isin(te_c).values
        use_saved = _dagjk_grouping_check(m03, d, ref, tier, families, G, seed, is_tr, is_te, out, log)
        for fam in families:
            for rep in [-1] + list(range(G)):
                if _done(done, tier=tier, family=fam, rep=rep):
                    continue
                w = d["w"].values.copy()
                if rep >= 0:
                    w = np.where(d["g"].values == rep, 0.0, w * G / (G - 1))
                dd = d.assign(w_rep=w)
                tr, te = dd[is_tr & (w > 0)], dd[is_te & (w > 0)]
                m0 = D4.fit_eval(m03, fam, tr, te, tier, "M0", seed, wcol="w_rep")[1]
                m2 = D4.fit_eval(m03, fam, tr, te, tier, "M2", seed, wcol="w_rep")[1]
                m1m0 = ref.dagjk_m1m0(tier, fam, rep) if use_saved else None
                src = "bac0_replicate"
                if m1m0 is None:
                    m1 = D4.fit_eval(m03, fam, tr, te, tier, "M1", seed, wcol="w_rep")[1]
                    m1m0, src = m1["pr_auc"] - m0["pr_auc"], "refit"
                row = dict(tier=tier, family=fam, rep=rep, M0=m0["pr_auc"], M2=m2["pr_auc"], M1_M0=m1m0,
                           M2_M0=m2["pr_auc"] - m0["pr_auc"], A_first_share=m1m0 / (m2["pr_auc"] - m0["pr_auc"]),
                           m1m0_source=src, threshold=THR)
                _ck_append(ck, [row])
                log(f"  DAGJK BAC>=0.08 {tier} {fam} rep={rep} M2-M0={row['M2_M0']:.4f} share={row['A_first_share']:.3f}")
    r = pd.read_csv(ck)
    full = r[r.rep == -1]
    for _, x in full.iterrows():
        m0ref = ref.pooled_metrics(x.tier, seed, x.family, "M0")
        if m0ref is not None:
            log(f"  DAGJK full-sample M0 check {x.tier} {x.family}: |refit - reference| = {abs(x.M0 - m0ref['pr_auc']):.2e}")


def _dagjk_grouping_check(m03, d, ref: Reference, tier, families, G, seed, is_tr, is_te, out: Path, log) -> bool:
    """Refit BAC > 0 M0 and M2 in replicate 0 with the first family and compare M2 - M0 with the saved BAC > 0
    replicate. A match shows the PSU groups and fits are the same, so the saved M1 - M0 replicates can be reused."""
    p = out / "dagjk_grouping_check_bac08.csv"
    prev = _ck_load(p)
    if _done(prev, tier=tier):
        return bool(prev[prev.tier == tier].iloc[0]["reuse_saved_m1m0"])
    fam = families[0]
    saved = ref.dagjk
    row = None if saved.empty else Reference._row(saved, tier=tier, family=fam, rep=0)
    if row is None:
        log(f"  DAGJK {tier}: no saved BAC > 0 replicates; M1 is refitted in every replicate")
        ok, diff = False, np.nan
    else:
        w = np.where(d["g"].values == 0, 0.0, d["w"].values * G / (G - 1))
        dd = bac0_view(d).assign(w_rep=w)
        tr, te = dd[is_tr & (w > 0)], dd[is_te & (w > 0)]
        m0 = D4.fit_eval(m03, fam, tr, te, tier, "M0", seed, wcol="w_rep")[1]["pr_auc"]
        m2 = D4.fit_eval(m03, fam, tr, te, tier, "M2", seed, wcol="w_rep")[1]["pr_auc"]
        diff = abs((m2 - m0) - float(row["M2_M0"]))
        ok = diff < 1e-6
        log(f"  DAGJK {tier} grouping check ({fam}, replicate 0, BAC > 0): |M2-M0 refit - saved| = {diff:.2e} "
            f"-> {'reuse saved M1-M0 replicates' if ok else 'mismatch: refit M1 in every replicate'}")
    _ck_append(p, [dict(tier=tier, family=fam, abs_diff=diff, reuse_saved_m1m0=ok)])
    return ok


def dagjk_summary_bac08(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tier, fam), x in r.groupby(["tier", "family"], sort=False):
        full = x[x.rep == -1].iloc[0]
        reps = x[x.rep >= 0]
        G = len(reps)
        for k in ["M2_M0", "A_first_share"]:
            se = float(np.sqrt((G - 1) / G * np.sum((reps[k] - full[k]) ** 2)))
            t = D4.tcrit(G - 1)
            rows.append(dict(cohort="cohort_ml", threshold=THR, tier=tier, family=fam, stat=k, est=full[k],
                             se_dagjk=se, groups=G, lo=full[k] - t * se, hi=full[k] + t * se))
    return pd.DataFrame(rows)


def annual_bac08(m03, d8, ref: Reference, families, seeds, out: Path, log) -> None:
    ck = out / "ml_annual_bac08.csv"
    done = _ck_load(ck)
    years = sorted(int(y) for y in d8.YEAR.unique())
    for y in years:
        dy = d8[d8.YEAR == y].copy()
        for seed in seeds:
            tr_c, _, te_c = m03.split_cases(dy, seed)
            tr, te = dy[dy.CASE_ID.isin(tr_c)], dy[dy.CASE_ID.isin(te_c)]
            for fam in families:
                if _done(done, year=y, seed=seed, family=fam):
                    continue
                m0 = ref.annual_metrics(y, seed, fam, "M0")
                src = "bac0_result"
                if m0 is None:
                    m0, src = D4.fit_eval(m03, fam, tr, te, "core", "M0", seed)[1], "refit"
                t0 = time.time()
                m2 = D4.fit_eval(m03, fam, tr, te, "core", "M2", seed)[1]
                _ck_append(ck, [dict(year=y, seed=seed, family=fam, threshold=THR, m0_source=src,
                                     M0_pr_auc=m0["pr_auc"], M2_pr_auc=m2["pr_auc"], M0_brier=m0["brier"], M2_brier=m2["brier"],
                                     delta_pr_auc=m2["pr_auc"] - m0["pr_auc"], delta_brier_gain=m0["brier"] - m2["brier"])])
                log(f"  annual {y} seed={seed} {fam} M2 BAC>=0.08 PR-AUC={m2['pr_auc']:.4f} ({time.time() - t0:.0f}s)")


def annual_summary(d8: pd.DataFrame, out: Path, log) -> None:
    from scipy.stats import spearmanr
    q = []
    for y, dy in d8.groupby("YEAR"):
        k = dy.R_RESOLVED == 1
        q.append(dict(year=int(y), q_ml=float(np.average(dy.loc[k, "RZ_POSITIVE"], weights=dy.loc[k, "SAMPLE_WEIGHT"]))
                      if k.any() else np.nan, resolved_n=int(k.sum())))
    q = pd.DataFrame(q)
    q.to_csv(out / "annual_q_ml_bac08.csv", index=False)
    a = pd.read_csv(out / "ml_annual_bac08.csv")
    s = a.groupby(["year", "family"]).agg(mean_delta_pr_auc=("delta_pr_auc", "mean"),
                                          mean_delta_brier_gain=("delta_brier_gain", "mean"),
                                          n_seeds=("seed", "nunique")).reset_index().merge(q, on="year")
    s.to_csv(out / "annual_apparent_value_bac08.csv", index=False)
    rows = []
    for fam, g in s.groupby("family"):
        if len(g) >= 3:
            rp, rb = spearmanr(g.q_ml, g.mean_delta_pr_auc), spearmanr(g.q_ml, g.mean_delta_brier_gain)
            rows.append(dict(family=fam, threshold=THR, n_years=len(g), rho_q_pr_gain=rp.statistic, p_q_pr_gain=rp.pvalue,
                             rho_q_brier_gain=rb.statistic, p_q_brier_gain=rb.pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(out / "annual_spearman_bac08.csv", index=False)
    log("Annual Spearman, BAC >= 0.08:\n" + t.round(3).to_string(index=False))


def summarize_pooled(out: Path, log) -> None:
    f = pd.read_csv(out / "ml_fits_cohort_ml_bac08.csv")
    met = f[~f.model.str.startswith("shares")]
    rows = []
    for (tier, fam), g in met.groupby(["tier", "family"], sort=False):
        p = g.pivot_table(index="seed", columns="model", values="pr_auc")
        r = dict(tier=tier, family=fam, threshold=THR, **{f"{m}": p[m].mean() for m in ["M0", "M1", "M_R", "M_AR", "M2"] if m in p})
        r["A_first_share_of_means"] = (r["M1"] - r["M0"]) / (r["M2"] - r["M0"])
        r["A_last_share_of_means"] = (r["M2"] - r["M_R"]) / (r["M2"] - r["M0"])
        r["A_shapley2_share_of_means"] = 0.5 * (r["A_first_share_of_means"] + r["A_last_share_of_means"])
        r["value_only_share_of_means"] = (r["M2"] - r["M_AR"]) / (r["M2"] - r["M0"])
        r["value_only_pr_auc"] = r["M2"] - r["M_AR"]
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "attribution_summary_bac08.csv", index=False)
    log("Block order, BAC >= 0.08 (ratios of five-seed means):\n" + t.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zipdir", type=Path, default=D4.DEFAULT_ZIPDIR)
    ap.add_argument("--out", type=Path, default=D4.DEFAULT_OUT)
    ap.add_argument("--ml-metrics-csv", type=Path, default=D4.DEFAULT_ML_METRICS)
    ap.add_argument("--families", default="lr,xgb,cat")
    ap.add_argument("--tiers", default="core,expanded")
    ap.add_argument("--seeds", default=",".join(map(str, D4.SEEDS)))
    ap.add_argument("--dagjk-groups", type=int, default=15)
    ap.add_argument("--dagjk-families", default="lr,xgb")
    ap.add_argument("--skip", default="", help="comma list of parts to skip: pooled,dagjk,annual")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: logistic regression, core tier, seed 42, 3 DAGJK groups, three years; "
                         "writes to <out>/bac08_quick and leaves the real outputs untouched")
    a = ap.parse_args()
    out = a.out / "bac08_quick" if a.quick else a.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run_log_bac08.txt", "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    if not a.zipdir.exists():
        sys.exit(f"--zipdir folder not found: {a.zipdir.resolve()}")
    fams = [f.strip().upper() for f in a.families.split(",")]
    dfams = [f.strip().upper() for f in a.dagjk_families.split(",")]
    tiers, seeds, G = a.tiers.split(","), [int(s) for s in a.seeds.split(",")], a.dagjk_groups
    years = D4.YEARS
    if a.quick:
        fams, dfams, tiers, seeds, G, years = ["LR"], ["LR"], ["core"], [42], 3, D4.YEARS
    skip = set(filter(None, a.skip.split(",")))
    t0 = time.time()
    log(f"BAC >= 0.08 result-block run started {time.strftime('%Y-%m-%d %H:%M:%S')}; out={out}")

    m03 = D4.load_ml_module()
    d, common = build_cohort_ml(m03, a.zipdir, years, log)
    d8 = frame_bac08(d, log)
    ref = Reference(a.out, a.ml_metrics_csv, log)
    alignment_check(m03, d8, ref, out, log)

    if "pooled" not in skip:
        pooled_block_order(m03, d8, ref, tiers, fams, seeds, out, log)
        summarize_pooled(out, log)
    if "dagjk" not in skip and G > 0:
        dagjk_bac08(m03, d8, ref, tiers, dfams, G, D4.SEEDS[0], out, log)
        dagjk_summary_bac08(pd.read_csv(out / "ml_dagjk_replicates_bac08.csv")).to_csv(
            out / "ml_dagjk_summary_bac08.csv", index=False)
    if "annual" not in skip:
        yrs = sorted(d8.YEAR.unique())[:3] if a.quick else None
        annual_bac08(m03, d8[d8.YEAR.isin(yrs)] if yrs else d8, ref, fams, seeds, out, log)
        annual_summary(d8, out, log)

    manifest = dict(script="bac08_result_block.py", version="0.7", run_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    args={k: str(v) for k, v in vars(a).items()}, python=sys.version, platform=platform.platform(),
                    packages={m: __import__(m).__version__ for m in ["numpy", "pandas", "scipy", "sklearn"]},
                    codemap=D4.CODEMAP, ml_pipeline=D4.ML_PIPELINE, ml_covariate_concepts=common, seeds=seeds,
                    inputs_sha256={f"CRSS{y}CSV.zip": D4.sha256(D4.zip_path(a.zipdir, y)) for y in years})
    for opt in ["xgboost", "catboost"]:
        try:
            manifest["packages"][opt] = __import__(opt).__version__
        except Exception:
            pass
    (out / "run_manifest_bac08.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(f"done in {(time.time() - t0) / 60:.0f} min -> {out}")


if __name__ == "__main__":
    main()
