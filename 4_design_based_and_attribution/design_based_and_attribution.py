#!/usr/bin/env python3
"""
Design-based uncertainty, cohort harmonization, block-order attribution
=======================================================================

Runs every analysis that needs CRSS driver-level microdata together with the
survey design variables:

  * Design-based (delete-one-PSU jackknife) uncertainty for annual prevalences,
    resolved shares, the survivor ceiling, MAS intervals, separation margins and
    the phi_L x gamma_cap frontier; Kish and design effective sample sizes;
    complete annual identification results for both BAC thresholds (BAC>0 and BAC>=0.08).
  * Cohort harmonization (identification audit on the ML cohort and ML premise
    test on the identification cohort); three-level testing status (tested /
    confirmed not tested / status unknown); decomposition of the survivor MAS
    ceiling by unresolved component.
  * Feature-block order sensitivity (A-first, A-last, Shapley over {A} and
    {R, RxZ}; optional three-block Shapley over {A}, {R}, {RxZ}) and the
    value-only contrast M2 - M_AR.
  * Cellwise conditional RFV given coarse partitions of W.
  * Design-based uncertainty for the ML contrasts: evaluation-stage PSU
    jackknife (fitted models fixed) and, optionally, a delete-a-group jackknife
    that refits the models within each replicate.

The ML analyses reuse 3_ml_premise/ml_premise_temporal_annual.py for the data
preparation, covariates, case split, models and metrics, so M0, M1 and M2 on the
ML cohort reproduce its pooled premise results seed by seed; when its
all_model_metrics.csv is available the run verifies this
(ml_reproduction_check.csv).

Outputs (--out, default outputs/4_design_based_and_attribution): CSV files,
Online Resource 1 (online_resource_1.docx, supplementary_tables.md / .xlsx; Tables
S1-S15, rebuilt from the CSV files by build_tables), split_ids_<cohort>.csv.gz, and
run_manifest.json recording versions, arguments, code map and input hashes.
When the gate CSV from 1_identification_gate is available, the identification
quantities are compared with it (gate_reproduction_check.csv).

Input
-----
One of
  --zipdir DIR    CRSS{year}CSV.zip files (default data/crss_zips; --download
                  fetches missing years from NHTSA),
  --drivers PATH  a driver-level parquet/csv with the columns in COLS
                  (rename with --colmap JSON; identification analyses only),
  --raw-dir DIR   unzipped CRSS CSVs in DIR/<year>/ (ACCIDENT, PERSON, VEHICLE).

Coding follows the identification gate (CODEMAP below); the run manifest
records the map used. Design variables follow NHTSA DOT HS 812 688:
STRATA = PSUSTRAT, CLUSTER = PSU_VAR, WEIGHT = WEIGHT; delete-one-PSU jackknife
(JKn) with the with-replacement approximation; domain estimates are computed on
the full sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIPDIR = REPO_ROOT / "data" / "crss_zips"
DEFAULT_GATE_CSV = REPO_ROOT / "outputs" / "1_identification_gate" / "crss_identification_gate.csv"
DEFAULT_OUT = REPO_ROOT / "outputs" / "4_design_based_and_attribution"
YEARS = list(range(2016, 2025))
SEEDS = [42, 1337, 2026, 31415, 271828]
GAMMA = 4.0
PHI_GRID = [0.0, 0.05, 0.10, 0.20, 0.30]

COLS = dict(year="YEAR", case="CASENUM", weight="WEIGHT", strat="PSUSTRAT", psu="PSU_VAR",
            inj="INJ_SEV", alc_status="ALC_STATUS", alc_res="ALC_RES", per_typ="PER_TYP")

# ---- code map (identification gate; CRSS Analytical User's Manual, DOT HS 813 796) ----
CODEMAP = {
    "INJ_SEV_fatal": [4],
    "INJ_SEV_died_prior": [6],
    "INJ_SEV_ml_cohort": [0, 1, 2, 3, 4],          # exact codes 0-4 (ML cohort)
    "ALC_STATUS_tested": [2],
    "ALC_STATUS_not_tested": [0, 1],                # 0 = not given; 1 = refused (2016 only), known A=0 as in the gate
    # every other ALC_STATUS value (e.g. 8 not reported, 9 unknown if tested) -> "unknown"
    "ALC_RES_numeric_max": 940,                     # 0..939 = BAC x 1000, 940 = .94 or greater
    "ALC_RES_positive_no_value": [998],             # resolves BAC>0 = 1 only
    "BAC08_threshold": 80,                          # ALC_RES >= 80  <=> BAC >= 0.08
}

# coarse partitions of W for cellwise conditional RFV (name -> function(df) -> labels)
def _night(df):
    h = pd.to_numeric(df.get("HOUR"), errors="coerce")
    return np.where(h.isna() | (h > 23), "unknown", np.where((h >= 20) | (h <= 5), "night", "day"))

def _agegrp(df):
    a = pd.to_numeric(df.get("AGE"), errors="coerce")
    a = a.where(a < 998)
    return pd.cut(a, [-1, 24, 64, 200], labels=["<25", "25-64", "65+"]).astype(str).replace("nan", "unknown")

PARTITIONS = {
    "night": lambda d: _night(d),
    "age_group": lambda d: _agegrp(d),
    "night_x_age": lambda d: pd.Series(_night(d), index=d.index).astype(str) + "|" + _agegrp(d).astype(str),
}


# ===================================================================== data
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_bytes(raw: bytes) -> pd.DataFrame:
    """Same reader as the gate: try utf-8-sig, utf-8, latin1; upper-case stripped column names."""
    import io
    last = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding=enc)
            df.columns = [str(c).strip().upper() for c in df.columns]
            return df
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise last


def _read_csv(p: Path) -> pd.DataFrame:
    return _read_bytes(p.read_bytes())


def _merge_year(per: pd.DataFrame, acc: pd.DataFrame, veh: pd.DataFrame, y: int) -> pd.DataFrame:
    """Drivers (PER_TYP == 1) with accident- and vehicle-level columns. As in the gate, a WEIGHT already
    in person.csv is kept and checked against accident.csv."""
    for need, df, name in ((["CASENUM", "PER_TYP", "INJ_SEV", "ALC_STATUS", "ALC_RES"], per, "person"),
                           (["CASENUM", "WEIGHT", "PSUSTRAT", "PSU_VAR"], acc, "accident")):
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise KeyError(f"{y}: {name}.csv missing {miss}")
    per = per[pd.to_numeric(per["PER_TYP"], errors="coerce") == 1].copy()
    a1 = acc.drop_duplicates("CASENUM")
    if "WEIGHT" in per.columns:
        chk = per[["CASENUM", "WEIGHT"]].drop_duplicates("CASENUM").merge(
            a1[["CASENUM", "WEIGHT"]].rename(columns={"WEIGHT": "W_ACC"}), on="CASENUM", how="left")
        if not np.allclose(pd.to_numeric(chk["WEIGHT"]), pd.to_numeric(chk["W_ACC"]), rtol=1e-9, atol=1e-9, equal_nan=True):
            raise ValueError(f"{y}: person.csv and accident.csv WEIGHT disagree")
    acc_cols = ["CASENUM"] + [c for c in a1.columns if c not in per.columns]
    per = per.merge(a1[acc_cols], on="CASENUM", how="left", validate="many_to_one")
    if veh is not None and "VEH_NO" in per.columns:
        v1 = veh.drop_duplicates(["CASENUM", "VEH_NO"])
        veh_cols = ["CASENUM", "VEH_NO"] + [c for c in v1.columns if c not in per.columns]
        per = per.merge(v1[veh_cols], on=["CASENUM", "VEH_NO"], how="left", validate="many_to_one")
    if pd.to_numeric(per["WEIGHT"], errors="coerce").isna().any():
        raise ValueError(f"{y}: driver rows missing WEIGHT")
    per[COLS["year"]] = y
    return per


NHTSA_STATIC = "https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip"  # same source as the gate


def download_zip(year: int, dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = NHTSA_STATIC.format(year=year)
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)


def zip_path(zipdir: Path, year: int) -> Path:
    """CRSS{year}CSV.zip in zipdir, its zips/ sub-folder, or any deeper sub-folder.
    Returns zipdir/CRSS{year}CSV.zip when the file is not found anywhere."""
    name = f"CRSS{year}CSV.zip"
    for cand in (zipdir / name, zipdir / "zips" / name):
        if cand.exists():
            return cand
    hits = sorted(zipdir.rglob(name)) if zipdir.exists() else []
    return hits[0] if hits else zipdir / name


def build_from_zips(zipdir: Path, download: bool = False) -> pd.DataFrame:
    """Read the same CRSS{year}CSV.zip files the identification gate uses."""
    import zipfile
    parts = []
    for y in YEARS:
        zp = zip_path(zipdir, y)
        if not zp.exists():
            if not download:
                raise FileNotFoundError(f"{zp} not found; add --download to fetch it from NHTSA")
            download_zip(y, zp)
        elif download and not zipfile.is_zipfile(zp):      # truncated earlier download
            download_zip(y, zp)
        with zipfile.ZipFile(zp) as zf:
            def member(base):
                hits = [n for n in zf.namelist() if Path(n).name.lower() == base]
                return zf.read(hits[0]) if hits else None
            per, acc, veh = (member(b) for b in ("person.csv", "accident.csv", "vehicle.csv"))
            if per is None or acc is None:
                raise FileNotFoundError(f"{zp}: person.csv or accident.csv missing")
            parts.append(_merge_year(_read_bytes(per), _read_bytes(acc),
                                     _read_bytes(veh) if veh is not None else None, y))
            print(f"read {zp.name}: {len(parts[-1]):,} drivers", flush=True)
    return pd.concat(parts, ignore_index=True)


def _find(d: Path, stem: str) -> Path:
    """Find STEM.csv in d or any sub-folder (CRSS zips sometimes unpack into a nested folder)."""
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.stem.upper() == stem and p.suffix.lower() == ".csv":
            return p
    raise FileNotFoundError(f"{stem}.csv not found in {d} or its sub-folders")


def build_from_raw(raw: Path) -> pd.DataFrame:
    parts = []
    for y in YEARS:
        d = raw / str(y)
        parts.append(_merge_year(_read_csv(_find(d, "PERSON")), _read_csv(_find(d, "ACCIDENT")),
                                 _read_csv(_find(d, "VEHICLE")), y))
    return pd.concat(parts, ignore_index=True)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    c = COLS
    out = df.copy()
    inj = pd.to_numeric(out[c["inj"]], errors="coerce")
    out["Y"] = inj.isin(CODEMAP["INJ_SEV_fatal"]).astype(int)
    out["cohort_id"] = ~inj.isin(CODEMAP["INJ_SEV_died_prior"])
    out["cohort_ml"] = inj.isin(CODEMAP["INJ_SEV_ml_cohort"])
    st = pd.to_numeric(out[c["alc_status"]], errors="coerce")
    out["status3"] = np.where(st.isin(CODEMAP["ALC_STATUS_tested"]), "tested",
                              np.where(st.isin(CODEMAP["ALC_STATUS_not_tested"]), "not_tested", "unknown"))
    out["A"] = (out["status3"] == "tested").astype(int)
    res = pd.to_numeric(out[c["alc_res"]], errors="coerce")
    numeric = res.between(0, CODEMAP["ALC_RES_numeric_max"]) & (out["A"] == 1)
    posnov = res.isin(CODEMAP["ALC_RES_positive_no_value"]) & (out["A"] == 1)
    out["R_bac0"] = (numeric | posnov).astype(int)
    out["Z_bac0"] = np.where(out["R_bac0"] == 1, ((numeric & (res > 0)) | posnov).astype(int), np.nan)
    out["R_bac08"] = numeric.astype(int)
    out["Z_bac08"] = np.where(numeric, (res >= CODEMAP["BAC08_threshold"]).astype(int), np.nan)
    out["w"] = pd.to_numeric(out[c["weight"]], errors="coerce").fillna(0.0)
    out["strat"] = out[c["strat"]].astype(str)
    out["psu"] = out[c["strat"]].astype(str) + ":" + out[c["psu"]].astype(str)
    out["case"] = out[c["year"]].astype(str) + ":" + out[c["case"]].astype(str)
    out["year"] = out[c["year"]].astype(int)
    return out


# ===================================================================== design-based jackknife
class JK:
    """Delete-one-PSU jackknife (JKn) for statistics that are smooth functions of
    weighted totals.  Replicate total for deleting PSU j of stratum h:
        T_(hj) = T - T_h + n_h/(n_h-1) * (T_h - T_hj).
    Singleton strata contribute no variance and are counted in .singletons."""

    def __init__(self, strat: pd.Series, psu: pd.Series):
        key = pd.DataFrame({"h": strat.values, "j": psu.values})
        self.psus = key.drop_duplicates().reset_index(drop=True)
        self.psu_index = pd.Index(self.psus["j"])
        self.n_h = self.psus.groupby("h")["j"].transform("count").values
        self.row_psu = self.psu_index.get_indexer(psu.values)
        self.singletons = int((self.psus.groupby("h")["j"].count() == 1).sum())
        self.df = int(len(self.psus) - self.psus["h"].nunique())

    def totals(self, M: np.ndarray):
        """M: rows x k matrix of weighted components. Returns full totals (k,) and
        replicate totals (n_psu, k)."""
        P = np.zeros((len(self.psus), M.shape[1]))
        np.add.at(P, self.row_psu, M)
        T = P.sum(0)
        Th = pd.DataFrame(P).groupby(self.psus["h"].values).transform("sum").values
        n = self.n_h[:, None].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rep = T[None, :] - Th + np.where(n > 1, n / (n - 1), np.nan) * (Th - P)
        return T, rep

    def variance(self, theta: np.ndarray, theta_rep: np.ndarray) -> np.ndarray:
        n = self.n_h.astype(float)
        ok = n > 1
        d = theta_rep[ok] - theta[None, :]
        fac = ((n[ok] - 1) / n[ok])[:, None]
        return np.nansum(fac * d ** 2, axis=0)


def tcrit(df: int) -> float:
    from scipy.stats import t
    return float(t.ppf(0.975, max(df, 1)))


# ===================================================================== identification
COMP = ["N_s", "N_f", "A_s", "A_f", "U_s", "U_f", "NT_s", "NT_f", "R_s", "R_f", "Z_s", "Z_f"]


def comp_matrix(d: pd.DataFrame, thr: str) -> np.ndarray:
    w = d["w"].values
    y = d["Y"].values
    s, f = (y == 0), (y == 1)
    A = d["A"].values == 1
    U = d["status3"].values == "unknown"
    NT = d["status3"].values == "not_tested"
    R = d[f"R_{thr}"].values == 1
    Z = np.nan_to_num(d[f"Z_{thr}"].values) == 1
    cols = [s, f, s & A, f & A, s & U, f & U, s & NT, f & NT, s & R, f & R, s & R & Z, f & R & Z]
    return np.column_stack([w * c for c in cols])


def id_stats(T: np.ndarray) -> dict:
    """T: (..., 12) totals -> dict of arrays."""
    g = {k: T[..., i] for i, k in enumerate(COMP)}
    with np.errstate(divide="ignore", invalid="ignore"):
        N = g["N_s"] + g["N_f"]
        out = dict(pi=g["N_f"] / N,
                   test_s=g["A_s"] / g["N_s"], test_f=g["A_f"] / g["N_f"],
                   unknown_s=g["U_s"] / g["N_s"], unknown_f=g["U_f"] / g["N_f"],
                   test_s_known=g["A_s"] / (g["A_s"] + g["NT_s"]), test_f_known=g["A_f"] / (g["A_f"] + g["NT_f"]),
                   alpha_s=g["R_s"] / g["N_s"], alpha_f=g["R_f"] / g["N_f"],
                   q_s=g["Z_s"] / g["R_s"], q_f=g["Z_f"] / g["R_f"])
        a_s, a_f, q_s, q_f = out["alpha_s"], out["alpha_f"], out["q_s"], out["q_f"]
        out["s_L"] = q_s * (a_s + (1 - a_s) / GAMMA)
        out["s_U"] = q_s
        out["f_L"] = a_f * q_f
        out["f_U"] = a_f * q_f + (1 - a_f)
        out["d_sep"] = out["f_L"] - out["s_U"]
        out["rev_margin"] = out["s_L"] - out["f_U"]
        for ph in PHI_GRID:
            out[f"c_crit_{ph:.2f}"] = (a_f * q_f + (1 - a_f) * ph - a_s * q_s) / (1 - a_s)
        # unresolved survivor components (shares of all survivors)
        tu = out["test_s"] - a_s                         # tested but unresolved
        unk = out["unknown_s"]
        out["share_TU_s"] = tu
        out["share_unknown_s"] = unk
        out["d_sep_TU_free"] = out["d_sep"] - tu * (1 - q_s)
        out["d_sep_TU_unknown_free"] = out["d_sep"] - (tu + unk) * (1 - q_s)
        out["d_sep_TU_unknown_negative"] = out["d_sep"] + (tu + unk) * q_s   # both components alcohol-negative
        out["OG_ceiling_nat"] = out["pi"] * (1 - out["pi"])
    return out


def kish(w: np.ndarray) -> float:
    s = w.sum()
    return float(s * s / (w * w).sum()) if len(w) else 0.0


def identification(df: pd.DataFrame, cohort: str, thr: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ess = [], []
    for y in YEARS:
        d = df[(df.year == y) & df[cohort]]
        if d.empty:
            continue
        jk = JK(d["strat"], d["psu"])
        T, rep = jk.totals(comp_matrix(d, thr))
        est = id_stats(T)
        est_rep = id_stats(rep)
        keys = list(est)
        th = np.array([est[k] for k in keys], float)
        threp = np.column_stack([est_rep[k] for k in keys])
        se = np.sqrt(jk.variance(th, threp))
        tc = tcrit(jk.df)
        for k, v, s in zip(keys, th, se):
            rows.append(dict(year=y, cohort=cohort, threshold=thr, stat=k, est=v, se=s,
                             lo=v - tc * s, hi=v + tc * s, jk_df=jk.df, singleton_strata=jk.singletons,
                             rep_max=np.nanmax(threp[:, keys.index(k)]), rep_min=np.nanmin(threp[:, keys.index(k)])))
        # Kish ESS by cell + design ESS for alpha/q
        R = d[f"R_{thr}"] == 1
        for lab, m in [("survivors", d.Y == 0), ("fatal", d.Y == 1),
                       ("resolved_survivors", (d.Y == 0) & R), ("resolved_fatal", (d.Y == 1) & R)]:
            ess.append(dict(year=y, cohort=cohort, threshold=thr, cell=lab, n_raw=int(m.sum()),
                            weighted=float(d.w[m].sum()), kish_ess=kish(d.w[m].values)))
        for k in ["alpha_s", "alpha_f", "q_s", "q_f"]:
            i = keys.index(k)
            p, v = th[i], se[i] ** 2
            ess.append(dict(year=y, cohort=cohort, threshold=thr, cell=f"design_ess_{k}",
                            n_raw=np.nan, weighted=np.nan, kish_ess=p * (1 - p) / v if v > 0 else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(ess)


GATE_MAP = {  # gate CSV column -> this script's statistic (identification cohort, BAC>0)
    "survivor_test_rate_weighted_all_survivors": "test_s",
    "fatal_test_rate_weighted_all_fatal": "test_f",
    "gt0_survivor_Z_resolved_share_all_survivors": "alpha_s",
    "gt0_fatal_Z_resolved_share_all_fatal": "alpha_f",
    "q_test_BAC_gt0_weighted": "q_s",
    "survivor_unknown_A_weighted_share": "unknown_s",
}


def check_against_gate(idlong: pd.DataFrame, gate_csv: Path, log) -> pd.DataFrame:
    v = pd.read_csv(gate_csv)
    mine = idlong[(idlong.cohort == "cohort_id") & (idlong.threshold == "bac0")].pivot_table(
        index="year", columns="stat", values="est")
    rows = []
    for vcol, stat in GATE_MAP.items():
        if vcol not in v.columns:
            continue
        m = v.set_index("year")[vcol].reindex(mine.index)
        d = (mine[stat] - m).abs()
        rows.append(dict(statistic=stat, gate_column=vcol, max_abs_diff=float(d.max())))
    out = pd.DataFrame(rows)
    worst = out.max_abs_diff.max() if len(out) else np.nan
    log(f"gate reproduction check: max |difference| = {worst:.2e} "
        + ("(MATCH)" if worst < 1e-9 else "(CHECK CODE MAP BEFORE USING RESULTS)"))
    return out


def wide(idlong: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    x = idlong[idlong.stat.isin(stats)].copy()
    x["cell"] = x.apply(lambda r: f"{r.est:.4f}" if pd.isna(r.lo) else f"{r.est:.4f} [{r.lo:.4f}, {r.hi:.4f}]", axis=1)
    return x.pivot_table(index=["year"], columns="stat", values="cell", aggfunc="first")[stats]


# ===================================================================== conditional RFV
def og(s, f, pi):
    m1 = pi * f + (1 - pi) * s
    m0 = 1 - m1
    return 0.0 if (m1 <= 0 or m0 <= 0) else (pi * (1 - pi)) ** 2 * (f - s) ** 2 / (m1 * m0)


def rfv_rect(sL, sU, fL, fU, pi):
    if fL > sU:
        return og(sU, fL, pi)
    if fU < sL:
        return og(sL, fU, pi)
    return 0.0


def conditional_rfv(df: pd.DataFrame, cohort: str, thr: str) -> pd.DataFrame:
    rows = []
    for y in YEARS:
        d = df[(df.year == y) & df[cohort]]
        if d.empty:
            continue
        Ntot = d.w.sum()
        for name, fn in PARTITIONS.items():
            lab = pd.Series(np.asarray(fn(d)), index=d.index)
            total, nsep, nnores, ncell = 0.0, 0, 0, 0
            pi_all = d.w[d.Y == 1].sum() / Ntot
            for cell, dc in d.groupby(lab):
                ncell += 1
                w = dc.w
                Ns, Nf = w[dc.Y == 0].sum(), w[dc.Y == 1].sum()
                if Ns <= 0 or Nf <= 0:
                    continue  # degenerate outcome within cell: Z cannot add Brier value
                R = dc[f"R_{thr}"] == 1
                Z = dc[f"Z_{thr}"] == 1
                a_s = w[(dc.Y == 0) & R].sum() / Ns
                a_f = w[(dc.Y == 1) & R].sum() / Nf
                q_f = w[(dc.Y == 1) & R & Z].sum() / max(w[(dc.Y == 1) & R].sum(), 1e-300)
                if a_s > 0:
                    q_s = w[(dc.Y == 0) & R & Z].sum() / w[(dc.Y == 0) & R].sum()
                    sL, sU = q_s * (a_s + (1 - a_s) / GAMMA), q_s
                else:  # no resolved survivors: MAS ceiling undefined -> unrestricted
                    nnores += 1
                    sL, sU = 0.0, 1.0
                fL, fU = a_f * q_f, a_f * q_f + (1 - a_f)
                piw = Nf / (Ns + Nf)
                v = rfv_rect(sL, sU, fL, fU, piw)
                nsep += v > 0
                total += (Ns + Nf) / Ntot * v
            rows.append(dict(year=y, cohort=cohort, threshold=thr, partition=name, cells=ncell,
                             separated_cells=nsep, cells_no_resolved_survivors=nnores,
                             conditional_RFV=total, conditional_RFV_over_Rdel=total / (pi_all * (1 - pi_all))))
    return pd.DataFrame(rows)


# ===================================================================== ML
# The ML analyses reuse 3_ml_premise/ml_premise_temporal_annual.py for the data
# preparation, covariates W, case split, models and metrics, so M0, M1 and M2 on
# the ML cohort reproduce its pooled premise results seed by seed.
ML_PIPELINE = "ml_premise_v1"   # tag written to the checkpoints; rows from other pipelines are ignored
FEATURE_SETS = {                # columns added to W (M0, M1, M2 come from ml_premise feature_sets)
    "M0": [], "M1": ["A_TEST"], "M2": ["A_TEST", "R_RESOLVED", "RZ_POSITIVE"],
    "M1_3": ["S3_TESTED", "S3_UNKNOWN"], "M_R": ["R_RESOLVED", "RZ_POSITIVE"],
    "M_AR": ["A_TEST", "R_RESOLVED"], "M2_3": ["S3_TESTED", "S3_UNKNOWN", "R_RESOLVED", "RZ_POSITIVE"],
}
THREE_BLOCK = {"S_R": ["R_RESOLVED"], "S_RZ": ["RZ_POSITIVE"], "S_A_RZ": ["A_TEST", "RZ_POSITIVE"]}
DESIGN_EXTRA = ["ALC_STATUS_RAW", "ALC_RES_RAW", "INJ_SEV_RAW", "DSGN_STRAT", "DSGN_PSU"]
DEFAULT_ML_METRICS = REPO_ROOT / "outputs" / "3_ml_premise" / "all_model_metrics.csv"


def load_ml_module():
    """Import 3_ml_premise/ml_premise_temporal_annual.py."""
    import importlib.util
    path = REPO_ROOT / "3_ml_premise" / "ml_premise_temporal_annual.py"
    spec = importlib.util.spec_from_file_location("ml_premise_temporal_annual", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_ml_frames(m03, zipdir: Path | None, raw_dir: Path | None, log) -> tuple[dict, list[str]]:
    """Pooled 2016-2024 driver frames built exactly as in 3_ml_premise.

    cohort_ml: INJ_SEV 0-4 (the ML premise cohort); cohort_id: every driver except
    those who died before the crash (the identification cohort). Both carry the
    design variables and the three-level testing status."""
    frames = {"cohort_ml": {}, "cohort_id": {}}
    for y in YEARS:
        if zipdir is not None:
            per, acc, veh = m03.read_year_tables(zip_path(zipdir, y))
        else:
            d = raw_dir / str(y)
            per, acc, veh = (_read_csv(_find(d, t)) for t in ("PERSON", "ACCIDENT", "VEHICLE"))
        frames["cohort_ml"][y] = m03.build_driver_frame(y, per, acc, veh, design=True)
        frames["cohort_id"][y] = m03.build_driver_frame(y, per, acc, veh, injury_codes=None, design=True)
    out, commons = {}, {}
    for coh, fr in frames.items():
        d, commons[coh] = m03.pool_years(fr, extra=DESIGN_EXTRA)
        st = d["ALC_STATUS_RAW"]
        d["S3_TESTED"] = st.isin(CODEMAP["ALC_STATUS_tested"]).astype(int)
        d["S3_UNKNOWN"] = (~st.isin(CODEMAP["ALC_STATUS_tested"] + CODEMAP["ALC_STATUS_not_tested"])).astype(int)
        d["Y"] = d["Y_FATAL"].astype(int)
        d["w"] = d["SAMPLE_WEIGHT"].astype(float)
        d["strat"] = d["DSGN_STRAT"].astype(str)
        d["psu"] = d["DSGN_STRAT"].astype(str) + ":" + d["DSGN_PSU"].astype(str)
        d["case"] = d["CASE_ID"]
        out[coh] = d
        log(f"ML frame {coh}: {len(d):,} drivers, {d.CASE_ID.nunique():,} cases, {len(commons[coh])} covariate concepts")
    if commons["cohort_ml"] != commons["cohort_id"]:
        raise RuntimeError("covariate concepts differ between the ML and identification cohorts")
    return out, commons["cohort_ml"]


def arm_features(m03, df: pd.DataFrame, tier: str, arm: str) -> tuple[list[str], list[str]]:
    """(numeric, categorical) columns for an arm; extra binary blocks are numeric, as A_TEST is in 3_ml_premise."""
    if arm in ("M0", "M1", "M2"):
        return m03.feature_sets(df, tier, arm)
    num, cat = m03.feature_sets(df, tier, "M0")
    return num + (FEATURE_SETS.get(arm) or THREE_BLOCK[arm]), cat


def fit_eval(m03, family, tr, te, tier, arm, seed, wcol="w"):
    """Fit one arm with 3_ml_premise.fit_model and return clipped test predictions and metrics."""
    num, cat = arm_features(m03, tr, tier, arm)
    model = m03.fit_model(family, tr[num + cat], tr["Y"].to_numpy(int), tr[wcol].to_numpy(float), num, cat, seed)
    p = np.clip(np.asarray(m03.predict_model(model, family, te[num + cat]), dtype=float), 1e-8, 1 - 1e-8)
    return p, m03.metrics(te["Y"].to_numpy(int), p, te[wcol].to_numpy(float))


def jk_eval(d_te: pd.DataFrame, preds: dict, pairs: list[tuple[str, str]]):
    """Evaluation-stage PSU jackknife of metric contrasts with fitted models fixed."""
    from sklearn.metrics import average_precision_score
    jk = JK(d_te["strat"], d_te["psu"])
    y, w = d_te["Y"].values, d_te["w"].values
    out = {}
    for a, b in pairs:
        full = average_precision_score(y, preds[a], sample_weight=w) - average_precision_score(y, preds[b], sample_weight=w)
        reps = []
        for (h, j), n in zip(jk.psus.itertuples(index=False, name=None), jk.n_h):
            if n <= 1:
                reps.append(full); continue
            in_h = (d_te["strat"].values == h)
            wr = w.copy()
            wr[in_h] *= n / (n - 1)
            wr[d_te["psu"].values == j] = 0.0
            reps.append(average_precision_score(y, preds[a], sample_weight=wr)
                        - average_precision_score(y, preds[b], sample_weight=wr))
        reps = np.array(reps)[:, None]
        se = float(np.sqrt(jk.variance(np.array([full]), reps))[0])
        out[f"{a}-{b}"] = (full, se, jk.df)
    return out


def shares(m: dict, key="pr_auc") -> dict:
    if not all(k in m for k in ("M0", "M1", "M2", "M_R", "M_AR")):
        return {}
    g = lambda k: m[k][key] if key == "pr_auc" else -m[k][key]  # Brier: gain = reduction
    tot = g("M2") - g("M0")
    r = dict(total_gain=tot,
             A_first=(g("M1") - g("M0")) / tot,
             A_last=(g("M2") - g("M_R")) / tot,
             value_only=(g("M2") - g("M_AR")) / tot)
    r["A_shapley2"] = 0.5 * (r["A_first"] + r["A_last"])
    if all(k in m for k in THREE_BLOCK):
        v = {(): g("M0"), ("A",): g("M1"), ("R",): g("S_R"), ("RZ",): g("S_RZ"), ("A", "R"): g("M_AR"),
             ("A", "RZ"): g("S_A_RZ"), ("R", "RZ"): g("M_R"), ("A", "R", "RZ"): g("M2")}
        import itertools, math
        P = ["A", "R", "RZ"]
        for p in P:
            others = [q for q in P if q != p]
            s = 0.0
            for k in range(3):
                for S in itertools.combinations(others, k):
                    S1 = tuple(x for x in P if x in S or x == p)
                    S0 = tuple(x for x in P if x in S)
                    s += math.factorial(k) * math.factorial(2 - k) / math.factorial(3) * (v[S1] - v[S0])
            r[f"shapley3_{p}"] = s / tot
    return r


ATTRIBUTION_SETS = {"cohort_ml": ["M0", "M1", "M1_3", "M_R", "M_AR", "M2"],   # block order, value-only, 3-level status
                    "cohort_id": ["M0", "M1"]}                                 # cohort harmonization


def write_split_ids(m03, d: pd.DataFrame, cohort: str, seeds, out: Path) -> None:
    """Case-level train/val/test assignment for every seed (one column per seed).
    For cohort_ml this is the split of the 3_ml_premise pooled premise experiment."""
    cases = pd.DataFrame({"CASE_ID": sorted(d["CASE_ID"].unique())})
    for seed in seeds:
        tr, va, te = m03.split_cases(d, seed)
        lab = {**{c: "train" for c in tr}, **{c: "val" for c in va}, **{c: "test" for c in te}}
        cases[f"seed_{seed}"] = cases["CASE_ID"].map(lab)
    cases.to_csv(out / f"split_ids_{cohort}.csv.gz", index=False, compression="gzip")


def _replace_group(path: Path, new: pd.DataFrame, key: tuple) -> None:
    """Replace the rows of one (tier, seed, family) group in a checkpoint file."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old.pipeline == ML_PIPELINE) & (old.tier == key[0]) & (old.seed == key[1]) & (old.family == key[2]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def _load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    x = pd.read_csv(path)
    return x[x.pipeline == ML_PIPELINE] if "pipeline" in x else pd.DataFrame()


def run_ml(m03, d, cohort, tiers, families, seeds, three_block, out: Path, log, only=None):
    """Fit the feature sets for every tier x seed x family on the 3_ml_premise case split.
    Each completed (tier, seed, family) group is saved to ml_fits_<cohort>.csv, so an
    interrupted run resumes where it stopped; a group is refitted when it lacks a requested model."""
    sets = dict(FEATURE_SETS)
    if three_block:
        sets.update(THREE_BLOCK)
    if only:
        sets = {k: v for k, v in sets.items() if k in only}
    ck_fit = out / f"ml_fits_{cohort}.csv"
    ck_jk = out / f"ml_eval_jackknife_{cohort}.csv"
    prev = _load_checkpoint(ck_fit)
    have = {}
    if len(prev):
        for k, g in prev[~prev.model.str.startswith("shares")].groupby(["tier", "seed", "family"]):
            have[k] = set(g.model)
    groups = [(t, sd, f) for t in tiers for sd in seeds for f in families]
    todo = [g for g in groups if not set(sets) <= have.get(g, set())]
    log(f"{cohort}: {len(groups)} groups x {len(sets)} models; {len(groups) - len(todo)} already done (checkpoint)")
    t_start, n_done = time.time(), 0
    for seed in seeds:
        if not any(g[1] == seed for g in todo):
            continue
        tr_c, _, te_c = m03.split_cases(d, seed)
        tr, te = d[d.CASE_ID.isin(tr_c)].copy(), d[d.CASE_ID.isin(te_c)].copy()
        for tier in tiers:
            for fam in families:
                if (tier, seed, fam) not in todo:
                    continue
                recs, jkrecs, preds, mets = [], [], {}, {}
                base = dict(pipeline=ML_PIPELINE, cohort=cohort, threshold="bac0", tier=tier, seed=seed, family=fam)
                for name in sets:
                    t0 = time.time()
                    preds[name], mets[name] = fit_eval(m03, fam, tr, te, tier, name, seed)
                    recs.append(dict(**base, model=name, **mets[name], n_train=len(tr), n_test=len(te),
                                     secs=time.time() - t0))
                    log(f"  {cohort} {tier} seed={seed} {fam} {name} PR-AUC={mets[name]['pr_auc']:.4f} "
                        f"({time.time() - t0:.0f}s)")
                for key in ["pr_auc", "brier"]:
                    sh = shares(mets, key)
                    if sh:
                        recs.append(dict(**base, model=f"shares_{key}", **sh))
                pairs = [pr for pr in [("M1", "M0"), ("M1_3", "M0"), ("M2", "M0"), ("M2", "M_AR")]
                         if pr[0] in preds and pr[1] in preds]
                for k, (est, se, dfree) in jk_eval(te, preds, pairs).items():
                    jkrecs.append(dict(**base, contrast=k, est=est, se_eval_jk=se, jk_df=dfree))
                _replace_group(ck_fit, pd.DataFrame(recs), (tier, seed, fam))
                if jkrecs:
                    _replace_group(ck_jk, pd.DataFrame(jkrecs), (tier, seed, fam))
                n_done += 1
                el = time.time() - t_start
                log(f"  group {n_done}/{len(todo)} done for {cohort}; elapsed {el / 60:.0f} min, "
                    f"about {el / n_done * (len(todo) - n_done) / 60:.0f} min left for this cohort")
    fits, jks = _load_checkpoint(ck_fit), _load_checkpoint(ck_jk)
    keep = lambda x: x[x.tier.isin(tiers) & x.seed.isin(seeds) & x.family.isin(families)] if len(x) else x
    return keep(fits), keep(jks)


def check_against_ml_premise(fits: pd.DataFrame, ml_csv: Path, log) -> pd.DataFrame:
    """Compare M0, M1, M2 on the ML cohort with the 3_ml_premise pooled premise results."""
    ref = pd.read_csv(ml_csv)
    ref = ref[(ref.experiment == "premise") & (ref.eval_set == "pooled_test")]
    mine = fits[(fits.cohort == "cohort_ml") & fits.model.isin(["M0", "M1", "M2"])]
    m = mine.merge(ref.rename(columns={"arm": "model"}), on=["tier", "family", "seed", "model"], suffixes=("", "_ref"))
    rows = [dict(metric=k, n_compared=len(m), max_abs_diff=float((m[k] - m[f"{k}_ref"]).abs().max()) if len(m) else np.nan)
            for k in ["pr_auc", "brier", "roc_auc", "log_loss"] if k in m and f"{k}_ref" in m]
    out = pd.DataFrame(rows)
    worst = out.max_abs_diff.max() if len(out) else np.nan
    verdict = "MATCH" if worst < 1e-9 else ("CLOSE (library or thread differences)" if worst < 1e-4 else "CHECK")
    log(f"ML reproduction check against {ml_csv.name}: {len(m)} fits compared, max |difference| = {worst:.2e} ({verdict})")
    return out


def attribution_summary(mls: pd.DataFrame, out: Path, log) -> pd.DataFrame:
    """Testing-status, cohort and block-order contrasts per tier x family, averaged over seeds (PR-AUC)."""
    met = mls[~mls.model.astype(str).str.startswith("shares")]
    pr = met.pivot_table(index=["cohort", "tier", "family", "seed"], columns="model", values="pr_auc")
    rows = []
    for (tier, fam), g in pr.loc["cohort_ml"].groupby(level=["tier", "family"]):
        r = dict(tier=tier, family=fam,
                 M0=g["M0"].mean(), M1=g["M1"].mean(), M2=g["M2"].mean(),
                 M1_minus_M0=(g["M1"] - g["M0"]).mean())
        r["A_first_share_of_means"] = r["M1_minus_M0"] / (r["M2"] - r["M0"])   # as in Table 7
        if "M1_3" in g:
            r["M1_3_minus_M0"] = (g["M1_3"] - g["M0"]).mean()
            r["three_level_change"] = r["M1_3_minus_M0"] - r["M1_minus_M0"]
        gi = met[(met.cohort == "cohort_id") & (met.tier == tier) & (met.family == fam)].pivot_table(
            index="seed", columns="model", values="pr_auc")
        if {"M0", "M1"} <= set(gi.columns):
            r["M1_minus_M0_id_cohort"] = (gi["M1"] - gi["M0"]).mean()
            r["cohort_change"] = r["M1_minus_M0_id_cohort"] - r["M1_minus_M0"]
        sh = mls[(mls.model == "shares_pr_auc") & (mls.cohort == "cohort_ml") & (mls.tier == tier) & (mls.family == fam)]
        for c in ["A_first", "A_last", "A_shapley2", "value_only"]:
            if c in sh:
                r[f"{c}_share"] = sh[c].mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "attribution_summary.csv", index=False)
    log("Attribution summary (PR-AUC, mean over seeds):\n" + t.round(4).to_string(index=False))
    return t


def run_dagjk(m03, d, cohort, tier, families, G, seed, out, log):
    """Delete-a-group jackknife with refitting (full design-based propagation for
    the M1-M0 and M2-M0 PR-AUC contrasts). PSUs are assigned to G groups
    systematically within strata after a random sort. The full-sample fit (rep -1)
    is the seed's pooled premise fit."""
    d = d.copy()
    psus = d[["strat", "psu"]].drop_duplicates().sample(frac=1, random_state=seed).sort_values("strat", kind="stable")
    psus["g"] = np.arange(len(psus)) % G
    d["g"] = d.psu.map(dict(zip(psus.psu, psus.g))).values
    tr_c, _, te_c = m03.split_cases(d, seed)
    is_tr, is_te = d.CASE_ID.isin(tr_c).values, d.CASE_ID.isin(te_c).values
    rows = []
    for fam in families:
        for rep in [-1] + list(range(G)):
            w = d["w"].values.copy()
            if rep >= 0:
                w = np.where(d["g"].values == rep, 0.0, w * G / (G - 1))
            dd = d.assign(w_rep=w)
            tr, te = dd[is_tr & (w > 0)], dd[is_te & (w > 0)]
            m = {name: fit_eval(m03, fam, tr, te, tier, name, seed, wcol="w_rep")[1]["pr_auc"]
                 for name in ["M0", "M1", "M2"]}
            rows.append(dict(family=fam, rep=rep, M1_M0=m["M1"] - m["M0"], M2_M0=m["M2"] - m["M0"],
                             A_first_share=(m["M1"] - m["M0"]) / (m["M2"] - m["M0"])))
            log(f"  DAGJK {tier} {fam} rep={rep} M1-M0={m['M1'] - m['M0']:.4f}")
    r = pd.DataFrame(rows)
    return r, dagjk_summary(r, cohort, tier)


def dagjk_summary(r: pd.DataFrame, cohort: str, tier: str) -> pd.DataFrame:
    """Jackknife SE and t-based 95% interval from the delete-a-group replicates (rep -1 = full sample)."""
    summ = []
    for fam, x in r.groupby("family"):
        full = x[x.rep == -1].iloc[0]
        reps = x[x.rep >= 0]
        G = len(reps)
        for k in ["M1_M0", "M2_M0", "A_first_share"]:
            var = (G - 1) / G * np.sum((reps[k] - full[k]) ** 2)
            summ.append(dict(cohort=cohort, threshold="bac0", tier=tier, family=fam, stat=k, est=full[k],
                             se_dagjk=np.sqrt(var), groups=G, lo=full[k] - tcrit(G - 1) * np.sqrt(var),
                             hi=full[k] + tcrit(G - 1) * np.sqrt(var)))
    return pd.DataFrame(summ)


# ===================================================================== reporting
def overlap_classification(idl: pd.DataFrame) -> pd.DataFrame:
    cls = idl[idl.stat.isin(["d_sep", "rev_margin"])].pivot_table(
        index=["cohort", "threshold", "year"], columns="stat", values=["hi", "rep_max"])
    cls.columns = [f"{a_}_{b}" for a_, b in cls.columns]
    cls["overlap_robust_95"] = (cls["hi_d_sep"] < 0) & (cls["hi_rev_margin"] < 0)
    cls["overlap_all_replicates"] = (cls["rep_max_d_sep"] < 0) & (cls["rep_max_rev_margin"] < 0)
    return cls


S4_STATS = ["unknown_s", "unknown_f", "test_s_known", "test_f_known", "share_TU_s",
            "d_sep_TU_free", "d_sep_TU_unknown_free", "d_sep_TU_unknown_negative"]


def build_tables(out: Path) -> dict[str, pd.DataFrame]:
    """Supplementary tables (Online Resource 1) from the CSV outputs in `out`.

    S1-S7 (a: BAC>0, b: BAC>=0.08) identification; S8 classification robustness;
    S9-S13 machine learning (when those outputs exist); S14-S15 reproduction checks."""
    idl = pd.read_csv(out / "identification_jackknife_long.csv")
    essl = pd.read_csv(out / "effective_sample_sizes.csv")
    cond = pd.read_csv(out / "conditional_rfv_cells.csv")
    tables = {}
    for thr in [t for t in ("bac0", "bac08") if t in set(idl.threshold)]:
        x = idl[(idl.cohort == "cohort_id") & (idl.threshold == thr)]
        if "d_sep_TU_unknown_negative" not in set(x.stat):   # outputs written before this statistic existed
            pv = x.pivot_table(index="year", columns="stat", values="est")
            extra = (pv.d_sep + (pv.share_TU_s + pv.share_unknown_s) * pv.q_s).rename("est").reset_index()
            x = pd.concat([x, extra.assign(stat="d_sep_TU_unknown_negative", lo=np.nan, hi=np.nan,
                                           cohort="cohort_id", threshold=thr)], ignore_index=True)
        tables[f"S1_prevalence_{thr}"] = wide(x, ["pi", "test_s", "test_f", "alpha_s", "alpha_f", "q_s", "q_f"])
        tables[f"S2_intervals_{thr}"] = wide(x, ["s_L", "s_U", "f_L", "f_U", "d_sep", "rev_margin"])
        tables[f"S3_frontier_{thr}"] = wide(x, [f"c_crit_{p:.2f}" for p in PHI_GRID])
        tables[f"S4_status_{thr}"] = wide(x, S4_STATS)
        # cohort harmonization: identification on the ML cohort, difference vs the identification cohort
        a_ = idl[(idl.threshold == thr)].pivot_table(index=["year", "stat"], columns="cohort", values="est")
        a_["diff_ml_minus_id"] = a_["cohort_ml"] - a_["cohort_id"]
        keep = ["pi", "alpha_s", "q_s", "alpha_f", "q_f", "d_sep", "rev_margin"]
        tables[f"S5_cohort_harmonization_{thr}"] = a_.reset_index().query("stat in @keep").pivot(
            index="year", columns="stat", values="diff_ml_minus_id")[keep]
        e = essl[(essl.cohort == "cohort_id") & (essl.threshold == thr)]
        tables[f"S6_ess_{thr}"] = e.pivot_table(index="year", columns="cell", values="kish_ess").round(1)
        c = cond[(cond.cohort == "cohort_id") & (cond.threshold == thr)]
        tables[f"S7_conditional_rfv_{thr}"] = c.set_index(["year", "partition"])[
            ["cells", "separated_cells", "cells_no_resolved_survivors", "conditional_RFV", "conditional_RFV_over_Rdel"]]
    tables["S8_overlap_classification_robustness"] = overlap_classification(idl)

    if (out / "ml_block_order_results.csv").exists():
        mls = pd.read_csv(out / "ml_block_order_results.csv")
        met = mls[~mls.model.str.startswith("shares")]
        tables["S9_ml_prauc_by_model"] = met.pivot_table(index=["cohort", "tier", "family"], columns="model",
                                                         values="pr_auc", aggfunc="mean").round(4)
        def _flat(t):
            t.columns = [f"{a_}_{b}" for a_, b in t.columns]
            return t
        for key, name in [("shares_pr_auc", "S10_block_order_shares_prauc"), ("shares_brier", "S11_block_order_shares_brier")]:
            sh = mls[mls.model == key]
            cols = [c for c in ["total_gain", "A_first", "A_last", "A_shapley2", "value_only",
                                "shapley3_A", "shapley3_R", "shapley3_RZ"] if c in sh and sh[c].notna().any()]
            tables[name] = _flat(sh.groupby(["cohort", "tier", "family"])[cols].agg(["mean", "std"]))
    if (out / "ml_eval_jackknife.csv").exists():
        jks = pd.read_csv(out / "ml_eval_jackknife.csv")
        tables["S12_ml_design_eval_uncertainty"] = jks.groupby(["cohort", "tier", "family", "contrast"]).agg(
            est=("est", "mean"), se_eval_jk=("se_eval_jk", "mean"), seed_sd=("est", "std")).round(5)
    if (out / "ml_dagjk_replicates.csv").exists():
        r = pd.read_csv(out / "ml_dagjk_replicates.csv")
        tables["S13_ml_dagjk_refit"] = pd.concat([dagjk_summary(g, "cohort_ml", t) for t, g in r.groupby("tier", sort=False)]
                                                 ).set_index(["tier", "family", "stat"]).round(5)
    if (out / "gate_reproduction_check.csv").exists():
        tables["S14_gate_reproduction_check"] = pd.read_csv(out / "gate_reproduction_check.csv").set_index("statistic")
    if (out / "ml_reproduction_check.csv").exists():
        tables["S15_ml_reproduction_check"] = pd.read_csv(out / "ml_reproduction_check.csv").set_index("metric")
    return tables


def write_resource(out: Path, tables: dict[str, pd.DataFrame]):
    md = ["# Supplementary tables (generated by design_based_and_attribution.py)\n"]
    for name, t in tables.items():
        md.append(f"\n## {name}\n")
        md.append(t.to_markdown() if hasattr(t, "to_markdown") else t.to_string())
    try:
        (out / "supplementary_tables.md").write_text("\n".join(md), encoding="utf-8")
    except Exception:
        (out / "supplementary_tables.md").write_text("\n\n".join(f"## {k}\n{v.to_string()}" for k, v in tables.items()), encoding="utf-8")
    try:
        with pd.ExcelWriter(out / "supplementary_tables.xlsx") as xw:
            for name, t in tables.items():
                t.to_excel(xw, sheet_name=name[:31])
    except Exception as e:  # openpyxl missing
        print("xlsx not written:", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--zipdir", type=Path, default=None,
                     help="folder with CRSS{year}CSV.zip, directly or in a sub-folder (default: data/crss_zips)")
    src.add_argument("--drivers", type=Path, help="a ready driver-level parquet/csv file")
    src.add_argument("--raw-dir", type=Path, help="unzipped CRSS CSVs in <dir>/<year>/")
    ap.add_argument("--download", action="store_true",
                    help="download any missing CRSS{year}CSV.zip from NHTSA into --zipdir")
    ap.add_argument("--gate-csv", type=Path, default=DEFAULT_GATE_CSV,
                    help="gate CSV used to verify that the identification numbers match "
                         "(default: outputs/1_identification_gate/crss_identification_gate.csv)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output folder (default: outputs/4_design_based_and_attribution)")
    ap.add_argument("--colmap", type=Path, help="JSON {standard_name: your_column} renames")
    ap.add_argument("--ml", action="store_true", help="run ML block-order / 3-level status / eval-JK analyses")
    ap.add_argument("--attribution", action="store_true",
                    help="preset: only the models needed for testing-status, cohort and block-order results "
                         "(ML cohort: M0, M1, M1_3, M_R, M_AR, M2; identification cohort: M0, M1), "
                         "all tiers/families/seeds, with checkpoint/resume")
    ap.add_argument("--ml-metrics-csv", type=Path, default=DEFAULT_ML_METRICS,
                    help="all_model_metrics.csv from 3_ml_premise, used to verify that M0/M1/M2 reproduce its "
                         "pooled premise results (default: outputs/3_ml_premise/all_model_metrics.csv)")
    ap.add_argument("--families", default="lr,xgb,cat")
    ap.add_argument("--tiers", default="core,expanded")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--three-block", action="store_true")
    ap.add_argument("--dagjk-groups", type=int, default=0, help=">0 runs refitting delete-a-group jackknife")
    ap.add_argument("--dagjk-families", default="lr,xgb")
    ap.add_argument("--thresholds", default="bac0,bac08")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "run_log.txt", "a", encoding="utf-8")
    log = lambda s: (print(s), logf.write(s + "\n"), logf.flush())

    if not (a.drivers or a.raw_dir or a.zipdir):
        a.zipdir = DEFAULT_ZIPDIR
    if a.zipdir and not a.zipdir.exists() and not a.download:
        sys.exit(f"--zipdir folder not found: {a.zipdir.resolve()}\nPoint it at the folder holding CRSS2016CSV.zip ... "
                 "CRSS2024CSV.zip, or add --download to fetch them from NHTSA into that folder.")
    if (a.ml or a.attribution) and a.drivers:
        sys.exit("The ML analyses rebuild the 3_ml_premise driver frames from the CRSS files; "
                 "use --zipdir or --raw-dir instead of --drivers.")
    if a.drivers and not a.drivers.exists():
        sys.exit(f"--drivers file not found: {a.drivers.resolve()}\n"
                 "Point --drivers at a driver-level file, or use --zipdir / --raw-dir with the CRSS files.")
    if a.raw_dir and not a.raw_dir.exists():
        sys.exit(f"--raw-dir folder not found: {a.raw_dir.resolve()}")
    if a.drivers:
        raw = pd.read_parquet(a.drivers) if a.drivers.suffix == ".parquet" else pd.read_csv(a.drivers, low_memory=False)
        inputs = {a.drivers.name: sha256(a.drivers)}
    elif a.zipdir:
        raw = build_from_zips(a.zipdir, a.download)
        inputs = {f"CRSS{y}CSV.zip": sha256(zip_path(a.zipdir, y)) for y in YEARS}
    else:
        raw = build_from_raw(a.raw_dir)
        inputs = {str(p.relative_to(a.raw_dir)): sha256(p) for p in sorted(a.raw_dir.rglob("*.csv")) if p.stem.upper() in ("ACCIDENT", "PERSON", "VEHICLE")}
    if a.colmap:
        raw = raw.rename(columns={v: k for k, v in json.loads(a.colmap.read_text()).items()})
    df = derive(raw)
    thresholds = a.thresholds.split(",")
    log(f"drivers={len(df):,} years={sorted(df.year.unique())}")

    # ---------------- identification + ESS + decomposition (both cohorts, both thresholds)
    idl, essl, cond = [], [], []
    for thr in thresholds:
        for coh in ["cohort_id", "cohort_ml"]:
            t, e = identification(df, coh, thr)
            idl.append(t); essl.append(e)
            cond.append(conditional_rfv(df, coh, thr))
            log(f"identification done: {coh} {thr}")
    idl, essl, cond = pd.concat(idl), pd.concat(essl), pd.concat(cond)
    idl.to_csv(a.out / "identification_jackknife_long.csv", index=False)
    essl.to_csv(a.out / "effective_sample_sizes.csv", index=False)
    cond.to_csv(a.out / "conditional_rfv_cells.csv", index=False)

    log(f"data source: {a.zipdir or a.drivers or a.raw_dir}; gate check file: {a.gate_csv}")
    if a.gate_csv and a.gate_csv.exists() and "bac0" in thresholds:
        chk = check_against_gate(idl, a.gate_csv, log)
        chk.to_csv(a.out / "gate_reproduction_check.csv", index=False)
    else:
        log(f"gate CSV not found at {a.gate_csv}; skipping reproduction check (pass --gate-csv)")
    overlap_classification(idl).to_csv(a.out / "overlap_classification_robustness.csv")

    # ---------------- ML
    ml_common = None
    if a.ml or a.attribution:
        m03 = load_ml_module()
        fams = [f.strip().upper() for f in a.families.split(",")]
        seeds = [int(s) for s in a.seeds.split(",")]
        tiers = a.tiers.split(",")
        frames, ml_common = build_ml_frames(m03, a.zipdir, a.raw_dir, log)
        mls, jks = [], []
        for coh in ["cohort_ml", "cohort_id"]:
            only = ATTRIBUTION_SETS[coh] if a.attribution else None
            write_split_ids(m03, frames[coh], coh, seeds, a.out)
            m, j = run_ml(m03, frames[coh], coh, tiers, fams, seeds, a.three_block and not a.attribution,
                          a.out, log, only=only)
            mls.append(m); jks.append(j)
        mls, jks = pd.concat(mls), pd.concat(jks)
        if a.ml_metrics_csv and a.ml_metrics_csv.exists():
            mchk = check_against_ml_premise(mls, a.ml_metrics_csv, log)
            mchk.to_csv(a.out / "ml_reproduction_check.csv", index=False)
        else:
            log(f"{a.ml_metrics_csv} not found; skipping the ML reproduction check (pass --ml-metrics-csv)")
        attribution_summary(mls, a.out, log)
        mls.to_csv(a.out / "ml_block_order_results.csv", index=False)
        jks.to_csv(a.out / "ml_eval_jackknife.csv", index=False)
        if a.dagjk_groups > 0:
            full = []
            dfams = [f.strip().upper() for f in a.dagjk_families.split(",")]
            for tier in tiers:
                r, _ = run_dagjk(m03, frames["cohort_ml"], "cohort_ml", tier, dfams, a.dagjk_groups, SEEDS[0], a.out, log)
                full.append(r.assign(tier=tier))
            pd.concat(full).to_csv(a.out / "ml_dagjk_replicates.csv", index=False)

    # Online Resource 1 is built from the CSV outputs in a.out (make_online_resource.py does the same
    # for an existing outputs folder without re-running any analysis)
    tables = build_tables(a.out)
    write_resource(a.out, tables)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from make_online_resource import write_docx
        write_docx(tables, a.out / "online_resource_1.docx")
        log(f"Online Resource 1 written to {a.out / 'online_resource_1.docx'}")
    except ImportError as e:
        log(f"online_resource_1.docx not written ({e}); install python-docx")
    manifest = dict(script="design_based_and_attribution.py", version="0.7", run_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    args={k: str(v) for k, v in vars(a).items()}, python=sys.version, platform=platform.platform(),
                    packages={m: __import__(m).__version__ for m in ["numpy", "pandas", "scipy", "sklearn"]},
                    codemap=CODEMAP, ml_pipeline=ML_PIPELINE, ml_covariate_concepts=ml_common,
                    seeds=SEEDS, gamma=GAMMA, inputs_sha256=inputs)
    for opt in ["xgboost", "catboost"]:
        try:
            manifest["packages"][opt] = __import__(opt).__version__
        except Exception:
            pass
    (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
