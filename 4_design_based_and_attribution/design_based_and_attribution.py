#!/usr/bin/env python3
"""
Design-based uncertainty, cohort harmonization, block-order attribution
=======================================================================

Runs every analysis that needs CRSS driver-level microdata together with the
survey design variables:

  * Design-based (delete-one-PSU jackknife) uncertainty for annual prevalences,
    resolved shares, the survivor ceiling, MAS intervals, separation margins and
    the phi_L x gamma_cap frontier; Kish and design effective sample sizes;
    complete annual results for both BAC thresholds (BAC>0 and BAC>=0.08).
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

Outputs (--out, default outputs/4_design_based_and_attribution): CSV files,
supplementary_tables.md / .xlsx, split_ids_<cohort>.csv.gz, and
run_manifest.json recording versions, arguments, code map and input hashes.
When the gate CSV from 1_identification_gate is available, the identification
quantities are compared with it (gate_reproduction_check.csv).

Input
-----
One of
  --zipdir DIR    CRSS{year}CSV.zip files (default data/crss_zips; --download
                  fetches missing years from NHTSA),
  --drivers PATH  a driver-level parquet/csv with the columns in COLS
                  (rename with --colmap JSON),
  --raw-dir DIR   unzipped CRSS CSVs in DIR/<year>/ (ACCIDENT, PERSON, VEHICLE).

Coding follows the identification gate (CODEMAP below); the run manifest
records the map used. Design variables follow NHTSA DOT HS 812 688:
STRATA = PSUSTRAT, CLUSTER = PSU_VAR, WEIGHT = WEIGHT; delete-one-PSU jackknife
(JKn) with the with-replacement approximation; domain estimates are computed on
the full sample.

Note: the ML covariates here are a re-implementation of the W sets in
3_ml_premise, so absolute baselines differ from that script; all comparisons
in this script are within-run.
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

# ---- covariate tiers ---------------------------------------------------------------
W_CORE_CAT = ["SEX", "MONTH", "DAY_WEEK", "HOUR", "WEATHER", "LGT_COND", "REL_ROAD",
              "TYP_INT", "WRK_ZONE", "BODY_TYP", "SPEEDREL", "P_CRASH1"]
W_CORE_NUM = ["AGE", "TRAV_SP"]
W_EXP_CAT = ["ROLLOVER", "IMPACT1", "TOWED", "DEFORMED"]
NUM_MISSING_CODES = {"AGE": [998, 999], "TRAV_SP": [997, 998, 999]}

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


def build_from_zips(zipdir: Path, download: bool = False) -> pd.DataFrame:
    """Read the same CRSS{year}CSV.zip files the identification gate uses."""
    import zipfile
    parts = []
    for y in YEARS:
        zp = zipdir / f"CRSS{y}CSV.zip"
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
    x["cell"] = x.apply(lambda r: f"{r.est:.4f} [{r.lo:.4f}, {r.hi:.4f}]", axis=1)
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
FEATURE_SETS = {
    "M0": [], "M1": ["A"], "M1_3": ["A3"], "M_R": ["R", "RZ"], "M_AR": ["A", "R"],
    "M2": ["A", "R", "RZ"], "M2_3": ["A3", "R", "RZ"],
}
THREE_BLOCK = {"S_R": ["R"], "S_RZ": ["RZ"], "S_A_RZ": ["A", "RZ"]}


def ml_frame(df: pd.DataFrame, cohort: str, thr: str, tier: str):
    d = df[df[cohort]].copy()
    d["R"] = d[f"R_{thr}"]
    d["RZ"] = (d["R"] == 1) & (d[f"Z_{thr}"] == 1)
    d["RZ"] = d["RZ"].astype(int)
    d["A3_tested"] = (d.status3 == "tested").astype(int)
    d["A3_unknown"] = (d.status3 == "unknown").astype(int)
    cat = list(W_CORE_CAT) + (list(W_EXP_CAT) if tier == "expanded" else [])
    num = list(W_CORE_NUM)
    for c in num:
        v = pd.to_numeric(d[c], errors="coerce")
        d[c] = v.where(~v.isin(NUM_MISSING_CODES.get(c, [])))
    for c in cat:
        d[c] = d[c].astype(str)
    return d, cat, num


def case_split(d: pd.DataFrame, seed: int):
    """70/10/20 crash-case split, approximately stratified by crash-level fatality."""
    rng = np.random.default_rng(seed)
    cf = d.groupby("case")["Y"].max()
    assign = pd.Series(index=cf.index, dtype=object)
    for v in [0, 1]:
        ids = cf.index[cf.values == v].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        a, b = int(0.7 * n), int(0.8 * n)
        assign[ids[:a]] = "train"; assign[ids[a:b]] = "val"; assign[ids[b:]] = "test"
    return d["case"].map(assign).values


def extra_cols(fs):
    cols = []
    for f in fs:
        cols += ["A3_tested", "A3_unknown"] if f == "A3" else [f]
    return cols


def fit_predict(family, Xtr, ytr, wtr, Xte, cat, num, extra, seed):
    wtr = wtr / wtr.mean()
    if family == "lr":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
        ct = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False)), num),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), ohe), cat),
            ("x", "passthrough", extra)])
        m = make_pipeline(ct, LogisticRegression(C=1.0, solver="liblinear", max_iter=1000))
        m.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
        return m.predict_proba(Xte)[:, 1]
    if family == "xgb":
        import xgboost as xgb
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        ct = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False)), num),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)), cat),
            ("x", "passthrough", extra)])
        Xt = ct.fit_transform(Xtr)
        m = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.85,
                              colsample_bytree=0.85, min_child_weight=5, reg_lambda=1.0,
                              tree_method="hist", random_state=seed, n_jobs=-1, eval_metric="logloss")
        m.fit(Xt, ytr, sample_weight=wtr)
        return m.predict_proba(ct.transform(Xte))[:, 1]
    if family == "cat":
        from catboost import CatBoostClassifier
        Xtr = Xtr.copy(); Xte = Xte.copy()
        for c in num:
            med = Xtr[c].median()
            Xtr[c] = Xtr[c].fillna(med); Xte[c] = Xte[c].fillna(med)
        m = CatBoostClassifier(iterations=500, depth=7, learning_rate=0.05, random_seed=seed,
                               verbose=False, thread_count=-1)
        m.fit(Xtr, ytr, cat_features=cat, sample_weight=wtr)
        return m.predict_proba(Xte)[:, 1]
    raise ValueError(family)


def wmetrics(y, p, w):
    from sklearn.metrics import average_precision_score
    return dict(pr_auc=average_precision_score(y, p, sample_weight=w),
                brier=float(np.sum(w * (p - y) ** 2) / np.sum(w)))


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


def write_split_ids(df: pd.DataFrame, cohort: str, seeds, out: Path) -> None:
    """Case-level train/val/test assignment for every seed (one column per seed)."""
    d = df[df[cohort]][["case", "Y"]].copy()
    cases = d.drop_duplicates("case")[["case"]].reset_index(drop=True)
    for seed in seeds:
        m = pd.Series(case_split(d, seed), index=d["case"].values)
        cases[f"seed_{seed}"] = cases["case"].map(m[~m.index.duplicated()]).values
    cases.sort_values("case").to_csv(out / f"split_ids_{cohort}.csv.gz", index=False, compression="gzip")


def run_ml(df, cohort, thr, tiers, families, seeds, three_block, out: Path, log, only=None):
    """Fit the feature sets for every tier x seed x family. Each completed (tier, seed, family) group is appended
    to ml_checkpoint_*.csv, so an interrupted run resumes where it stopped."""
    sets = dict(FEATURE_SETS)
    if three_block:
        sets.update(THREE_BLOCK)
    if only:
        sets = {k: v for k, v in sets.items() if k in only}
    ck_fit = out / f"ml_checkpoint_fits_{cohort}_{thr}.csv"
    ck_jk = out / f"ml_checkpoint_jk_{cohort}_{thr}.csv"
    done = set()
    if ck_fit.exists():
        prev = pd.read_csv(ck_fit)
        done = set(zip(prev.tier, prev.seed, prev.family))
    groups = [(t, sd, f) for t in tiers for sd in seeds for f in families]
    todo = [g for g in groups if g not in done]
    log(f"{cohort}: {len(groups)} groups x {len(sets)} models; {len(groups) - len(todo)} already done (checkpoint)")
    t_start, n_done = time.time(), 0
    for tier in tiers:
        if not any(g[0] == tier for g in todo):
            continue
        d, cat, num = ml_frame(df, cohort, thr, tier)
        for seed in seeds:
            if not any(g[:2] == (tier, seed) for g in todo):
                continue
            part = case_split(d, seed)
            tr, te = d[part == "train"], d[part == "test"]
            for fam in families:
                if (tier, seed, fam) not in todo:
                    continue
                recs, jkrecs, preds, mets = [], [], {}, {}
                for name, fs in sets.items():
                    ex = extra_cols(fs)
                    t0 = time.time()
                    p = fit_predict(fam, tr[cat + num + ex], tr["Y"].values, tr["w"].values,
                                    te[cat + num + ex], cat, num, ex, seed)
                    preds[name] = p
                    mets[name] = wmetrics(te["Y"].values, p, te["w"].values)
                    recs.append(dict(cohort=cohort, threshold=thr, tier=tier, seed=seed, family=fam, model=name,
                                     **mets[name], secs=time.time() - t0))
                    log(f"  {cohort} {tier} seed={seed} {fam} {name} PR-AUC={mets[name]['pr_auc']:.4f} "
                        f"({time.time() - t0:.0f}s)")
                for key in ["pr_auc", "brier"]:
                    sh = shares(mets, key)
                    if sh:
                        recs.append(dict(cohort=cohort, threshold=thr, tier=tier, seed=seed, family=fam,
                                         model=f"shares_{key}", **sh))
                pairs = [pr for pr in [("M1", "M0"), ("M1_3", "M0"), ("M2", "M0"), ("M2", "M_AR")]
                         if pr[0] in preds and pr[1] in preds]
                for k, (est, se, dfree) in jk_eval(te, preds, pairs).items():
                    jkrecs.append(dict(cohort=cohort, threshold=thr, tier=tier, seed=seed, family=fam,
                                       contrast=k, est=est, se_eval_jk=se, jk_df=dfree))
                pd.DataFrame(recs).to_csv(ck_fit, mode="a", header=not ck_fit.exists(), index=False)
                if jkrecs:
                    pd.DataFrame(jkrecs).to_csv(ck_jk, mode="a", header=not ck_jk.exists(), index=False)
                n_done += 1
                el = time.time() - t_start
                left = len(todo) - n_done
                log(f"  group {n_done}/{len(todo)} done for {cohort}; elapsed {el / 60:.0f} min, "
                    f"about {el / n_done * left / 60:.0f} min left for this cohort")
    fits = pd.read_csv(ck_fit) if ck_fit.exists() else pd.DataFrame()
    jks = pd.read_csv(ck_jk) if ck_jk.exists() else pd.DataFrame()
    return fits, jks


def attribution_summary(mls: pd.DataFrame, out: Path, log) -> pd.DataFrame:
    """Testing-status, cohort and block-order contrasts per tier x family, averaged over seeds (PR-AUC)."""
    met = mls[~mls.model.astype(str).str.startswith("shares")]
    pr = met.pivot_table(index=["cohort", "tier", "family", "seed"], columns="model", values="pr_auc")
    rows = []
    for (tier, fam), g in pr.loc["cohort_ml"].groupby(level=["tier", "family"]):
        r = dict(tier=tier, family=fam,
                 M0=g["M0"].mean(), M1=g["M1"].mean(), M2=g["M2"].mean(),
                 M1_minus_M0=(g["M1"] - g["M0"]).mean())
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


def run_dagjk(df, cohort, thr, tier, families, G, seed, out, log):
    """Delete-a-group jackknife with refitting (full design-based propagation for
    the M1-M0 and M2-M0 PR-AUC contrasts). PSUs are assigned to G groups
    systematically within strata after a random sort."""
    d, cat, num = ml_frame(df, cohort, thr, tier)
    rng = np.random.default_rng(seed)
    psus = d[["strat", "psu"]].drop_duplicates().sample(frac=1, random_state=seed).sort_values("strat", kind="stable")
    psus["g"] = np.arange(len(psus)) % G
    gmap = dict(zip(psus.psu, psus.g))
    d["g"] = d.psu.map(gmap).values
    part = case_split(d, seed)
    rows = []
    for fam in families:
        for rep in [-1] + list(range(G)):
            w = d["w"].values.copy()
            if rep >= 0:
                w = np.where(d["g"].values == rep, 0.0, w * G / (G - 1))
            dd = d.assign(w_rep=w)
            tr, te = dd[(part == "train") & (dd.w_rep > 0)], dd[(part == "test") & (dd.w_rep > 0)]
            m = {}
            for name in ["M0", "M1", "M2"]:
                ex = extra_cols(FEATURE_SETS[name])
                p = fit_predict(fam, tr[cat + num + ex], tr["Y"].values, tr["w_rep"].values,
                                te[cat + num + ex], cat, num, ex, seed)
                m[name] = wmetrics(te["Y"].values, p, te["w_rep"].values)["pr_auc"]
            rows.append(dict(family=fam, rep=rep, M1_M0=m["M1"] - m["M0"], M2_M0=m["M2"] - m["M0"],
                             A_first_share=(m["M1"] - m["M0"]) / (m["M2"] - m["M0"])))
            log(f"  DAGJK {fam} rep={rep} M1-M0={m['M1'] - m['M0']:.4f}")
    r = pd.DataFrame(rows)
    summ = []
    for fam, x in r.groupby("family"):
        full = x[x.rep == -1].iloc[0]
        reps = x[x.rep >= 0]
        for k in ["M1_M0", "M2_M0", "A_first_share"]:
            var = (G - 1) / G * np.sum((reps[k] - full[k]) ** 2)
            summ.append(dict(cohort=cohort, threshold=thr, tier=tier, family=fam, stat=k, est=full[k],
                             se_dagjk=np.sqrt(var), groups=G, lo=full[k] - tcrit(G - 1) * np.sqrt(var),
                             hi=full[k] + tcrit(G - 1) * np.sqrt(var)))
    return r, pd.DataFrame(summ)


# ===================================================================== reporting
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
                     help="folder with CRSS{year}CSV.zip (default: data/crss_zips)")
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
        inputs = {f"CRSS{y}CSV.zip": sha256(a.zipdir / f"CRSS{y}CSV.zip") for y in YEARS}
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

    tables = {}
    log(f"data source: {a.zipdir or a.drivers or a.raw_dir}; gate check file: {a.gate_csv}")
    if a.gate_csv and a.gate_csv.exists() and "bac0" in thresholds:
        chk = check_against_gate(idl, a.gate_csv, log)
        chk.to_csv(a.out / "gate_reproduction_check.csv", index=False)
        tables["S0_gate_reproduction_check"] = chk.set_index("statistic")
    else:
        log(f"gate CSV not found at {a.gate_csv}; skipping reproduction check (pass --gate-csv)")
    for thr in thresholds:
        x = idl[(idl.cohort == "cohort_id") & (idl.threshold == thr)]
        tables[f"S1_prevalence_{thr}"] = wide(x, ["pi", "test_s", "test_f", "alpha_s", "alpha_f", "q_s", "q_f"])
        tables[f"S2_intervals_{thr}"] = wide(x, ["s_L", "s_U", "f_L", "f_U", "d_sep", "rev_margin"])
        tables[f"S3_frontier_{thr}"] = wide(x, [f"c_crit_{p:.2f}" for p in PHI_GRID])
        tables[f"S4_status_{thr}"] = wide(x, ["unknown_s", "unknown_f", "test_s_known", "test_f_known",
                                              "share_TU_s", "share_unknown_s", "d_sep_TU_free", "d_sep_TU_unknown_free"])
        # cohort harmonization: identification on ML cohort, difference vs ID cohort
        a_ = idl[(idl.threshold == thr)].pivot_table(index=["year", "stat"], columns="cohort", values="est")
        a_["diff_ml_minus_id"] = a_["cohort_ml"] - a_["cohort_id"]
        keep = ["pi", "alpha_s", "q_s", "alpha_f", "q_f", "d_sep", "rev_margin"]
        tables[f"S5_cohort_harmonization_{thr}"] = a_.reset_index().query("stat in @keep").pivot(
            index="year", columns="stat", values="diff_ml_minus_id")[keep]
        e = essl[(essl.cohort == "cohort_id") & (essl.threshold == thr)]
        tables[f"S6_ess_{thr}"] = e.pivot_table(index="year", columns="cell", values="kish_ess").round(1)
        tables[f"S7_conditional_rfv_{thr}"] = cond[(cond.cohort == "cohort_id") & (cond.threshold == thr)].set_index(["year", "partition"])
    # classification robustness summary
    cls = idl[idl.stat.isin(["d_sep", "rev_margin"])].pivot_table(
        index=["cohort", "threshold", "year"], columns="stat", values=["hi", "rep_max"])
    cls.columns = [f"{a_}_{b}" for a_, b in cls.columns]
    cls["overlap_robust_95"] = (cls["hi_d_sep"] < 0) & (cls["hi_rev_margin"] < 0)
    cls["overlap_all_replicates"] = (cls["rep_max_d_sep"] < 0) & (cls["rep_max_rev_margin"] < 0)
    tables["S8_overlap_classification_robustness"] = cls
    cls.to_csv(a.out / "overlap_classification_robustness.csv")

    # ---------------- ML
    if a.ml or a.attribution:
        fams = a.families.split(",")
        seeds = [int(s) for s in a.seeds.split(",")]
        tiers = a.tiers.split(",")
        mls, jks = [], []
        for coh in ["cohort_ml", "cohort_id"]:
            only = ATTRIBUTION_SETS[coh] if a.attribution else None
            write_split_ids(df, coh, seeds, a.out)
            m, j = run_ml(df, coh, "bac0", tiers, fams, seeds, a.three_block and not a.attribution, a.out, log, only=only)
            mls.append(m); jks.append(j)
        mls, jks = pd.concat(mls), pd.concat(jks)
        attribution_summary(mls, a.out, log)
        mls.to_csv(a.out / "ml_block_order_results.csv", index=False)
        jks.to_csv(a.out / "ml_eval_jackknife.csv", index=False)
        met = mls[~mls.model.str.startswith("shares")]
        tables["S9_ml_prauc_by_model"] = met.pivot_table(index=["cohort", "tier", "family"], columns="model",
                                                         values="pr_auc", aggfunc="mean").round(4)
        sh = mls[mls.model == "shares_pr_auc"]
        cols = [c for c in ["total_gain", "A_first", "A_last", "A_shapley2", "value_only",
                            "shapley3_A", "shapley3_R", "shapley3_RZ"] if c in sh]
        def _flat(t):
            t.columns = [f"{a_}_{b}" for a_, b in t.columns]
            return t
        tables["S10_block_order_shares_prauc"] = _flat(sh.groupby(["cohort", "tier", "family"])[cols].agg(["mean", "std"]).round(4))
        shb = mls[mls.model == "shares_brier"]
        tables["S11_block_order_shares_brier"] = _flat(shb.groupby(["cohort", "tier", "family"])[cols].agg(["mean", "std"]).round(4))
        jsum = jks.groupby(["cohort", "tier", "family", "contrast"]).agg(est=("est", "mean"), se_eval_jk=("se_eval_jk", "mean"),
                                                                       seed_sd=("est", "std")).round(5)
        tables["S12_ml_design_eval_uncertainty"] = jsum
        if a.dagjk_groups > 0:
            full, summ = [], []
            for tier in tiers:
                r, s = run_dagjk(df, "cohort_ml", "bac0", tier, a.dagjk_families.split(","), a.dagjk_groups, SEEDS[0], a.out, log)
                full.append(r.assign(tier=tier)); summ.append(s)
            pd.concat(full).to_csv(a.out / "ml_dagjk_replicates.csv", index=False)
            tables["S13_ml_dagjk_refit"] = pd.concat(summ).set_index(["tier", "family", "stat"]).round(5)

    write_resource(a.out, tables)
    manifest = dict(script="design_based_and_attribution.py", version="0.7", run_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    args={k: str(v) for k, v in vars(a).items()}, python=sys.version, platform=platform.platform(),
                    packages={m: __import__(m).__version__ for m in ["numpy", "pandas", "scipy", "sklearn"]},
                    codemap=CODEMAP, w_core=W_CORE_CAT + W_CORE_NUM, w_expanded_extra=W_EXP_CAT,
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
