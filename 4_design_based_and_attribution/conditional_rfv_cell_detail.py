"""Cell-level detail of the conditional RFV analysis (Sect. 6.11; Online Resource 1, Table S7).

For one year (default 2023), prints and saves, for every cell of the age and night x age partitions,
the resolved survivor and fatal counts (raw), the cell bounds s_U and f_L, and the cell-specific RFV
under BAC > 0 and BAC >= 0.08. It shows which cells are separated and how many resolved drivers the
separation rests on; in 2023 these cells also explain why Tables S7a and S7b coincide.

Usage (from the repository root):

    python 4_design_based_and_attribution/conditional_rfv_cell_detail.py
    (add --zipdir "path/to/folder" if the CRSS zip files are not in data/crss_zips; --year to change the year)

Writes conditional_rfv_cells_<year>.csv in the current folder.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_design_module():
    path = Path(__file__).resolve().parent / "design_based_and_attribution.py"
    sp = importlib.util.spec_from_file_location("design_based_and_attribution", path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--zipdir", type=Path, help="folder with the CRSS{year}CSV.zip files (default data/crss_zips)")
    src.add_argument("--raw-dir", type=Path, help="unzipped CRSS CSVs in <dir>/<year>/")
    src.add_argument("--drivers", type=Path, help="a ready driver-level parquet/csv file")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()

    a.csv = a.csv or Path(f"conditional_rfv_cells_{a.year}.csv")
    m = load_design_module()
    if not (a.zipdir or a.raw_dir or a.drivers):
        a.zipdir = m.DEFAULT_ZIPDIR
    if a.zipdir:
        raw = m.build_from_zips(a.zipdir, False)
    elif a.raw_dir:
        raw = m.build_from_raw(a.raw_dir)
    else:
        raw = pd.read_parquet(a.drivers) if a.drivers.suffix == ".parquet" else pd.read_csv(a.drivers, low_memory=False)
    df = m.derive(raw)
    d = df[(df.year == a.year) & df["cohort_id"]].copy()

    rows = []
    for name in ("age_group", "night_x_age"):
        lab = pd.Series(np.asarray(m.PARTITIONS[name](d)), index=d.index)
        for cell, dc in d.groupby(lab):
            w = dc.w
            Ns, Nf = w[dc.Y == 0].sum(), w[dc.Y == 1].sum()
            rec = dict(partition=name, cell=cell, n_raw=len(dc), n_raw_fatal=int((dc.Y == 1).sum()))
            for thr in ("bac0", "bac08"):
                R = dc[f"R_{thr}"] == 1
                Z = dc[f"Z_{thr}"] == 1
                a_s = w[(dc.Y == 0) & R].sum() / Ns if Ns > 0 else np.nan
                a_f = w[(dc.Y == 1) & R].sum() / Nf if Nf > 0 else np.nan
                q_s = w[(dc.Y == 0) & R & Z].sum() / w[(dc.Y == 0) & R].sum() if (Ns > 0 and a_s > 0) else np.nan
                q_f = w[(dc.Y == 1) & R & Z].sum() / w[(dc.Y == 1) & R].sum() if (Nf > 0 and a_f > 0) else np.nan
                unrestricted = not (a_s > 0)
                sL, sU = (0.0, 1.0) if unrestricted else (q_s * (a_s + (1 - a_s) / m.GAMMA), q_s)
                fL = a_f * q_f if a_f > 0 else 0.0
                fU = fL + (1 - a_f) if a_f == a_f else 1.0
                pi_w = Nf / (Ns + Nf) if (Ns + Nf) > 0 else np.nan
                v = m.rfv_rect(sL, sU, fL, fU, pi_w) if (Ns > 0 and Nf > 0) else 0.0
                rec.update({f"{thr}_raw_resolved_surv": int(((dc.Y == 0) & R).sum()),
                            f"{thr}_raw_resolved_pos_surv": int(((dc.Y == 0) & R & Z).sum()),
                            f"{thr}_raw_resolved_fatal": int(((dc.Y == 1) & R).sum()),
                            f"{thr}_raw_resolved_pos_fatal": int(((dc.Y == 1) & R & Z).sum()),
                            f"{thr}_s_U": sU, f"{thr}_f_L": fL, f"{thr}_RFV_cell": v})
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(a.csv, index=False)
    sep = out[(out.bac0_RFV_cell > 0) | (out.bac08_RFV_cell > 0)]
    pd.set_option("display.width", 200)
    print(f"Separated cells in {a.year} (full table in {a.csv}):")
    print(sep.T.to_string())
    same = np.allclose(sep.bac0_RFV_cell, sep.bac08_RFV_cell)
    print("\nCell RFV identical under both definitions:", same)


if __name__ == "__main__":
    main()
