#!/usr/bin/env python
"""
CRSS 2016-2017 drug-variable coding audit (supplementary)
=========================================================

A semantics audit that must precede any drug-testing replication of the
alcohol analysis. It does not assume that similarly named drug variables share
the alcohol coding: it inventories every drug/toxicology/test-related field,
records value frequencies and writes crosstabs for plausible status/result
pairs. The mapping must be checked against the CRSS codebook and frozen before
any drug RFV calculation; this script does not construct one.

Data: CRSS{year}CSV.zip in --zipdir (default data/crss_zips; downloaded if missing).
Outputs: --output-dir (default outputs/5_supplementary_audits/crss_drug_coding).
"""
from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs" / "5_supplementary_audits" / "crss_drug_coding"
ZIP_DIR = REPO_ROOT / "data" / "crss_zips"
NHTSA_STATIC = "https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip"
YEARS = [2016, 2017]
PAT = re.compile(r"DRUG|TOX|NARC|TEST", re.I)


def log(msg: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def download(year: int) -> Path:
    p = ZIP_DIR / f"CRSS{year}CSV.zip"
    if p.exists() and zipfile.is_zipfile(p):
        return p
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    url = NHTSA_STATIC.format(year=year)
    log(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with p.open("wb") as f:
            for ch in r.iter_content(1024 * 1024):
                if ch:
                    f.write(ch)
    return p


def read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    raw = zf.read(member)
    last = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding=enc)
            df.columns = [str(c).strip().upper() for c in df.columns]
            return df
        except Exception as exc:
            last = exc
    raise last


def main():
    global OUT, ZIP_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zipdir", type=Path, default=ZIP_DIR, help="folder with CRSS{year}CSV.zip (default: data/crss_zips)")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    OUT, ZIP_DIR = args.output_dir, args.zipdir
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    freq_rows = []
    pair_notes = []

    for year in YEARS:
        zpath = download(year)
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                df = read_member(zf, member)
                hits = [c for c in df.columns if PAT.search(c)]
                if not hits:
                    continue
                table = Path(member).name
                for c in hits:
                    inventory.append({"year": year, "table": table, "column": c, "dtype": str(df[c].dtype)})
                    vc = df[c].value_counts(dropna=False).head(100)
                    for val, n in vc.items():
                        freq_rows.append({"year": year, "table": table, "column": c, "value": str(val), "raw_n": int(n)})

                # Crosstab every plausible status/result pair within the same table.
                status_like = [c for c in hits if re.search(r"STATUS|TST|TEST", c, re.I)]
                result_like = [c for c in hits if re.search(r"RES|RESULT|POS|DRUG", c, re.I) and c not in status_like]
                for s in status_like:
                    for r in result_like:
                        tab = pd.crosstab(df[s], df[r], dropna=False)
                        fn = OUT / f"{year}_{Path(table).stem}_{s}_by_{r}.csv"
                        tab.to_csv(fn)
                        pair_notes.append({"year": year, "table": table, "status_candidate": s, "result_candidate": r, "crosstab_file": fn.name})

    pd.DataFrame(inventory).to_csv(OUT / "drug_candidate_field_inventory.csv", index=False)
    pd.DataFrame(freq_rows).to_csv(OUT / "drug_candidate_value_frequencies.csv", index=False)
    pd.DataFrame(pair_notes).to_csv(OUT / "drug_candidate_crosstab_index.csv", index=False)

    note = (
        "Drug replication audit completed.\n"
        "Do not construct A/R/Z until the 2016 and 2017 CRSS codebook definitions are checked against these empirical frequencies.\n"
        "The alcohol code mapping must NOT be copied to drug variables by analogy.\n"
    )
    (OUT / "drug_coding_audit_note.txt").write_text(note, encoding="utf-8")
    print(note)
    log(f"Saved audit to {OUT}")


if __name__ == "__main__":
    main()
