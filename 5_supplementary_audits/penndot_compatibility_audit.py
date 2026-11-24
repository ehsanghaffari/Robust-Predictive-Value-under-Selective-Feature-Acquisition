#!/usr/bin/env python
"""
PennDOT schema compatibility audit (supplementary)
==================================================

Checks whether PennDOT Statewide crash files (2016-2025) contain distinct
acquisition (A), result-resolution (R) and latent alcohol state (Z/BAC) fields
comparable to CRSS ALC_STATUS/ALC_RES. It does not train models and does not
reinterpret final PennDOT crash flags as alcohol-test acquisition.

Data (not redistributed)
------------------------
--data-dir (default data/penndot) containing Statewide_<year>/ folders with the
PennDOT Statewide CSV files and, optionally, a penndot_profile/ folder.

Outputs (--output-dir, default outputs/5_supplementary_audits/penndot_compatibility)
-----------------------------------------------------------------------------------
schema_inventory.csv, alcohol_drug_candidate_fields.csv,
required_field_audit.csv, penndot_compatibility_verdict.txt
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "penndot"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "5_supplementary_audits" / "penndot_compatibility"
YEARS = range(2016, 2026)

DIRECT_REQUIRED_PATTERNS = {
    "acquisition_A": [r"^ALC_STATUS$", r"ALCOHOL.*TEST", r"(^|_)BAC($|_).*TEST", r"TEST.*ALCOHOL"],
    "resolution_R": [r"^ALC_RES$", r"(^|_)BAC($|_)", r"ALCOHOL.*RESULT", r"TEST.*RESULT"],
    "latent_Z": [r"^ALC_RES$", r"(^|_)BAC($|_)", r"BLOOD.*ALCOHOL"],
}
CANDIDATE_PATTERN = re.compile(r"ALC|ALCOHOL|(^|_)BAC($|_)|DRUG|TOX|TEST|DRINK", re.I)
KNOWN_FINAL_FLAGS = {"ALCOHOL_RELATED", "DRINKING_DRIVER", "DRUGGED_DRIVER", "DRUG_RELATED", "ILLEGAL_DRUG_RELATED", "MARIJUANA_RELATED"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def locate_csvs(year_dir: Path, year: int) -> list[Path]:
    if not year_dir.exists():
        return []
    files = list(year_dir.glob("*.csv")) + list(year_dir.glob("*.CSV"))
    return sorted(files)


def read_header(path: Path) -> list[str]:
    last = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return [str(x).strip().upper() for x in next(csv.reader(f))]
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Could not read header for {path}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="folder with Statewide_<year>/ sub-folders (default: data/penndot)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    DATA_DIR = args.data_dir
    PROFILE_DIR = DATA_DIR / "penndot_profile"
    YEAR_DIRS = {y: DATA_DIR / f"Statewide_{y}" for y in YEARS}
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {OUTPUT_DIR}")

    schema_rows = []
    candidate_rows = []
    required_rows = []

    for year, folder in YEAR_DIRS.items():
        csvs = locate_csvs(folder, year)
        if not csvs:
            log(f"WARNING {year}: no CSV files found in {folder}")
            schema_rows.append({"year": year, "table": "<missing>", "file": folder.name, "n_columns": 0, "columns": ""})
            continue

        all_columns = set()
        for path in csvs:
            cols = read_header(path)
            all_columns.update(cols)
            schema_rows.append({
                "year": year,
                "table": path.stem,
                "file": path.name,
                "n_columns": len(cols),
                "columns": "|".join(cols),
            })
            for c in cols:
                if CANDIDATE_PATTERN.search(c):
                    candidate_rows.append({
                        "year": year,
                        "table": path.stem,
                        "column": c,
                        "is_known_final_flag": c in KNOWN_FINAL_FLAGS,
                    })

        for role, pats in DIRECT_REQUIRED_PATTERNS.items():
            hits = sorted(c for c in all_columns if any(re.search(p, c, re.I) for p in pats))
            required_rows.append({
                "year": year,
                "required_role": role,
                "direct_field_found": bool(hits),
                "matching_columns": "|".join(hits),
            })

    schema_df = pd.DataFrame(schema_rows)
    cand_df = pd.DataFrame(candidate_rows)
    req_df = pd.DataFrame(required_rows)

    schema_df.to_csv(OUTPUT_DIR / "schema_inventory.csv", index=False)
    cand_df.to_csv(OUTPUT_DIR / "alcohol_drug_candidate_fields.csv", index=False)
    req_df.to_csv(OUTPUT_DIR / "required_field_audit.csv", index=False)

    per_year = req_df.pivot(index="year", columns="required_role", values="direct_field_found") if not req_df.empty else pd.DataFrame()
    direct_compatible_years = []
    if not per_year.empty:
        for y, row in per_year.iterrows():
            if bool(row.get("acquisition_A", False)) and bool(row.get("resolution_R", False)) and bool(row.get("latent_Z", False)):
                direct_compatible_years.append(int(y))

    profile_status = "missing"
    profile_files = []
    if PROFILE_DIR.exists():
        profile_status = "present"
        for p in PROFILE_DIR.rglob("*"):
            if p.is_file():
                profile_files.append(str(p.relative_to(PROFILE_DIR)))
        pd.DataFrame({"profile_file": profile_files}).to_csv(OUTPUT_DIR / "penndot_profile_file_inventory.csv", index=False)

    verdict_lines = [
        "PennDOT schema compatibility audit",
        "==================================",
        f"Profile directory: penndot_profile ({profile_status})",
        f"Statewide years requested: 2016-2025",
        "",
        "Primary CRSS-paper requirement:",
        "  Distinct variables for acquisition A, result resolution R, and latent alcohol state Z/BAC.",
        "",
    ]

    if direct_compatible_years:
        verdict_lines += [
            f"DIRECT-COMPATIBLE YEARS FOUND: {direct_compatible_years}",
            "Inspect required_field_audit.csv before use; matching by name alone does not prove semantic equivalence.",
        ]
    else:
        verdict_lines += [
            "VERDICT: PennDOT Statewide files are NOT directly compatible with the primary CRSS A/R/Z selective-acquisition analysis.",
            "The Statewide schema may contain final administrative flags such as ALCOHOL_RELATED and DRINKING_DRIVER,",
            "but those are not equivalent to an alcohol-test acquisition indicator or a resolved BAC result.",
            "Do NOT substitute those flags for ALC_STATUS/ALC_RES in the primary analysis.",
            "They may be used only in a separately labeled supplementary administrative-flag shortcut audit.",
        ]

    verdict = "\n".join(verdict_lines)
    (OUTPUT_DIR / "penndot_compatibility_verdict.txt").write_text(verdict, encoding="utf-8")
    print("\n" + verdict)
    log("Audit complete.")


if __name__ == "__main__":
    main()
