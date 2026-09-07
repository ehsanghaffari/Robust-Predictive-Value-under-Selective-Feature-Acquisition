#!/usr/bin/env python3
"""Prepare the PennDOT driver-level analysis dataset for 2016-2025."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = list(range(2016, 2026))

CRASH_COLS = [
    "CRN", "CRASH_MONTH", "DAY_OF_WEEK", "HOUR_OF_DAY", "ILLUMINATION",
    "RELATION_TO_ROAD", "ROAD_CONDITION", "WEATHER1", "INTERSECTION_RELATED",
    "INTERSECT_TYPE", "WORK_ZONE_IND", "URBAN_RURAL", "COLLISION_TYPE",
]

VEH_COLS = [
    "CRN", "UNIT_NUM", "BODY_TYPE", "TRAVEL_SPD", "VEH_MOVEMENT",
    "UNIT_TYPE", "RDWY_ALIGNMENT",
]

PERS_COLS = [
    "CRN", "UNIT_NUM", "PERSON_TYPE", "INJ_SEVERITY", "DVR_PED_CONDITION",
    "AGE", "SEX",
]


def read_driver_data(data_dir: Path) -> pd.DataFrame:
    parts = []

    for year in YEARS:
        zpath = data_dir / f"Statewide_{year}.zip"
        if not zpath.exists():
            raise FileNotFoundError(f"Missing required input: {zpath}")

        with zipfile.ZipFile(zpath) as z:
            person = pd.read_csv(
                z.open(f"PERSON_{year}.csv"),
                usecols=PERS_COLS,
                low_memory=False,
            )
            person = person.loc[person["PERSON_TYPE"] == 1].copy()

            crash = pd.read_csv(
                z.open(f"CRASH_{year}.csv"),
                usecols=CRASH_COLS,
                low_memory=False,
            )

            vehicle = pd.read_csv(
                z.open(f"VEHICLE_{year}.csv"),
                usecols=VEH_COLS,
                low_memory=False,
            )

        d = person.merge(
            vehicle,
            on=["CRN", "UNIT_NUM"],
            how="left",
            validate="many_to_one",
        )
        d = d.merge(
            crash,
            on="CRN",
            how="left",
            validate="many_to_one",
        )

        d["YEAR"] = year
        d["Y_FATAL"] = (d["INJ_SEVERITY"] == 1).astype(np.int8)
        d["R_RESOLVED"] = d["DVR_PED_CONDITION"].isin([0, 1, 2, 3, 4, 5, 6]).astype(np.int8)
        d["Z_DRINK"] = (d["DVR_PED_CONDITION"] == 1).astype(np.int8)
        d["Z_STATE"] = np.where(
            d["R_RESOLVED"] == 0,
            0,
            np.where(d["Z_DRINK"] == 1, 2, 1),
        ).astype(np.int8)

        parts.append(d)

    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing Statewide_2016.zip through Statewide_2025.zip",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Prepared driver-level .pkl.gz file",
    )
    args = parser.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    driver = read_driver_data(args.data_dir)
    driver.to_pickle(args.output_file, compression="gzip")

    print(f"Prepared {len(driver):,} driver records")
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
