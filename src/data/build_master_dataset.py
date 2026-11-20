
"""
EcoRoute AI
Master Dataset Builder - Version 1.0

Purpose:
    Consolidate original vehicle trajectory files
    while preserving observation order and identity.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "external" / "dryad_truck"

OUTPUT_DIR = ROOT / "data" / "processed"

REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


EXPECTED_COLUMNS = [
    "Speed",
    "FuelRate",
    "DistanceToIntersection"
]


def extract_metadata(file):

    pattern = (
        r"^(\d{8})([^_]+)_"
        r"([^_]+)_([^_]+)_"
        r"([^_]+)_(baseline|eco)$"
    )

    match = re.fullmatch(
        pattern,
        file.stem,
        flags=re.IGNORECASE
    )

    if not match:
        raise ValueError(
            f"Unexpected filename: {file.name}"
        )

    date, vehicle, corridor, direction, experiment, scenario = (
        match.groups()
    )

    experiment_date = pd.to_datetime(
        date,
        format="%Y%m%d"
    ).strftime("%Y-%m-%d")

    return {
        "experiment_date": experiment_date,
        "vehicle_id": vehicle,
        "corridor": corridor,
        "direction": direction,
        "experiment_id": experiment,
        "scenario": scenario.lower()
    }


def process_file(file):

    df = pd.read_csv(file)

    if df.empty:
        raise ValueError("Empty trajectory")

    if set(df.columns) != set(EXPECTED_COLUMNS):
        raise ValueError("Unexpected column schema")

    # Validate original measurements
    df = df[EXPECTED_COLUMNS].apply(
        pd.to_numeric,
        errors="raise"
    )

    values = df.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Missing or infinite measurements"
        )

    if (values < 0).any():
        raise ValueError("Negative measurements")

    metadata = extract_metadata(file)

    source_file = file.relative_to(
        DATA_DIR
    ).as_posix()

    # Preserve original trajectory identity
    df["trajectory_id"] = source_file

    df["source_file"] = source_file

    for key, value in metadata.items():
        df[key] = value

    # Preserve observation order
    df["sample_index"] = np.arange(len(df))

    # Preserve original measurement values
    df = df.rename(columns={
        "Speed": "speed_raw",
        "FuelRate": "fuel_rate_raw",
        "DistanceToIntersection":
            "distance_to_intersection_raw"
    })

    df["data_source"] = "dryad_truck"

    return df


def main():

    files = sorted(DATA_DIR.rglob("*.csv"))

    print("=" * 65)
    print("ECOROUTE AI - MASTER DATASET BUILDER")
    print("=" * 65)

    print("CSV files discovered:", len(files))

    if not files:
        raise RuntimeError("No CSV files found.")

    frames = []
    errors = []

    for index, file in enumerate(files, start=1):

        try:

            frames.append(process_file(file))

        except Exception as error:

            errors.append({
                "filename": file.name,
                "error": str(error)
            })

        if index % 50 == 0:
            print(
                f"Processed {index}/{len(files)} files"
            )

    # Save errors for investigation
    error_df = pd.DataFrame(
        errors,
        columns=["filename", "error"]
    )

    error_df.to_csv(
        REPORT_DIR / "consolidation_errors.csv",
        index=False
    )

    if errors:
        print("Files requiring investigation:", len(errors))
        print(error_df.head(10).to_string(index=False))

        raise RuntimeError(
            "Consolidation stopped. Review error report."
        )

    # Combine validated trajectories
    master = pd.concat(
        frames,
        ignore_index=True
    )

    column_order = [
        "trajectory_id",
        "source_file",
        "experiment_date",
        "vehicle_id",
        "corridor",
        "direction",
        "experiment_id",
        "scenario",
        "sample_index",
        "speed_raw",
        "fuel_rate_raw",
        "distance_to_intersection_raw",
        "data_source"
    ]

    master = master[column_order]

    # Integrity checks
    expected_rows = sum(len(df) for df in frames)

    if len(master) != expected_rows:
        raise RuntimeError("Row count mismatch")

    if master["trajectory_id"].nunique() != len(files):
        raise RuntimeError("Trajectory count mismatch")

    if master.duplicated(
        subset=["trajectory_id", "sample_index"]
    ).any():
        raise RuntimeError("Duplicate observation IDs")

    # Save master dataset
    output_file = (
        OUTPUT_DIR / "truck_telemetry_master.csv"
    )

    master.to_csv(output_file, index=False)

    print("\n" + "=" * 65)
    print("MASTER DATASET SUMMARY")
    print("=" * 65)

    print("Total rows:", len(master))
    print("Total columns:", len(master.columns))
    print("Unique trajectories:",
          master["trajectory_id"].nunique())

    print("Unique vehicles:",
          master["vehicle_id"].nunique())

    print("\nDRIVING SCENARIOS")
    print(master["scenario"].value_counts())

    print("\nVEHICLE IDENTIFIERS")
    print(master["vehicle_id"].value_counts())

    print("\nROAD CORRIDORS")
    print(master["corridor"].value_counts())

    print("\nTRAVEL DIRECTIONS")
    print(master["direction"].value_counts())

    print("\nEXPERIMENT DATES")
    print(master["experiment_date"].value_counts())

    print("\nMISSING VALUES")
    print(master.isna().sum())

    print("\nFIRST FIVE OBSERVATIONS")
    print(master.head().to_string(index=False))

    print("\nSaved to:", output_file)

    print("Dataset size:",
          round(output_file.stat().st_size / 1048576, 2),
          "MB")

    print("\nMaster dataset created successfully!")
    print("Original data remains unchanged.")


if __name__ == "__main__":
    main()
