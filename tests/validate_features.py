
"""
EcoRoute AI
Feature Validation Engine - Version 1.0

Purpose:
    Verify the mathematical correctness and
    integrity of the engineered dataset.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"

MASTER_FILE = DATA_DIR / "truck_telemetry_master.csv"

FEATURE_FILE = DATA_DIR / "truck_telemetry_features.csv"

CONFIG_FILE = ROOT / "configs" / "dataset_config.json"

results = []


def check(name, condition):

    passed = bool(condition)

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {name}")

    results.append(passed)


def main():

    print("=" * 65)
    print("ECOROUTE AI - FEATURE VALIDATION")
    print("=" * 65)

    with open(CONFIG_FILE) as file:
        config = json.load(file)

    dt = config["sampling_interval_seconds"]

    emission_factor = config[
        "diesel_co2_kg_per_litre"
    ]

    master = pd.read_csv(MASTER_FILE)

    df = pd.read_csv(FEATURE_FILE)

    df = df.sort_values(
        ["trajectory_id", "sample_index"]
    ).reset_index(drop=True)

    groups = df.groupby("trajectory_id", sort=False)

    first = df["sample_index"].eq(0)

    first_two = df["sample_index"].lt(2)

    print("\nDATASET INTEGRITY\n")

    check(
        "Observation count unchanged",
        len(master) == len(df)
    )

    check(
        "Trajectory count unchanged",
        master["trajectory_id"].nunique()
        == df["trajectory_id"].nunique()
    )

    check(
        "Unique observation identifiers",
        not df.duplicated(
            ["trajectory_id", "sample_index"]
        ).any()
    )

    check(
        "Correct sample sequences",
        df["sample_index"].eq(
            groups.cumcount()
        ).all()
    )

    print("\nTIME AND MOTION VALIDATION\n")

    check(
        "Correct elapsed time",
        np.allclose(
            df["elapsed_time_s"],
            df["sample_index"] * dt
        )
    )

    check(
        "Correct speed conversion",
        np.allclose(
            df["speed_kmh"],
            df["speed_raw"] * 1.609344
        )
    )

    check(
        "Correct acceleration calculations",
        np.allclose(
            df["acceleration_ms2"],
            groups["speed_ms"].diff() / dt,
            equal_nan=True
        )
    )

    check(
        "Correct jerk calculations",
        np.allclose(
            df["jerk_ms3"],
            groups["acceleration_ms2"].diff() / dt,
            equal_nan=True
        )
    )

    check(
        "Acceleration undefined at boundaries",
        df["acceleration_ms2"].isna().eq(first).all()
    )

    check(
        "Jerk undefined at first two samples",
        df["jerk_ms3"].isna().eq(first_two).all()
    )

    print("\nDISTANCE VALIDATION\n")

    check(
        "No negative interval distance",
        df["interval_distance_m"].ge(0).all()
    )

    check(
        "Cumulative distance starts at zero",
        df.loc[first, "cumulative_distance_m"]
        .eq(0).all()
    )

    check(
        "Cumulative distance never decreases",
        groups["cumulative_distance_m"]
        .diff().dropna().ge(-1e-9).all()
    )

    expected_distance = (
        df["speed_ms"]
        + groups["speed_ms"].shift(1)
    ) / 2 * dt

    check(
        "Correct distance integration",
        np.allclose(
            df["interval_distance_m"],
            expected_distance.fillna(0)
        )
    )

    print("\nFUEL VALIDATION\n")

    check(
        "No negative interval fuel",
        df["interval_fuel_l"].ge(0).all()
    )

    check(
        "Cumulative fuel starts at zero",
        df.loc[first, "cumulative_fuel_l"]
        .eq(0).all()
    )

    check(
        "Cumulative fuel never decreases",
        groups["cumulative_fuel_l"]
        .diff().dropna().ge(-1e-9).all()
    )

    expected_fuel = (
        df["fuel_rate_lps"]
        + groups["fuel_rate_lps"].shift(1)
    ) / 2 * dt

    check(
        "Correct fuel integration",
        np.allclose(
            df["interval_fuel_l"],
            expected_fuel.fillna(0)
        )
    )

    print("\nEMISSION VALIDATION\n")

    check(
        "Correct interval CO2 calculations",
        np.allclose(
            df["interval_co2_kg"],
            df["interval_fuel_l"] * emission_factor
        )
    )

    check(
        "Correct cumulative CO2 calculations",
        np.allclose(
            df["cumulative_co2_kg"],
            df["cumulative_fuel_l"] * emission_factor
        )
    )

    print("\nDRIVING STATE VALIDATION\n")

    valid_states = [
        "unknown",
        "stationary",
        "accelerating",
        "decelerating",
        "cruising"
    ]

    check(
        "Valid driving state categories",
        df["driving_state"].isin(valid_states).all()
    )

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    passed = sum(results)
    total = len(results)

    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", total - passed)

    if passed == total:
        print("\nALL VALIDATION TESTS PASSED!")
        print("Dataset ready for exploratory analysis.")

    else:
        print("\nVALIDATION FAILED.")
        print("Investigate before proceeding.")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
