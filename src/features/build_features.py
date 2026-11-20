
"""
EcoRoute AI
Feature Engineering Engine - Version 1.0

Purpose:
    Reconstruct vehicle timelines and derive
    physics-based features from measured telemetry.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "truck_telemetry_master.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "truck_telemetry_features.csv"
)

CONFIG_FILE = (
    ROOT / "configs" / "dataset_config.json"
)


def load_config():

    with open(CONFIG_FILE, "r") as file:
        return json.load(file)


def build_features(df, config):

    dt = config["sampling_interval_seconds"]

    # Preserve chronological order
    df = df.sort_values(
        ["trajectory_id", "sample_index"]
    ).reset_index(drop=True)

    # Verify observation sequences
    expected_index = df.groupby(
        "trajectory_id"
    ).cumcount()

    if not df["sample_index"].eq(expected_index).all():
        raise ValueError(
            "Invalid observation sequence"
        )

    # Time reconstruction
    df["elapsed_time_s"] = df["sample_index"] * dt

    # Speed normalization
    df["speed_kmh"] = (
        df["speed_raw"] * config["mph_to_kmh"]
    )

    df["speed_ms"] = (
        df["speed_raw"] * config["mph_to_ms"]
    )

    df["fuel_rate_lps"] = df["fuel_rate_raw"]

    df["distance_to_intersection_m"] = (
        df["distance_to_intersection_raw"]
    )

    groups = df.groupby(
        "trajectory_id", sort=False
    )

    # Previous speed
    previous_speed = groups["speed_ms"].shift(1)

    # Acceleration
    df["acceleration_ms2"] = (
        df["speed_ms"] - previous_speed
    ) / dt

    # Jerk
    df["jerk_ms3"] = (
        df.groupby("trajectory_id")[
            "acceleration_ms2"
        ].diff() / dt
    )

    # Distance travelled over each interval
    df["interval_distance_m"] = (
        (df["speed_ms"] + previous_speed)
        / 2 * dt
    ).fillna(0)

    df["cumulative_distance_m"] = (
        df.groupby("trajectory_id")[
            "interval_distance_m"
        ].cumsum()
    )

    # Fuel consumed over each interval
    previous_fuel_rate = groups[
        "fuel_rate_lps"
    ].shift(1)

    df["interval_fuel_l"] = (
        (df["fuel_rate_lps"] + previous_fuel_rate)
        / 2 * dt
    ).fillna(0)

    df["cumulative_fuel_l"] = (
        df.groupby("trajectory_id")[
            "interval_fuel_l"
        ].cumsum()
    )

    # Estimated CO2 emissions
    emission_factor = config[
        "diesel_co2_kg_per_litre"
    ]

    df["interval_co2_kg"] = (
        df["interval_fuel_l"] * emission_factor
    )

    df["cumulative_co2_kg"] = (
        df["cumulative_fuel_l"] * emission_factor
    )

    # Driving state classification
    thresholds = config["driving_state_thresholds"]

    conditions = [
        df["acceleration_ms2"].isna(),

        df["speed_kmh"] <=
        thresholds["stationary_speed_kmh"],

        df["acceleration_ms2"] >
        thresholds["acceleration_ms2"],

        df["acceleration_ms2"] <
        thresholds["deceleration_ms2"]
    ]

    states = [
        "unknown",
        "stationary",
        "accelerating",
        "decelerating"
    ]

    df["driving_state"] = np.select(
        conditions,
        states,
        default="cruising"
    )

    return df


def main():

    print("=" * 65)
    print("ECOROUTE AI - FEATURE ENGINEERING")
    print("=" * 65)

    config = load_config()

    if not config["metadata_verified"]:
        raise RuntimeError(
            "Verify sampling interval and units "
            "against the README before proceeding."
        )

    df = pd.read_csv(INPUT_FILE)

    print("Original rows:", len(df))
    print("Original columns:", len(df.columns))

    features = build_features(df, config)

    if len(features) != len(df):
        raise RuntimeError("Observation count changed")

    # Validate trajectory boundaries
    first_rows = features["sample_index"].eq(0)

    assert features.loc[
        first_rows, "acceleration_ms2"
    ].isna().all()

    assert features.loc[
        first_rows, "cumulative_distance_m"
    ].eq(0).all()

    assert features.loc[
        first_rows, "cumulative_fuel_l"
    ].eq(0).all()

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFEATURE DATASET SUMMARY")

    print("Total rows:", len(features))
    print("Total columns:", len(features.columns))

    print("Unique trajectories:",
          features["trajectory_id"].nunique())

    print("\nDRIVING STATES")
    print(features["driving_state"].value_counts())

    print("\nFEATURE STATISTICS")

    columns = [
        "speed_kmh",
        "acceleration_ms2",
        "jerk_ms3",
        "interval_distance_m",
        "interval_fuel_l"
    ]

    print(features[columns].describe())

    print("\nTOTAL ESTIMATED DISTANCE (km):")

    print(
        features["interval_distance_m"].sum()
        / 1000
    )

    print("\nTOTAL ESTIMATED FUEL (L):")

    print(features["interval_fuel_l"].sum())

    print("\nTOTAL ESTIMATED CO2 (kg):")

    print(features["interval_co2_kg"].sum())

    print("\nMISSING VALUES")
    print(features.isna().sum())

    print("\nSaved to:", OUTPUT_FILE)
    print("Feature engineering completed!")


if __name__ == "__main__":
    main()
