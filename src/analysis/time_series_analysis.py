
"""
EcoRoute AI
Time-Series Telemetry Analysis
Version 1.0

Purpose:
    Reconstruct recorded truck trajectories,
    analyse driving behaviour through time,
    and generate journey-level statistics.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from visualization.plot_theme import (
    COLORS,
    apply_theme,
    save_plot
)


DATA_FILE = (
    ROOT / "data" / "processed"
    / "truck_telemetry_features.csv"
)

FIGURE_DIR = ROOT / "assets" / "figures" / "eda"

REPORT_DIR = ROOT / "reports" / "eda"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

apply_theme()


def load_data():

    df = pd.read_csv(DATA_FILE)

    df = df.sort_values(
        ["trajectory_id", "sample_index"]
    ).reset_index(drop=True)

    df["fuel_rate_lph"] = (
        df["fuel_rate_lps"] * 3600
    )

    return df


def build_trajectory_summary(df):

    summary = (
        df.groupby("trajectory_id")
        .agg(
            scenario=("scenario", "first"),
            observations=("sample_index", "size"),
            duration_s=("elapsed_time_s", "max"),
            distance_m=("cumulative_distance_m", "last"),
            total_fuel_l=("cumulative_fuel_l", "last"),
            total_co2_kg=("cumulative_co2_kg", "last"),
            mean_speed_kmh=("speed_kmh", "mean"),
            max_speed_kmh=("speed_kmh", "max"),
            mean_fuel_lph=("fuel_rate_lph", "mean"),
            peak_fuel_lph=("fuel_rate_lph", "max"),
            mean_acceleration=("acceleration_ms2", "mean"),
            mean_absolute_jerk=(
                "jerk_ms3",
                lambda x: x.abs().mean()
            )
        )
        .reset_index()
    )

    summary["distance_km"] = (
        summary["distance_m"] / 1000
    )

    summary["fuel_l_per_100km"] = np.divide(
        summary["total_fuel_l"] * 100,
        summary["distance_km"],
        out=np.full(len(summary), np.nan),
        where=summary["distance_km"] > 0
    )

    summary["co2_g_per_km"] = np.divide(
        summary["total_co2_kg"] * 1000,
        summary["distance_km"],
        out=np.full(len(summary), np.nan),
        where=summary["distance_km"] > 0
    )

    summary.to_csv(
        REPORT_DIR / "trajectory_summary.csv",
        index=False
    )

    print("\nTRAJECTORY SUMMARY CREATED")
    print("Total trajectories:", len(summary))

    print(
        "\nTotal recorded distance (km):",
        round(summary["distance_km"].sum(), 3)
    )

    print(
        "Total recorded fuel (L):",
        round(summary["total_fuel_l"].sum(), 3)
    )

    print(
        "Estimated total CO2 (kg):",
        round(summary["total_co2_kg"].sum(), 3)
    )

    return summary


def select_representative_trajectory(
    summary,
    scenario
):

    group = summary[
        (summary["scenario"] == scenario)
        & (summary["distance_km"] > 0)
        & (summary["duration_s"] > 0)
    ].copy()

    if group.empty:
        raise ValueError(
            f"No valid trajectories: {scenario}"
        )

    median_distance = group[
        "distance_km"
    ].median()

    group["distance_difference"] = (
        group["distance_km"] - median_distance
    ).abs()

    selected = group.sort_values(
        ["distance_difference", "trajectory_id"]
    ).iloc[0]

    return selected["trajectory_id"]


def plot_trajectory(df, trajectory_id, scenario):

    data = df[
        df["trajectory_id"] == trajectory_id
    ].copy()

    time = data["elapsed_time_s"]

    fig, axes = plt.subplots(
        5, 1,
        figsize=(15, 17),
        sharex=True
    )

    features = [
        (
            "speed_kmh",
            "Vehicle Speed",
            "km/h",
            COLORS["baseline"]
        ),
        (
            "acceleration_ms2",
            "Vehicle Acceleration",
            "m/s²",
            COLORS["eco"]
        ),
        (
            "fuel_rate_lph",
            "Instantaneous Fuel Rate",
            "L/hour",
            COLORS["fuel"]
        ),
        (
            "cumulative_fuel_l",
            "Cumulative Fuel Consumption",
            "Litres",
            COLORS["warning"]
        ),
        (
            "cumulative_co2_kg",
            "Cumulative CO2 Emissions",
            "kg",
            COLORS["purple"]
        )
    ]

    for ax, feature in zip(axes, features):

        column, title, unit, color = feature

        ax.plot(
            time,
            data[column],
            color=color,
            linewidth=1.7
        )

        ax.fill_between(
            time,
            0,
            data[column],
            color=color,
            alpha=0.10
        )

        ax.set_title(
            title,
            loc="left",
            fontsize=13
        )

        ax.set_ylabel(unit)

    axes[1].axhline(
        0,
        color=COLORS["muted"],
        linestyle="--"
    )

    axes[-1].set_xlabel(
        "Elapsed time (seconds)"
    )

    fig.suptitle(
        f"{scenario.title()} Driving Telemetry",
        fontsize=20
    )

    save_plot(
        fig,
        FIGURE_DIR /
        f"{scenario}_trajectory_telemetry.png"
    )

    print("\nSELECTED TRAJECTORY:", scenario)
    print(trajectory_id)

    print(
        "Duration (s):",
        data["elapsed_time_s"].max()
    )

    print(
        "Distance (m):",
        round(
            data["cumulative_distance_m"].iloc[-1],
            3
        )
    )

    print(
        "Fuel consumed (L):",
        round(
            data["cumulative_fuel_l"].iloc[-1],
            4
        )
    )

    print(
        f"Saved: {scenario}_trajectory_telemetry.png"
    )


def plot_intersection_behaviour(
    df,
    trajectory_id,
    scenario
):

    data = df[
        df["trajectory_id"] == trajectory_id
    ].copy()

    fig, axes = plt.subplots(
        3, 1,
        figsize=(13, 10),
        sharex=True
    )

    columns = [
        (
            "distance_to_intersection_m",
            "Distance to Intersection (m)",
            COLORS["purple"]
        ),
        (
            "speed_kmh",
            "Vehicle Speed (km/h)",
            COLORS["baseline"]
        ),
        (
            "fuel_rate_lph",
            "Fuel Rate (L/hour)",
            COLORS["fuel"]
        )
    ]

    for ax, (column, label, color) in zip(
        axes,
        columns
    ):

        ax.plot(
            data["elapsed_time_s"],
            data[column],
            color=color,
            linewidth=2
        )

        ax.set_ylabel(label)

    axes[-1].set_xlabel(
        "Elapsed time (seconds)"
    )

    fig.suptitle(
        f"{scenario.title()} Intersection Behaviour",
        fontsize=18
    )

    save_plot(
        fig,
        FIGURE_DIR /
        f"{scenario}_intersection_behaviour.png"
    )

    print(
        f"Saved: {scenario}_intersection_behaviour.png"
    )


def plot_trajectory_fuel_distance(summary):

    fig, ax = plt.subplots(figsize=(12, 7))

    for scenario, color in [
        ("baseline", COLORS["baseline"]),
        ("eco", COLORS["eco"])
    ]:

        group = summary[
            summary["scenario"] == scenario
        ]

        ax.scatter(
            group["distance_km"],
            group["total_fuel_l"],
            color=color,
            alpha=0.60,
            s=30,
            label=scenario.title()
        )

    ax.set_title(
        "Recorded Distance vs Total Fuel Consumption"
    )

    ax.set_xlabel("Recorded distance (km)")

    ax.set_ylabel("Total fuel consumed (L)")

    ax.legend()

    save_plot(
        fig,
        FIGURE_DIR /
        "trajectory_distance_vs_fuel.png"
    )

    print(
        "Saved: trajectory_distance_vs_fuel.png"
    )


def analyse_stationary_behaviour(df):

    # Identify intervals where both boundary
    # observations are classified as stationary.

    previous_state = (
        df.groupby("trajectory_id")[
            "driving_state"
        ].shift(1)
    )

    stationary = (
        df["driving_state"].eq("stationary")
        & previous_state.eq("stationary")
    )

    data = df[stationary].copy()

    summary = (
        data.groupby(
            ["trajectory_id", "scenario"]
        )
        .agg(
            stationary_intervals=(
                "sample_index", "size"
            ),
            stationary_fuel_l=(
                "interval_fuel_l", "sum"
            ),
            stationary_co2_kg=(
                "interval_co2_kg", "sum"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        REPORT_DIR /
        "stationary_fuel_summary.csv",
        index=False
    )

    totals = (
        summary.groupby("scenario")
        [
            [
                "stationary_intervals",
                "stationary_fuel_l",
                "stationary_co2_kg"
            ]
        ]
        .sum()
    )

    print("\nSTATIONARY BEHAVIOUR")

    print(totals.round(4).to_string())

    if totals.empty:
        print("No stationary intervals found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {
        "baseline": COLORS["baseline"],
        "eco": COLORS["eco"]
    }

    ax.bar(
        totals.index,
        totals["stationary_fuel_l"],
        color=[
            palette.get(s, COLORS["muted"])
            for s in totals.index
        ]
    )

    ax.set_title(
        "Total Recorded Stationary Fuel Consumption"
    )

    ax.set_ylabel("Fuel consumed (L)")

    ax.set_xlabel("Driving scenario")

    save_plot(
        fig,
        FIGURE_DIR /
        "stationary_fuel_consumption.png"
    )

    print(
        "Saved: stationary_fuel_consumption.png"
    )


def main():

    print("=" * 65)
    print("ECOROUTE AI - TIME SERIES ANALYSIS")
    print("=" * 65)

    df = load_data()

    print("\nObservations:", len(df))

    print(
        "Trajectories:",
        df["trajectory_id"].nunique()
    )

    summary = build_trajectory_summary(df)

    print("\nGENERATING TELEMETRY VISUALIZATIONS")

    for scenario in ["baseline", "eco"]:

        trajectory_id = (
            select_representative_trajectory(
                summary,
                scenario
            )
        )

        plot_trajectory(
            df,
            trajectory_id,
            scenario
        )

        plot_intersection_behaviour(
            df,
            trajectory_id,
            scenario
        )

    plot_trajectory_fuel_distance(summary)

    analyse_stationary_behaviour(df)

    print("\nFigures saved to:", FIGURE_DIR)

    print("Reports saved to:", REPORT_DIR)

    print("\nTIME SERIES ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
