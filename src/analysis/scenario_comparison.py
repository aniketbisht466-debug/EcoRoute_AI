
"""
EcoRoute AI
Baseline vs Eco-driving Comparison
Version 1.0

Purpose:
    Compare fuel efficiency, carbon emissions,
    driving behaviour and comparable conditions.
"""

from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

SUMMARY_FILE = (
    ROOT / "reports" / "eda"
    / "trajectory_summary.csv"
)

FIGURE_DIR = ROOT / "assets" / "figures" / "eda"
REPORT_DIR = ROOT / "reports" / "eda"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

apply_theme()

PALETTE = {
    "baseline": COLORS["baseline"],
    "eco": COLORS["eco"]
}

SCENARIOS = ["baseline", "eco"]

# Exclude very short trajectories from
# distance-normalized comparisons.
MIN_DISTANCE_KM = 0.1

# Minimum recordings per scenario
# for comparable-condition groups.
MIN_GROUP_SIZE = 5


def load_data():

    telemetry = pd.read_csv(DATA_FILE)

    journeys = pd.read_csv(SUMMARY_FILE)

    valid = journeys[
        (journeys["distance_km"] >= MIN_DISTANCE_KM)
        & (journeys["duration_s"] > 0)
        & journeys["scenario"].isin(SCENARIOS)
    ].copy()

    print("\nDATASET SUMMARY")
    print("Telemetry observations:", len(telemetry))
    print("Original trajectories:", len(journeys))
    print("Eligible trajectories:", len(valid))
    print("Excluded trajectories:", len(journeys) - len(valid))

    return telemetry, valid


def calculate_savings(baseline, eco):

    if baseline <= 0:
        return np.nan

    return (baseline - eco) / baseline * 100


def build_scenario_summary(journeys):

    records = []

    for scenario in SCENARIOS:

        group = journeys[
            journeys["scenario"] == scenario
        ]

        if group.empty:
            raise ValueError(
                f"No eligible trajectories: {scenario}"
            )

        distance = group["distance_km"].sum()
        fuel = group["total_fuel_l"].sum()
        co2 = group["total_co2_kg"].sum()
        duration = group["duration_s"].sum()

        records.append({
            "scenario": scenario,
            "trajectories": len(group),
            "total_distance_km": distance,
            "total_fuel_l": fuel,
            "total_co2_kg": co2,
            "weighted_fuel_l_per_100km":
                fuel / distance * 100,
            "weighted_fuel_efficiency_kmpl":
                distance / fuel if fuel > 0 else np.nan,
            "weighted_co2_g_per_km":
                co2 * 1000 / distance,
            "overall_speed_kmh":
                distance / (duration / 3600),
            "median_fuel_l_per_100km":
                group["fuel_l_per_100km"].median(),
            "mean_duration_s":
                group["duration_s"].mean()
        })

    summary = pd.DataFrame(records)

    summary.to_csv(
        REPORT_DIR / "scenario_comparison.csv",
        index=False
    )

    print("\nSCENARIO COMPARISON")

    print(summary.round(4).to_string(index=False))

    return summary


def build_behaviour_summary(telemetry, journeys):

    data = telemetry[
        telemetry["trajectory_id"].isin(
            journeys["trajectory_id"]
        )
    ].copy()

    data = data.sort_values(
        ["trajectory_id", "sample_index"]
    )

    data["absolute_acceleration"] = (
        data["acceleration_ms2"].abs()
    )

    data["absolute_jerk"] = (
        data["jerk_ms3"].abs()
    )

    data["interval_seconds"] = (
        data.groupby("trajectory_id")[
            "elapsed_time_s"
        ].diff().fillna(0)
    )

    previous_state = (
        data.groupby("trajectory_id")[
            "driving_state"
        ].shift(1)
    )

    data["stationary_seconds"] = np.where(
        data["driving_state"].eq("stationary")
        & previous_state.eq("stationary"),
        data["interval_seconds"],
        0
    )

    behaviour = (
        data.groupby(
            ["trajectory_id", "scenario"]
        )
        .agg(
            mean_absolute_acceleration=(
                "absolute_acceleration", "mean"
            ),
            mean_absolute_jerk=(
                "absolute_jerk", "mean"
            ),
            stationary_seconds=(
                "stationary_seconds", "sum"
            ),
            duration_seconds=(
                "interval_seconds", "sum"
            ),
            mean_speed_kmh=(
                "speed_kmh", "mean"
            )
        )
        .reset_index()
    )

    behaviour["stationary_percent"] = np.where(
        behaviour["duration_seconds"] > 0,
        behaviour["stationary_seconds"]
        / behaviour["duration_seconds"] * 100,
        np.nan
    )

    behaviour.to_csv(
        REPORT_DIR / "scenario_driving_behaviour.csv",
        index=False
    )

    return behaviour


def plot_metric_comparison(summary):

    metrics = [
        (
            "weighted_fuel_l_per_100km",
            "Fuel Consumption",
            "L/100 km",
            "scenario_weighted_fuel.png"
        ),
        (
            "weighted_co2_g_per_km",
            "Carbon Emission Intensity",
            "g CO2/km",
            "scenario_emission_intensity.png"
        ),
        (
            "overall_speed_kmh",
            "Distance-Time Average Speed",
            "km/h",
            "scenario_average_speed.png"
        )
    ]

    for column, title, unit, filename in metrics:

        fig, ax = plt.subplots(figsize=(9, 6))

        bars = ax.bar(
            summary["scenario"].str.title(),
            summary[column],
            color=[
                PALETTE[s]
                for s in summary["scenario"]
            ],
            width=0.5
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=5
        )

        ax.set_ylim(
            0,
            max(summary[column].max() * 1.2, 1)
        )

        ax.set_title(title)
        ax.set_ylabel(unit)

        save_plot(fig, FIGURE_DIR / filename)

        print("Saved:", filename)


def plot_fuel_distribution(journeys):

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.boxplot(
        data=journeys,
        x="scenario",
        y="fuel_l_per_100km",
        hue="scenario",
        order=SCENARIOS,
        palette=PALETTE,
        legend=False,
        showfliers=True,
        ax=ax
    )

    ax.set_title(
        "Journey-Level Fuel Consumption Distribution"
    )

    ax.set_xlabel("Driving scenario")
    ax.set_ylabel("Fuel consumption (L/100 km)")

    save_plot(
        fig,
        FIGURE_DIR / "scenario_fuel_distribution.png"
    )

    print("Saved: scenario_fuel_distribution.png")


def plot_behaviour_comparison(behaviour):

    metrics = [
        (
            "mean_absolute_acceleration",
            "Absolute Acceleration",
            "m/s²"
        ),
        (
            "mean_absolute_jerk",
            "Absolute Jerk",
            "m/s³"
        ),
        (
            "stationary_percent",
            "Stationary Time",
            "%"
        )
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    for ax, (column, title, unit) in zip(
        axes, metrics
    ):

        sns.boxplot(
            data=behaviour,
            x="scenario",
            y=column,
            hue="scenario",
            order=SCENARIOS,
            palette=PALETTE,
            legend=False,
            showfliers=False,
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(unit)

    fig.suptitle(
        "Baseline vs Eco-driving Behaviour",
        fontsize=18
    )

    save_plot(
        fig,
        FIGURE_DIR / "scenario_behaviour_comparison.png"
    )

    print("Saved: scenario_behaviour_comparison.png")


def extract_recording_metadata(trajectory_id):

    # Supports identifiers containing names such as:
    # 20190830VNL491_Alameda_N_0_baseline.csv

    pattern = (
        r"(?P<date>\d{8})"
        r"(?P<vehicle>[^_/\\]+)_"
        r"(?P<location>.+?)_"
        r"(?P<direction>[NSEW]+)_"
        r"(?P<run>\d+)_"
        r"(?P<scenario>baseline|eco)"
    )

    match = re.search(
        pattern,
        str(trajectory_id),
        flags=re.IGNORECASE
    )

    if match is None:
        return {
            "vehicle": None,
            "location": None,
            "direction": None
        }

    return {
        "vehicle": match.group("vehicle"),
        "location": match.group("location"),
        "direction": match.group("direction")
    }


def compare_conditions(journeys):

    data = journeys.copy()

    metadata = pd.DataFrame(
        data["trajectory_id"]
        .apply(extract_recording_metadata)
        .tolist(),
        index=data.index
    )

    data = pd.concat(
        [data, metadata],
        axis=1
    )

    data = data.dropna(
        subset=["vehicle", "location", "direction"]
    )

    print("\nCOMPARABLE CONDITIONS")
    print("Metadata identified:", len(data))
    print(
        "Unidentified recordings:",
        len(journeys) - len(data)
    )

    if data.empty:
        print(
            "Metadata unavailable in trajectory identifiers."
        )
        print(
            "Inspect source metadata before matched analysis."
        )
        return pd.DataFrame()

    records = []

    group_columns = [
        "vehicle",
        "location",
        "direction"
    ]

    for keys, group in data.groupby(group_columns):

        baseline = group[
            group["scenario"] == "baseline"
        ]

        eco = group[
            group["scenario"] == "eco"
        ]

        if (
            len(baseline) < MIN_GROUP_SIZE
            or len(eco) < MIN_GROUP_SIZE
        ):
            continue

        baseline_distance = (
            baseline["distance_km"].sum()
        )

        eco_distance = (
            eco["distance_km"].sum()
        )

        baseline_rate = (
            baseline["total_fuel_l"].sum()
            / baseline_distance * 100
        )

        eco_rate = (
            eco["total_fuel_l"].sum()
            / eco_distance * 100
        )

        records.append({
            "vehicle": keys[0],
            "location": keys[1],
            "direction": keys[2],
            "baseline_trajectories": len(baseline),
            "eco_trajectories": len(eco),
            "baseline_distance_km": baseline_distance,
            "eco_distance_km": eco_distance,
            "baseline_l_per_100km": baseline_rate,
            "eco_l_per_100km": eco_rate,
            "fuel_savings_percent": calculate_savings(
                baseline_rate, eco_rate
            )
        })

    comparison = pd.DataFrame(records)

    if comparison.empty:
        print("No groups meet minimum support.")
        return comparison

    comparison = comparison.sort_values(
        "fuel_savings_percent",
        ascending=False
    )

    comparison.to_csv(
        REPORT_DIR / "comparable_conditions.csv",
        index=False
    )

    print(comparison.round(4).to_string(index=False))

    return comparison


def plot_condition_savings(comparison):

    if comparison.empty:
        return

    data = comparison.dropna(
        subset=["fuel_savings_percent"]
    ).copy()

    if data.empty:
        return

    data = data.sort_values(
        "fuel_savings_percent"
    )

    labels = (
        data["vehicle"] + " | "
        + data["location"] + " | "
        + data["direction"]
    )

    values = data["fuel_savings_percent"]

    colors = [
        COLORS["eco"] if v >= 0 else "#EF4444"
        for v in values
    ]

    fig, ax = plt.subplots(
        figsize=(12, max(6, len(data) * 0.45))
    )

    ax.barh(
        labels,
        values,
        color=colors
    )

    ax.axvline(
        0,
        color=COLORS["muted"],
        linestyle="--"
    )

    ax.set_title(
        "Observed Fuel Savings by Comparable Conditions"
    )

    ax.set_xlabel("Relative fuel savings (%)")

    save_plot(
        fig,
        FIGURE_DIR / "comparable_condition_savings.png"
    )

    print("Saved: comparable_condition_savings.png")


def print_final_findings(summary):

    baseline = summary[
        summary["scenario"] == "baseline"
    ].iloc[0]

    eco = summary[
        summary["scenario"] == "eco"
    ].iloc[0]

    fuel_savings = calculate_savings(
        baseline["weighted_fuel_l_per_100km"],
        eco["weighted_fuel_l_per_100km"]
    )

    co2_savings = calculate_savings(
        baseline["weighted_co2_g_per_km"],
        eco["weighted_co2_g_per_km"]
    )

    print("\n" + "=" * 65)
    print("FINAL DESCRIPTIVE FINDINGS")
    print("=" * 65)

    print(
        "Baseline fuel consumption:",
        round(baseline["weighted_fuel_l_per_100km"], 4),
        "L/100 km"
    )

    print(
        "Eco-driving fuel consumption:",
        round(eco["weighted_fuel_l_per_100km"], 4),
        "L/100 km"
    )

    print(
        "Observed relative fuel savings:",
        round(fuel_savings, 4),
        "%"
    )

    print(
        "Observed relative CO2 savings:",
        round(co2_savings, 4),
        "%"
    )

    print(
        "\nThese are descriptive associations."
    )

    print(
        "Statistical significance and causal effects "
        "have not been established."
    )


def main():

    print("=" * 65)
    print("ECOROUTE AI - SCENARIO COMPARISON")
    print("=" * 65)

    telemetry, journeys = load_data()

    summary = build_scenario_summary(journeys)

    behaviour = build_behaviour_summary(
        telemetry, journeys
    )

    plot_metric_comparison(summary)

    plot_fuel_distribution(journeys)

    plot_behaviour_comparison(behaviour)

    comparison = compare_conditions(journeys)

    plot_condition_savings(comparison)

    print_final_findings(summary)

    print("\nSCENARIO COMPARISON COMPLETED!")


if __name__ == "__main__":
    main()
