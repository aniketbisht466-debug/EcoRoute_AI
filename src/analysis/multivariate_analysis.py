
"""
EcoRoute AI
Multivariate Exploratory Data Analysis
Version 1.0

Purpose:
    Investigate combined effects of speed,
    acceleration and driving behaviour on fuel rate.
"""

from pathlib import Path
import sys

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

FIGURE_DIR = ROOT / "assets" / "figures" / "eda"

REPORT_DIR = ROOT / "reports" / "eda"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

apply_theme()

MIN_OBSERVATIONS = 30
MIN_TRAJECTORIES = 5


def prepare_data(df):

    columns = [
        "trajectory_id",
        "speed_kmh",
        "acceleration_ms2",
        "fuel_rate_lps",
        "driving_state",
        "scenario"
    ]

    data = df[columns].dropna().copy()

    data["fuel_rate_lph"] = (
        data["fuel_rate_lps"] * 3600
    )

    # Speed intervals of 5 km/h
    data["speed_bin"] = (
        np.floor(data["speed_kmh"] / 5) * 5
    )

    # Acceleration intervals of 0.25 m/s²
    data["acceleration_bin"] = (
        np.floor(
            data["acceleration_ms2"] / 0.25
        ) * 0.25
    )

    return data


def build_fuel_map(data):

    summary = (
        data.groupby(
            ["speed_bin", "acceleration_bin"]
        )
        .agg(
            mean_fuel_lph=("fuel_rate_lph", "mean"),
            median_fuel_lph=("fuel_rate_lph", "median"),
            std_fuel_lph=("fuel_rate_lph", "std"),
            observations=("fuel_rate_lph", "size"),
            trajectories=("trajectory_id", "nunique")
        )
        .reset_index()
    )

    summary["reliable_bin"] = (
        (summary["observations"] >= MIN_OBSERVATIONS)
        & (summary["trajectories"] >= MIN_TRAJECTORIES)
    )

    summary.to_csv(
        REPORT_DIR / "multivariate_fuel_map.csv",
        index=False
    )

    return summary


def plot_fuel_heatmap(summary):

    supported = summary[
        summary["reliable_bin"]
    ]

    if supported.empty:
        print("No sufficiently supported fuel-map bins.")
        return

    pivot = supported.pivot(
        index="acceleration_bin",
        columns="speed_bin",
        values="mean_fuel_lph"
    ).sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 9))

    sns.heatmap(
        pivot,
        cmap="inferno",
        mask=pivot.isna(),
        cbar_kws={
            "label": "Mean fuel rate (L/hour)"
        },
        ax=ax
    )

    ax.set_title(
        "Fuel Consumption Map: Speed and Acceleration"
    )

    ax.set_xlabel("Speed bin lower edge (km/h)")

    ax.set_ylabel("Acceleration bin lower edge (m/s²)")

    save_plot(
        fig,
        FIGURE_DIR / "multivariate_fuel_heatmap.png"
    )

    print("Saved: multivariate_fuel_heatmap.png")


def plot_driving_density(data):

    fig, ax = plt.subplots(figsize=(12, 8))

    density = ax.hexbin(
        data["speed_kmh"],
        data["acceleration_ms2"],
        gridsize=65,
        bins="log",
        mincnt=1,
        cmap="viridis"
    )

    colorbar = fig.colorbar(density, ax=ax)

    colorbar.set_label(
        "Observation count (log scale)"
    )

    ax.axhline(
        0,
        linestyle="--",
        color=COLORS["muted"]
    )

    ax.set_title("Driving Condition Density")

    ax.set_xlabel("Speed (km/h)")

    ax.set_ylabel("Acceleration (m/s²)")

    save_plot(
        fig,
        FIGURE_DIR / "driving_condition_density.png"
    )

    print("Saved: driving_condition_density.png")


def plot_fuel_surface(summary):

    supported = summary[
        summary["reliable_bin"]
    ]

    if supported.empty:
        print("Insufficient data for fuel surface.")
        return

    pivot = supported.pivot(
        index="acceleration_bin",
        columns="speed_bin",
        values="mean_fuel_lph"
    ).sort_index()

    # Regular grid; missing cells remain missing.
    speed_bins = np.arange(
        pivot.columns.min(),
        pivot.columns.max() + 5,
        5
    )

    acceleration_bins = np.arange(
        pivot.index.min(),
        pivot.index.max() + 0.25,
        0.25
    )

    pivot = pivot.reindex(
        index=acceleration_bins,
        columns=speed_bins
    )

    x, y = np.meshgrid(
        speed_bins + 2.5,
        acceleration_bins + 0.125
    )

    z = pivot.to_numpy()

    fig = plt.figure(figsize=(13, 10))

    ax = fig.add_subplot(111, projection="3d")

    ax.xaxis.pane.set_facecolor(COLORS["surface"])
    ax.yaxis.pane.set_facecolor(COLORS["surface"])
    ax.zaxis.pane.set_facecolor(COLORS["surface"])

    # Plot measured bin means without inventing
    # values for unobserved driving conditions.
    valid = np.isfinite(z)

    surface = ax.scatter(
        x[valid],
        y[valid],
        z[valid],
        c=z[valid],
        cmap="inferno",
        s=25,
        depthshade=False
    )

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.65,
        label="Mean fuel rate (L/hour)"
    )

    ax.set_title("3D Fuel Consumption Response Map")

    ax.set_xlabel("Speed (km/h)")

    ax.set_ylabel("Acceleration (m/s²)")

    ax.set_zlabel("Mean fuel rate (L/hour)")

    ax.view_init(elev=28, azim=-135)

    save_plot(
        fig,
        FIGURE_DIR / "fuel_response_3d.png"
    )

    print("Saved: fuel_response_3d.png")


def plot_driving_state_comparison(data):

    states = [
        "stationary",
        "decelerating",
        "cruising",
        "accelerating"
    ]

    valid = data[
        data["driving_state"].isin(states)
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.boxplot(
        data=valid,
        x="driving_state",
        y="fuel_rate_lph",
        hue="scenario",
        order=states,
        palette={
            "baseline": COLORS["baseline"],
            "eco": COLORS["eco"]
        },
        showfliers=False,
        ax=ax
    )

    ax.set_title(
        "Fuel Consumption Across Driving States"
    )

    ax.set_xlabel("Driving state")

    ax.set_ylabel("Fuel rate (L/hour)")

    ax.legend(title="Driving scenario")

    save_plot(
        fig,
        FIGURE_DIR / "driving_state_fuel_comparison.png"
    )

    print("Saved: driving_state_fuel_comparison.png")

    summary = (
        valid.groupby(
            ["scenario", "driving_state"]
        )
        .agg(
            mean_fuel_lph=("fuel_rate_lph", "mean"),
            median_fuel_lph=("fuel_rate_lph", "median"),
            observations=("fuel_rate_lph", "size"),
            trajectories=("trajectory_id", "nunique")
        )
        .reset_index()
    )

    summary.to_csv(
        REPORT_DIR / "driving_state_fuel_summary.csv",
        index=False
    )

    print("\nDRIVING STATE FUEL SUMMARY")

    print(summary.round(4).to_string(index=False))


def plot_scenario_fuel_maps(data):

    scenarios = ["baseline", "eco"]

    maps = {}

    for scenario in scenarios:

        group = data[
            data["scenario"] == scenario
        ]

        summary = (
            group.groupby(
                ["acceleration_bin", "speed_bin"]
            )
            .agg(
                fuel=("fuel_rate_lph", "mean"),
                count=("fuel_rate_lph", "size"),
                trajectories=("trajectory_id", "nunique")
            )
            .reset_index()
        )

        summary = summary[
            (summary["count"] >= MIN_OBSERVATIONS)
            & (summary["trajectories"] >= MIN_TRAJECTORIES)
        ]

        if summary.empty:
            print("Insufficient supported bins:", scenario)
            return

        maps[scenario] = summary.pivot(
            index="acceleration_bin",
            columns="speed_bin",
            values="fuel"
        )

    # Both maps use identical axes and color scales.
    rows = (
        maps["baseline"].index
        .union(maps["eco"].index)
        .sort_values(ascending=False)
    )

    cols = (
        maps["baseline"].columns
        .union(maps["eco"].columns)
        .sort_values()
    )

    for scenario in scenarios:

        maps[scenario] = maps[scenario].reindex(
            index=rows,
            columns=cols
        )

    vmax = max(
        np.nanmax(maps[s].to_numpy())
        for s in scenarios
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(19, 8)
    )

    for ax, scenario in zip(axes, scenarios):

        pivot = maps[scenario]

        sns.heatmap(
            pivot,
            mask=pivot.isna(),
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            cbar_kws={
                "label": "Mean fuel rate (L/hour)"
            },
            ax=ax
        )

        ax.set_title(scenario.title())

        ax.set_xlabel("Speed bin lower edge (km/h)")

        ax.set_ylabel(
            "Acceleration bin lower edge (m/s²)"
        )

    fig.suptitle(
        "Baseline vs Eco-driving Fuel Maps",
        fontsize=18
    )

    save_plot(
        fig,
        FIGURE_DIR / "scenario_fuel_maps.png"
    )

    print("Saved: scenario_fuel_maps.png")


def identify_high_fuel_conditions(summary):

    reliable = summary[
        summary["reliable_bin"]
    ]

    highest = reliable.sort_values(
        "mean_fuel_lph",
        ascending=False
    ).head(10)

    highest.to_csv(
        REPORT_DIR / "high_fuel_conditions.csv",
        index=False
    )

    print("\nTOP 10 HIGH FUEL DEMAND CONDITIONS")

    print(
        highest[
            [
                "speed_bin",
                "acceleration_bin",
                "mean_fuel_lph",
                "observations",
                "trajectories"
            ]
        ].round(4).to_string(index=False)
    )


def main():

    print("=" * 65)
    print("ECOROUTE AI - MULTIVARIATE ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(DATA_FILE)

    data = prepare_data(df)

    print("\nOriginal observations:", len(df))

    print("Analysis observations:", len(data))

    print(
        "Undefined or missing observations:",
        len(df) - len(data)
    )

    print(
        "Trajectories:",
        data["trajectory_id"].nunique()
    )

    summary = build_fuel_map(data)

    print("\nTotal occupied driving condition bins:", len(summary))

    print(
        "Sufficiently supported bins:",
        summary["reliable_bin"].sum()
    )

    print("\nGENERATING VISUALIZATIONS\n")

    plot_fuel_heatmap(summary)

    plot_driving_density(data)

    plot_fuel_surface(summary)

    plot_driving_state_comparison(data)

    plot_scenario_fuel_maps(data)

    identify_high_fuel_conditions(summary)

    print("\nFigures saved to:", FIGURE_DIR)

    print("Reports saved to:", REPORT_DIR)

    print("\nMULTIVARIATE ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
