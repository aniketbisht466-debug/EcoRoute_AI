
"""
EcoRoute AI
Bivariate Exploratory Data Analysis
Version 1.0

Purpose:
    Investigate relationships between vehicle
    telemetry and fuel consumption.
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

TARGET = "fuel_rate_lps"

FEATURES = {
    "speed_kmh": "Speed (km/h)",
    "acceleration_ms2": "Acceleration (m/s²)",
    "jerk_ms3": "Jerk (m/s³)",
    "distance_to_intersection_m":
        "Distance to intersection (m)"
}


def plot_relationship(df, column, label):

    data = df[[column, TARGET]].dropna()

    fig, ax = plt.subplots(figsize=(11, 7))

    # Hexagonal density plot
    density = ax.hexbin(
        data[column],
        data[TARGET] * 3600,
        gridsize=65,
        bins="log",
        cmap="viridis",
        mincnt=1
    )

    colorbar = fig.colorbar(density, ax=ax)

    colorbar.set_label(
        "Observations per hexagon (log scale)"
    )

    pearson = data[column].corr(
        data[TARGET],
        method="pearson"
    )

    spearman = data[column].corr(
        data[TARGET],
        method="spearman"
    )

    ax.set_title(
        f"{label.split(' (')[0]} vs Fuel Consumption"
    )

    ax.set_xlabel(label)

    ax.set_ylabel("Fuel rate (L/hour)")

    ax.text(
        0.03,
        0.97,
        f"Pearson: {pearson:.3f}\n"
        f"Spearman: {spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={
            "facecolor": COLORS["background"],
            "edgecolor": COLORS["grid"],
            "alpha": 0.9
        }
    )

    output = (
        FIGURE_DIR / f"{column}_vs_fuel.png"
    )

    save_plot(fig, output)

    print("Saved:", output.name)


def plot_correlation_matrices(df):

    columns = list(FEATURES.keys()) + [TARGET]

    data = df[columns]

    labels = [
        "Speed",
        "Acceleration",
        "Jerk",
        "Intersection distance",
        "Fuel rate"
    ]

    for method in ["pearson", "spearman"]:

        correlation = data.corr(method=method)

        correlation.to_csv(
            REPORT_DIR /
            f"{method}_correlation.csv"
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".3f",
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_title(
            f"{method.title()} Correlation Matrix"
        )

        plt.setp(
            ax.get_xticklabels(),
            rotation=35,
            ha="right"
        )

        output = (
            FIGURE_DIR /
            f"{method}_correlation_heatmap.png"
        )

        save_plot(fig, output)

        print("Saved:", output.name)

        print(f"\n{method.upper()} CORRELATION")

        print(correlation.round(4).to_string())


def plot_speed_fuel_trend(df):

    data = df[[
        "speed_kmh",
        TARGET,
        "scenario"
    ]].dropna().copy()

    # Group speed observations into 5 km/h bins
    data["speed_bin"] = (
        np.floor(data["speed_kmh"] / 5) * 5
    )

    summary = (
        data.groupby(
            ["scenario", "speed_bin"]
        )[TARGET]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    summary["mean_fuel_lph"] = (
        summary["mean"] * 3600
    )

    summary.to_csv(
        REPORT_DIR / "speed_fuel_summary.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for scenario, color in [
        ("baseline", COLORS["baseline"]),
        ("eco", COLORS["eco"])
    ]:

        group = summary[
            (summary["scenario"] == scenario)
            & (summary["count"] >= 30)
        ].sort_values("speed_bin")

        ax.plot(
            group["speed_bin"] + 2.5,
            group["mean_fuel_lph"],
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=color,
            label=scenario.title()
        )

    ax.set_title("Average Fuel Rate by Speed Range")

    ax.set_xlabel("Speed bin midpoint (km/h)")

    ax.set_ylabel("Mean fuel rate (L/hour)")

    ax.legend()

    save_plot(
        fig,
        FIGURE_DIR / "speed_fuel_trend.png"
    )

    print("Saved: speed_fuel_trend.png")


def plot_acceleration_fuel_trend(df):

    data = df[[
        "acceleration_ms2",
        TARGET,
        "scenario"
    ]].dropna().copy()

    # Acceleration bins of 0.25 m/s²
    data["acceleration_bin"] = (
        np.floor(
            data["acceleration_ms2"] / 0.25
        ) * 0.25
    )

    summary = (
        data.groupby(
            ["scenario", "acceleration_bin"]
        )[TARGET]
        .agg(["mean", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for scenario, color in [
        ("baseline", COLORS["baseline"]),
        ("eco", COLORS["eco"])
    ]:

        group = summary[
            (summary["scenario"] == scenario)
            & (summary["count"] >= 30)
        ].sort_values("acceleration_bin")

        ax.plot(
            group["acceleration_bin"] + 0.125,
            group["mean"] * 3600,
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=scenario.title()
        )

    ax.axvline(
        0,
        color=COLORS["muted"],
        linestyle="--"
    )

    ax.set_title("Acceleration and Fuel Consumption")

    ax.set_xlabel("Acceleration (m/s²)")

    ax.set_ylabel("Mean fuel rate (L/hour)")

    ax.legend()

    save_plot(
        fig,
        FIGURE_DIR / "acceleration_fuel_trend.png"
    )

    print("Saved: acceleration_fuel_trend.png")


def main():

    print("=" * 65)
    print("ECOROUTE AI - BIVARIATE ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(DATA_FILE)

    print("Observations:", len(df))

    print(
        "Trajectories:",
        df["trajectory_id"].nunique()
    )

    print("\nGENERATING RELATIONSHIP PLOTS\n")

    for column, label in FEATURES.items():

        plot_relationship(df, column, label)

    plot_correlation_matrices(df)

    plot_speed_fuel_trend(df)

    plot_acceleration_fuel_trend(df)

    print("\nFEATURE CORRELATION WITH FUEL RATE")

    for column in FEATURES:

        pearson = df[column].corr(
            df[TARGET],
            method="pearson"
        )

        spearman = df[column].corr(
            df[TARGET],
            method="spearman"
        )

        print(
            f"{column:30s} "
            f"Pearson: {pearson:8.4f} "
            f"Spearman: {spearman:8.4f}"
        )

    print("\nFigures saved to:", FIGURE_DIR)

    print("Reports saved to:", REPORT_DIR)

    print("\nBIVARIATE ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
