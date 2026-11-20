
"""
EcoRoute AI
Univariate Exploratory Data Analysis
Version 1.0

Purpose:
    Investigate individual telemetry features
    and generate scientific visualizations.
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


FEATURES = {

    "speed_kmh": {
        "title": "Vehicle Speed Distribution",
        "label": "Speed (km/h)",
        "color": COLORS["baseline"]
    },

    "fuel_rate_lps": {
        "title": "Fuel Consumption Rate",
        "label": "Fuel rate (L/s)",
        "color": COLORS["fuel"]
    },

    "acceleration_ms2": {
        "title": "Vehicle Acceleration",
        "label": "Acceleration (m/s²)",
        "color": COLORS["eco"]
    },

    "jerk_ms3": {
        "title": "Vehicle Jerk Distribution",
        "label": "Jerk (m/s³)",
        "color": COLORS["purple"]
    },

    "distance_to_intersection_m": {
        "title": "Distance to Intersection",
        "label": "Distance (m)",
        "color": COLORS["baseline"]
    }
}


def plot_distribution(df, column, settings):

    values = df[column].dropna()

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.histplot(
        values,
        bins=60,
        color=settings["color"],
        edgecolor=COLORS["surface"],
        alpha=0.85,
        ax=ax
    )

    ax.axvline(
        values.mean(),
        color=COLORS["warning"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {values.mean():.4f}"
    )

    ax.axvline(
        values.median(),
        color=COLORS["text"],
        linestyle=":",
        linewidth=2,
        label=f"Median: {values.median():.4f}"
    )

    ax.set_title(settings["title"])

    ax.set_xlabel(settings["label"])

    ax.set_ylabel("Number of observations")

    ax.legend()

    output = FIGURE_DIR / f"{column}_distribution.png"

    save_plot(fig, output)

    print("Saved:", output.name)


def plot_trajectory_duration(df):

    duration = df.groupby(
        "trajectory_id"
    )["elapsed_time_s"].max()

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.histplot(
        duration,
        bins=40,
        color=COLORS["eco"],
        edgecolor=COLORS["surface"],
        ax=ax
    )

    ax.axvline(
        duration.median(),
        color=COLORS["text"],
        linestyle="--",
        label=f"Median: {duration.median():.1f} s"
    )

    ax.set_title("Recorded Trajectory Duration")

    ax.set_xlabel("Duration (seconds)")

    ax.set_ylabel("Number of trajectories")

    ax.legend()

    save_plot(
        fig,
        FIGURE_DIR / "trajectory_duration.png"
    )

    print("Saved: trajectory_duration.png")


def plot_driving_states(df):

    counts = df["driving_state"].value_counts()

    percentages = counts / len(df) * 100

    palette = {
        "stationary": COLORS["fuel"],
        "accelerating": COLORS["eco"],
        "decelerating": COLORS["warning"],
        "cruising": COLORS["baseline"],
        "unknown": COLORS["muted"]
    }

    colors = [
        palette.get(state, COLORS["muted"])
        for state in counts.index
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(
        counts.index,
        percentages,
        color=colors
    )

    ax.invert_yaxis()

    for bar, value in zip(bars, percentages):

        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center"
        )

    ax.set_xlim(0, max(percentages) + 10)

    ax.set_title("Vehicle Driving State Distribution")

    ax.set_xlabel("Percentage of observations")

    save_plot(
        fig,
        FIGURE_DIR / "driving_states.png"
    )

    print("Saved: driving_states.png")


def generate_statistics(df):

    columns = list(FEATURES.keys())

    stats = df[columns].describe().T

    stats["median"] = df[columns].median()

    stats["skewness"] = df[columns].skew()

    stats["kurtosis"] = df[columns].kurtosis()

    stats["missing_values"] = (
        df[columns].isna().sum()
    )

    stats["zero_values"] = (
        df[columns].eq(0).sum()
    )

    stats.to_csv(
        REPORT_DIR / "univariate_statistics.csv"
    )

    print("\nDESCRIPTIVE STATISTICS")

    print(stats.round(5).to_string())

    return stats


def main():

    print("=" * 65)
    print("ECOROUTE AI - UNIVARIATE ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(DATA_FILE)

    print("Observations:", len(df))

    print(
        "Trajectories:",
        df["trajectory_id"].nunique()
    )

    print("\nGENERATING VISUALIZATIONS\n")

    for column, settings in FEATURES.items():

        plot_distribution(
            df,
            column,
            settings
        )

    plot_trajectory_duration(df)

    plot_driving_states(df)

    generate_statistics(df)

    print("\nDRIVING STATES")

    print(df["driving_state"].value_counts())

    print("\nFIGURES SAVED TO:")

    print(FIGURE_DIR)

    print("\nREPORTS SAVED TO:")

    print(REPORT_DIR)

    print("\nUNIVARIATE ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
