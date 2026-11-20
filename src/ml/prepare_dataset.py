
"""
EcoRoute AI
Machine Learning Dataset Preparation
Version 1.0

Purpose:
    Prepare a reproducible ML dataset,
    prevent target leakage, and create
    trajectory-level stratified partitions.
"""

from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    ROOT / "data" / "processed"
    / "truck_telemetry_features.csv"
)

ML_DATA_DIR = ROOT / "data" / "ml"
REPORT_DIR = ROOT / "reports" / "ml"
FIGURE_DIR = ROOT / "assets" / "figures" / "ml"

for directory in [
    ML_DATA_DIR,
    REPORT_DIR,
    FIGURE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42

FEATURES = [
    "speed_kmh",
    "acceleration_ms2",
    "jerk_ms3",
    "distance_to_intersection_m"
]

TARGET = "fuel_rate_lps"

METADATA = [
    "trajectory_id",
    "scenario",
    "sample_index"
]

OPTIONAL_METADATA = [
    "elapsed_time_s",
    "interval_fuel_l",
    "cumulative_fuel_l",
    "cumulative_distance_m"
]

REQUIRED_COLUMNS = METADATA + FEATURES + [TARGET]

COLORS = {
    "train": "#38BDF8",
    "validation": "#A78BFA",
    "test": "#F59E0B",
    "baseline": "#38BDF8",
    "eco": "#10B981"
}

plt.style.use("dark_background")


# ==========================================
# DATA LOADING AND AUDITING
# ==========================================

def load_dataset():

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(SOURCE_FILE)

    df = pd.read_csv(SOURCE_FILE)

    print("\nORIGINAL DATASET")
    print("Observations:", len(df))
    print("Columns:", len(df.columns))

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    columns = REQUIRED_COLUMNS + [
        col for col in OPTIONAL_METADATA
        if col in df.columns
    ]

    df = df[columns].copy()

    audit = []

    for column in FEATURES + [TARGET]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        infinite = np.isinf(df[column]).sum()

        df[column] = df[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

        audit.append({
            "column": column,
            "missing_values": int(
                df[column].isna().sum()
            ),
            "infinite_values": int(infinite),
            "minimum": df[column].min(),
            "maximum": df[column].max(),
            "mean": df[column].mean(),
            "standard_deviation": df[column].std()
        })

    pd.DataFrame(audit).to_csv(
        REPORT_DIR / "ml_feature_audit.csv",
        index=False
    )

    return df


# ==========================================
# DATA CLEANING
# ==========================================

def clean_dataset(df):

    original_rows = len(df)

    df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    df["scenario"] = (
        df["scenario"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["scenario"].isin(
            ["baseline", "eco"]
        )
    ].copy()

    df = df[
        (df[TARGET] >= 0)
        & (df["speed_kmh"] >= 0)
    ].copy()

    duplicated_keys = df.duplicated(
        subset=[
            "trajectory_id",
            "sample_index"
        ],
        keep=False
    )

    if duplicated_keys.any():

        raise ValueError(
            "Duplicate trajectory/sample keys "
            "require investigation."
        )

    scenario_counts = df.groupby(
        "trajectory_id"
    )["scenario"].nunique()

    if (scenario_counts > 1).any():

        raise ValueError(
            "A trajectory contains multiple scenarios."
        )

    df = df.sort_values(
        ["trajectory_id", "sample_index"]
    ).reset_index(drop=True)

    removed_rows = original_rows - len(df)

    print("\nCLEANING RESULTS")
    print("Original rows:", original_rows)
    print("Removed rows:", removed_rows)
    print("Remaining rows:", len(df))
    print(
        "Trajectories:",
        df["trajectory_id"].nunique()
    )

    if df.empty:
        raise ValueError(
            "No valid observations remain."
        )

    return df


# ==========================================
# FEATURE SAFETY CHECK
# ==========================================

def validate_features():

    prohibited_terms = [
        "fuel",
        "co2",
        "emission",
        "trajectory",
        "sample_index",
        "scenario"
    ]

    for feature in FEATURES:

        for term in prohibited_terms:

            if term in feature.lower():

                raise ValueError(
                    f"Potential leakage: {feature}"
                )

    if TARGET in FEATURES:

        raise ValueError(
            "Target included in model inputs."
        )

    print("\nFEATURE CONFIGURATION")

    for index, feature in enumerate(
        FEATURES, start=1
    ):
        print(f"{index}. {feature}")

    print("Prediction target:", TARGET)
    print("Feature allowlist: PASSED")


# ==========================================
# TRAJECTORY-LEVEL SPLITTING
# ==========================================

def split_dataset(df):

    groups = (
        df.groupby("trajectory_id")
        .agg(
            scenario=("scenario", "first"),
            observations=("sample_index", "size")
        )
        .reset_index()
    )

    # Split complete trajectories:
    # 70% training, 30% temporary.

    train_groups, temporary_groups = (
        train_test_split(
            groups,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=groups["scenario"]
        )
    )

    # Split the remaining 30%:
    # 15% validation, 15% testing.

    validation_groups, test_groups = (
        train_test_split(
            temporary_groups,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=temporary_groups["scenario"]
        )
    )

    group_partitions = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups
    }

    datasets = {}
    assignments = []

    for name, group in group_partitions.items():

        ids = group["trajectory_id"]

        partition = df[
            df["trajectory_id"].isin(ids)
        ].copy()

        datasets[name] = partition

        assignment = group.copy()
        assignment["partition"] = name

        assignments.append(assignment)

    pd.concat(
        assignments,
        ignore_index=True
    ).to_csv(
        REPORT_DIR / "trajectory_split_manifest.csv",
        index=False
    )

    return datasets


# ==========================================
# DATA LEAKAGE VALIDATION
# ==========================================

def validate_partitions(df, datasets):

    train_ids = set(
        datasets["train"]["trajectory_id"]
    )

    validation_ids = set(
        datasets["validation"]["trajectory_id"]
    )

    test_ids = set(
        datasets["test"]["trajectory_id"]
    )

    checks = {
        "train_validation_overlap":
            len(train_ids & validation_ids) == 0,

        "train_test_overlap":
            len(train_ids & test_ids) == 0,

        "validation_test_overlap":
            len(validation_ids & test_ids) == 0,

        "all_trajectories_preserved":
            (
                train_ids
                | validation_ids
                | test_ids
            ) == set(df["trajectory_id"]),

        "all_rows_preserved":
            sum(
                len(data)
                for data in datasets.values()
            ) == len(df),

        "target_not_in_features":
            TARGET not in FEATURES,

        "finite_model_values":
            all(
                np.isfinite(
                    data[FEATURES + [TARGET]]
                    .to_numpy()
                ).all()
                for data in datasets.values()
            ),

        "all_partitions_have_both_scenarios":
            all(
                set(data["scenario"])
                == {"baseline", "eco"}
                for data in datasets.values()
            )
    }

    print("\nPARTITION VALIDATION")

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {name}")

    with open(
        REPORT_DIR / "split_validation.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(checks, file, indent=4)

    if not all(checks.values()):

        raise RuntimeError(
            "Dataset partition validation failed."
        )

    print(
        "\nValidation passed:",
        sum(checks.values()),
        "/",
        len(checks)
    )


# ==========================================
# SAVE MACHINE LEARNING DATASETS
# ==========================================

def save_datasets(datasets):

    records = []

    total_rows = sum(
        len(data)
        for data in datasets.values()
    )

    for name, data in datasets.items():

        # Keep full metadata for future
        # trajectory-level evaluation.

        output_file = (
            ML_DATA_DIR / f"{name}.csv"
        )

        data.to_csv(
            output_file,
            index=False
        )

        trajectory_counts = (
            data.groupby("scenario")[
                "trajectory_id"
            ].nunique()
        )

        records.append({
            "partition": name,
            "observations": len(data),
            "row_percentage":
                len(data) / total_rows * 100,

            "trajectories":
                data["trajectory_id"].nunique(),

            "baseline_trajectories":
                int(trajectory_counts.get(
                    "baseline", 0
                )),

            "eco_trajectories":
                int(trajectory_counts.get(
                    "eco", 0
                ))
        })

        print("\n[SAVED]", output_file.name)
        print("Observations:", len(data))

        print(
            "Trajectories:",
            data["trajectory_id"].nunique()
        )

    summary = pd.DataFrame(records)

    summary.to_csv(
        REPORT_DIR / "dataset_split_summary.csv",
        index=False
    )

    print("\nFINAL DATASET DISTRIBUTION")

    print(
        summary.round(3).to_string(index=False)
    )

    return summary


# ==========================================
# CONFIGURATION MANIFEST
# ==========================================

def save_configuration():

    source_hash = hashlib.sha256(
        SOURCE_FILE.read_bytes()
    ).hexdigest()

    config = {
        "project": "EcoRoute AI",

        "task":
            "Supervised regression",

        "features": FEATURES,

        "target": TARGET,

        "target_unit": "L/s",

        "group_column": "trajectory_id",

        "stratification_column": "scenario",

        "random_seed": RANDOM_SEED,

        "source_dataset": str(
            SOURCE_FILE.relative_to(ROOT)
        ),

        "source_sha256": source_hash,

        "split_ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        }
    }

    with open(
        ML_DATA_DIR / "ml_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(config, file, indent=4)

    print("\n[SAVED] ml_config.json")


# ==========================================
# VISUALIZATION 01
# ==========================================

def plot_dataset_split(summary):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    fig.patch.set_facecolor("#080F19")
    ax.set_facecolor("#111D2C")

    bars = ax.bar(
        summary["partition"].str.title(),
        summary["observations"],
        color=[
            COLORS[name]
            for name in summary["partition"]
        ],
        width=0.55
    )

    ax.bar_label(
        bars,
        labels=[
            f"{value:,}"
            for value in summary["observations"]
        ],
        padding=6
    )

    ax.set_title(
        "Machine Learning Dataset Distribution",
        fontsize=17,
        pad=20
    )

    ax.set_ylabel("Telemetry observations")

    ax.set_ylim(
        0,
        summary["observations"].max() * 1.2
    )

    ax.grid(
        axis="y",
        alpha=0.15
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "ml_dataset_split.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# VISUALIZATION 02
# ==========================================

def plot_scenario_balance(summary):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    fig.patch.set_facecolor("#080F19")
    ax.set_facecolor("#111D2C")

    baseline_percent = (
        summary["baseline_trajectories"]
        / summary["trajectories"] * 100
    )

    eco_percent = (
        summary["eco_trajectories"]
        / summary["trajectories"] * 100
    )

    labels = summary["partition"].str.title()

    ax.bar(
        labels,
        baseline_percent,
        color=COLORS["baseline"],
        label="Baseline",
        width=0.5
    )

    ax.bar(
        labels,
        eco_percent,
        bottom=baseline_percent,
        color=COLORS["eco"],
        label="Eco-driving",
        width=0.5
    )

    ax.set_title(
        "Scenario Balance Across Partitions",
        fontsize=17,
        pad=20
    )

    ax.set_ylabel("Trajectory percentage")

    ax.set_ylim(0, 110)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "ml_scenario_balance.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# VISUALIZATION 03
# ==========================================

def plot_training_features(train):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10)
    )

    fig.patch.set_facecolor("#080F19")

    for ax, feature in zip(
        axes.flatten(), FEATURES
    ):

        ax.set_facecolor("#111D2C")

        ax.hist(
            train[feature],
            bins=50,
            color="#34D399",
            alpha=0.85
        )

        ax.set_title(
            feature.replace("_", " ").title()
        )

        ax.set_ylabel("Frequency")

        ax.grid(alpha=0.1)

    fig.suptitle(
        "Training Feature Distributions",
        fontsize=20
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "ml_training_feature_distributions.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    print("=" * 65)
    print("ECOROUTE AI - ML DATASET PREPARATION")
    print("=" * 65)

    df = load_dataset()

    df = clean_dataset(df)

    validate_features()

    datasets = split_dataset(df)

    validate_partitions(df, datasets)

    summary = save_datasets(datasets)

    save_configuration()

    plot_dataset_split(summary)

    plot_scenario_balance(summary)

    plot_training_features(
        datasets["train"]
    )

    print("\n" + "=" * 65)

    print("ML DATASET PREPARATION COMPLETED!")

    print("Input features:", len(FEATURES))
    print("Prediction target:", TARGET)
    print("Dataset partitions:", len(datasets))

    print("Trajectory leakage: VERIFIED ABSENT")

    print("=" * 65)


if __name__ == "__main__":
    main()
