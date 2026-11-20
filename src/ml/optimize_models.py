
"""
EcoRoute AI
Model Optimization and Final Evaluation
Version 1.0
"""

from pathlib import Path
import json
import logging
import time
import os
import hashlib
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn
import xgboost

from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer
)

from xgboost import XGBRegressor

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "ml"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "ml"
FIGURE_DIR = ROOT / "assets" / "figures" / "ml"

for directory in [
    MODEL_DIR,
    REPORT_DIR,
    FIGURE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


SEED = 42
N_JOBS = min(4, os.cpu_count() or 1)

CV_FOLDS = 3
SEARCH_ITERATIONS = 8

CO2_FACTOR = 2.68
SAMPLING_INTERVAL = 1.0

BACKGROUND = "#080F19"
SURFACE = "#111D2C"
GREEN = "#10B981"
BLUE = "#38BDF8"
PURPLE = "#A78BFA"
AMBER = "#F59E0B"

plt.style.use("dark_background")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            REPORT_DIR / "optimization.log",
            mode="w",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("EcoRoute AI")


# ==========================================
# DATA LOADING
# ==========================================

def load_config():

    config = json.loads(
        (DATA_DIR / "ml_config.json").read_text(
            encoding="utf-8"
        )
    )

    allowed = {
        "speed_kmh",
        "acceleration_ms2",
        "jerk_ms3",
        "distance_to_intersection_m"
    }

    if (
        set(config["features"]) != allowed
        or len(config["features"]) != 4
        or config["target"] != "fuel_rate_lps"
    ):
        raise ValueError(
            "Invalid ML feature configuration."
        )

    return config


def load_partition(name, config):

    data = pd.read_csv(
        DATA_DIR / f"{name}.csv"
    )

    required = config["features"] + [
        config["target"],
        "trajectory_id",
        "scenario",
        "sample_index"
    ]

    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}"
        )

    if data.empty:
        raise ValueError(
            f"{name} dataset is empty."
        )

    if data[required].isna().any().any():
        raise ValueError(
            f"{name}: missing required values."
        )

    if not np.isfinite(
        data[
            config["features"] + [config["target"]]
        ].to_numpy()
    ).all():
        raise ValueError(
            f"{name}: non-finite model values."
        )

    if (data[config["target"]] < 0).any():
        raise ValueError(
            f"{name}: negative target values."
        )

    if data.duplicated(
        ["trajectory_id", "sample_index"]
    ).any():
        raise ValueError(
            f"{name}: duplicate telemetry keys."
        )

    return data


# ==========================================
# PERFORMANCE METRICS
# ==========================================

def predict(model, X):

    values = np.asarray(
        model.predict(X),
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Model produced invalid predictions."
        )

    return np.maximum(values, 0)


def rmse(y_true, y_pred):

    y_pred = np.maximum(y_pred, 0)

    return float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


SCORER = make_scorer(
    rmse,
    greater_is_better=False
)


def evaluate(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_pred - y_true

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),

        "rmse": rmse(y_true, y_pred),

        "r2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) > 1
            else None
        ),

        "bias": float(
            np.mean(residuals)
        ),

        "median_absolute_error": float(
            np.median(np.abs(residuals))
        ),

        "observations": int(len(y_true))
    }


# ==========================================
# OPTIMIZATION CONFIGURATION
# ==========================================

def build_search_spaces():

    spaces = {

        "xgboost": (

            XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=SEED,
                n_jobs=N_JOBS,
                verbosity=0
            ),

            {
                "n_estimators": [200, 350, 500, 700],
                "max_depth": [4, 5, 6, 8],
                "learning_rate": [0.025, 0.05, 0.1],
                "subsample": [0.75, 0.85, 1.0],
                "colsample_bytree": [0.75, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 10],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [1, 3, 5]
            }
        )
    }

    if RandomForestRegressor is not None:

        spaces["random_forest"] = (

            RandomForestRegressor(
                random_state=SEED,
                n_jobs=N_JOBS
            ),

            {
                "n_estimators": [100, 150, 200],
                "max_depth": [10, 14, 18, 22],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [2, 4, 8],
                "max_features": [0.75, 1.0],
                "max_samples": [0.7, 0.9, None]
            }
        )

    else:
        logger.warning(
            "Random Forest unavailable."
        )

    return spaces


# ==========================================
# GROUPED HYPERPARAMETER SEARCH
# ==========================================

def optimize(
    train,
    validation,
    config
):

    features = config["features"]
    target = config["target"]

    X_train = train[features]
    y_train = train[target]

    X_val = validation[features]
    y_val = validation[target]

    groups = train["trajectory_id"]

    cv = GroupKFold(
        n_splits=CV_FOLDS
    )

    spaces = build_search_spaces()

    candidates = {}
    records = []

    # Benchmark original model configurations
    # against optimized configurations.

    for name, (model, parameters) in spaces.items():

        original_path = (
            MODEL_DIR / f"{name}.joblib"
        )

        if original_path.exists():

            original = joblib.load(original_path)

            original.fit(X_train, y_train)

            original_pred = predict(
                original,
                X_val
            )

            candidates[f"{name}_original"] = original

            records.append({
                "candidate": f"{name}_original",
                "algorithm": name,
                "cv_rmse": np.nan,
                "validation_mae":
                    mean_absolute_error(
                        y_val, original_pred
                    ),
                "validation_rmse":
                    rmse(y_val, original_pred),
                "validation_r2":
                    r2_score(y_val, original_pred),
                "parameters":
                    json.dumps(
                        original.get_params(),
                        default=str
                    )
            })

        logger.info("=" * 60)
        logger.info("Optimizing %s", name)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            n_iter=SEARCH_ITERATIONS,
            scoring=SCORER,
            cv=cv,
            random_state=SEED,
            n_jobs=1,
            verbose=2,
            refit=True,
            error_score="raise",
            return_train_score=False
        )

        start = time.perf_counter()

        search.fit(
            X_train,
            y_train,
            groups=groups
        )

        elapsed = (
            time.perf_counter() - start
        )

        best = search.best_estimator_

        validation_pred = predict(
            best,
            X_val
        )

        metrics = evaluate(
            y_val,
            validation_pred
        )

        candidate_name = f"{name}_optimized"

        candidates[candidate_name] = best

        pd.DataFrame(
            search.cv_results_
        ).sort_values(
            "rank_test_score"
        ).to_csv(
            REPORT_DIR / f"{name}_search_results.csv",
            index=False
        )

        joblib.dump(
            best,
            MODEL_DIR / f"{candidate_name}.joblib",
            compress=3
        )

        records.append({
            "candidate": candidate_name,
            "algorithm": name,

            "cv_rmse":
                -float(search.best_score_),

            "validation_mae":
                metrics["mae"],

            "validation_rmse":
                metrics["rmse"],

            "validation_r2":
                metrics["r2"],

            "parameters":
                json.dumps(
                    search.best_params_,
                    default=str
                )
        })

        logger.info(
            "Best CV RMSE: %.8f",
            -search.best_score_
        )

        logger.info(
            "Validation RMSE: %.8f",
            metrics["rmse"]
        )

        logger.info(
            "Validation R2: %.6f",
            metrics["r2"]
        )

        logger.info(
            "Optimization time: %.2f seconds",
            elapsed
        )

    comparison = pd.DataFrame(
        records
    ).sort_values(
        "validation_rmse"
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "rank",
        np.arange(1, len(comparison) + 1)
    )

    comparison.to_csv(
        REPORT_DIR / "optimization_comparison.csv",
        index=False
    )

    winner = comparison.iloc[0]

    best_name = winner["candidate"]

    return candidates[best_name], comparison


# ==========================================
# FINAL MODEL TRAINING
# ==========================================

def train_final_model(
    selected,
    train,
    validation,
    config
):

    combined = pd.concat(
        [train, validation],
        ignore_index=True
    )

    final_model = clone(selected)

    logger.info(
        "Retraining selected configuration "
        "on training + validation data."
    )

    final_model.fit(
        combined[config["features"]],
        combined[config["target"]]
    )

    return final_model, combined


# ==========================================
# HELD-OUT TEST EVALUATION
# ==========================================

def evaluate_test(
    model,
    test,
    config
):

    features = config["features"]
    target = config["target"]

    result = test.copy()

    result["actual_fuel_lps"] = result[target]

    result["predicted_fuel_lps"] = predict(
        model,
        result[features]
    )

    result["residual_lps"] = (
        result["actual_fuel_lps"]
        - result["predicted_fuel_lps"]
    )

    result["absolute_error_lps"] = (
        result["residual_lps"].abs()
    )

    metrics = evaluate(
        result["actual_fuel_lps"],
        result["predicted_fuel_lps"]
    )

    scenario_records = []

    for scenario, group in result.groupby(
        "scenario"
    ):

        scenario_records.append({
            "scenario": scenario,
            **evaluate(
                group["actual_fuel_lps"],
                group["predicted_fuel_lps"]
            )
        })

    scenario_metrics = pd.DataFrame(
        scenario_records
    )

    scenario_metrics.to_csv(
        REPORT_DIR / "final_scenario_metrics.csv",
        index=False
    )

    result.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False
    )

    return result, metrics, scenario_metrics


# ==========================================
# TRAJECTORY FUEL INTEGRATION
# ==========================================

def evaluate_trajectories(predictions):

    records = []

    for trajectory_id, group in predictions.groupby(
        "trajectory_id"
    ):

        group = group.sort_values(
            "sample_index"
        )

        if len(group) < 2:
            continue

        actual = group[
            "actual_fuel_lps"
        ].to_numpy()

        predicted = group[
            "predicted_fuel_lps"
        ].to_numpy()

        sample_indices = group[
            "sample_index"
        ].to_numpy()

        # Integrate only contiguous observations.
        # Never bridge missing telemetry intervals.

        steps = np.diff(sample_indices)

        valid = steps == 1

        if not valid.any():
            continue

        actual_intervals = (
            (actual[:-1] + actual[1:])
            / 2 * SAMPLING_INTERVAL
        )

        predicted_intervals = (
            (predicted[:-1] + predicted[1:])
            / 2 * SAMPLING_INTERVAL
        )

        actual_fuel = float(
            actual_intervals[valid].sum()
        )

        predicted_fuel = float(
            predicted_intervals[valid].sum()
        )

        error = predicted_fuel - actual_fuel

        error_percent = (
            error / actual_fuel * 100
            if actual_fuel > 0
            else np.nan
        )

        records.append({

            "trajectory_id": trajectory_id,

            "scenario":
                group["scenario"].iloc[0],

            "observations": len(group),

            "integrated_duration_s":
                int(valid.sum()) * SAMPLING_INTERVAL,

            "actual_fuel_l": actual_fuel,

            "predicted_fuel_l": predicted_fuel,

            "fuel_error_l": error,

            "absolute_error_l": abs(error),

            "fuel_error_percent": error_percent,

            "absolute_error_percent":
                abs(error_percent),

            "actual_co2_kg":
                actual_fuel * CO2_FACTOR,

            "predicted_co2_kg":
                predicted_fuel * CO2_FACTOR
        })

    trajectories = pd.DataFrame(records)

    if trajectories.empty:
        raise ValueError(
            "No integrable test trajectories."
        )

    trajectories.to_csv(
        REPORT_DIR / "trajectory_prediction_metrics.csv",
        index=False
    )

    total_actual = float(
        trajectories["actual_fuel_l"].sum()
    )

    total_predicted = float(
        trajectories["predicted_fuel_l"].sum()
    )

    total_bias = (
        (total_predicted - total_actual)
        / total_actual * 100
        if total_actual > 0
        else None
    )

    wape = (
        float(
            trajectories["absolute_error_l"].sum()
            / total_actual * 100
        )
        if total_actual > 0
        else None
    )

    valid_percentages = trajectories[
        "absolute_error_percent"
    ].dropna()

    summary = {
        "trajectories": len(trajectories),

        "actual_total_fuel_l":
            total_actual,

        "predicted_total_fuel_l":
            total_predicted,

        "total_fuel_bias_percent":
            total_bias,

        "trajectory_mae_l":
            float(
                trajectories["absolute_error_l"]
                .mean()
            ),

        "trajectory_mape_percent":
            float(valid_percentages.mean())
            if len(valid_percentages)
            else None,

        "trajectory_wape_percent":
            wape
    }

    return trajectories, summary


# ==========================================
# MODEL DOMAIN
# ==========================================

def build_feature_domain(data, features):

    result = {}

    for feature in features:

        values = data[feature]

        result[feature] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(values.quantile(0.01)),
            "p99": float(values.quantile(0.99)),
            "median": float(values.median())
        }

    return result


# ==========================================
# SAVE PRODUCTION CANDIDATE
# ==========================================

def save_final(
    model,
    comparison,
    config,
    combined,
    test_metrics,
    trajectory_metrics
):

    winner = comparison.iloc[0]

    domain = build_feature_domain(
        combined,
        config["features"]
    )

    model_name = str(winner["algorithm"])

    bundle = {
        "model": model,
        "model_name": model_name,
        "candidate_name": str(winner["candidate"]),
        "features": config["features"],
        "target": config["target"],
        "target_unit": "litres_per_second",
        "clip_minimum": 0.0,
        "feature_domain": domain,
        "sampling_interval_seconds":
            SAMPLING_INTERVAL,
        "diesel_co2_kg_per_litre":
            CO2_FACTOR
    }

    model_path = MODEL_DIR / "final_model.joblib"

    joblib.dump(
        bundle,
        model_path,
        compress=3
    )

    # Portable native format when XGBoost wins.

    if isinstance(model, XGBRegressor):

        model.save_model(
            MODEL_DIR / "final_xgboost.ubj"
        )

    metadata = {
        "project": "EcoRoute AI",
        "version": "1.0",
        "status": "held_out_test_evaluated",
        "model_name": model_name,
        "candidate_name": str(winner["candidate"]),
        "training_observations": len(combined),

        "training_trajectories":
            combined["trajectory_id"].nunique(),

        "features": config["features"],
        "target": config["target"],

        "validation_rmse_before_refit":
            float(winner["validation_rmse"]),

        "test_metrics":
            test_metrics,

        "trajectory_metrics":
            trajectory_metrics,

        "feature_domain":
            domain,

        "model_sha256":
            hashlib.sha256(
                model_path.read_bytes()
            ).hexdigest(),

        "random_seed":
            SEED,

        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__
        }
    }

    (MODEL_DIR / "final_model_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=4,
            default=str,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    (REPORT_DIR / "final_test_metrics.json").write_text(
        json.dumps(
            test_metrics,
            indent=4,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    (REPORT_DIR / "trajectory_metrics_summary.json").write_text(
        json.dumps(
            trajectory_metrics,
            indent=4,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    logger.info(
        "Final model saved: %s",
        model_path
    )


# ==========================================
# VISUALIZATIONS
# ==========================================

def save_plot(fig, filename):

    fig.patch.set_facecolor(BACKGROUND)
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=220,
        bbox_inches="tight",
        facecolor=BACKGROUND
    )

    plt.close(fig)


def plot_optimization(comparison):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.set_facecolor(SURFACE)

    bars = ax.barh(
        comparison["candidate"]
        .str.replace("_", " ")
        .str.title(),

        comparison["validation_rmse"],

        color=[
            GREEN if i == 0 else BLUE
            for i in range(len(comparison))
        ]
    )

    ax.bar_label(
        bars,
        fmt="%.7f",
        padding=5
    )

    ax.invert_yaxis()
    ax.margins(x=0.25)

    ax.set_title(
        "Optimized Model Comparison",
        fontsize=18
    )

    ax.set_xlabel("Validation RMSE (L/s)")
    ax.grid(axis="x", alpha=0.12)

    save_plot(
        fig,
        "optimized_model_comparison.png"
    )


def plot_test_predictions(data):

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.set_facecolor(SURFACE)

    density = ax.hexbin(
        data["actual_fuel_lps"],
        data["predicted_fuel_lps"],
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="viridis"
    )

    limit = max(
        data["actual_fuel_lps"].max(),
        data["predicted_fuel_lps"].max()
    ) * 1.05

    ax.plot(
        [0, limit],
        [0, limit],
        "--",
        color=AMBER,
        label="Perfect prediction"
    )

    ax.set_title(
        "Final Model | Held-out Test Performance"
    )

    ax.set_xlabel("Actual fuel rate (L/s)")
    ax.set_ylabel("Predicted fuel rate (L/s)")

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)

    ax.legend()

    fig.colorbar(
        density,
        ax=ax,
        label="Observation density"
    )

    save_plot(
        fig,
        "final_test_actual_vs_predicted.png"
    )


def plot_trajectory_performance(trajectories):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 7)
    )

    for ax in axes:
        ax.set_facecolor(SURFACE)

    for scenario, group in trajectories.groupby(
        "scenario"
    ):

        color = (
            GREEN if scenario == "eco"
            else BLUE
        )

        axes[0].scatter(
            group["actual_fuel_l"],
            group["predicted_fuel_l"],
            color=color,
            alpha=0.75,
            label=scenario.title()
        )

    limit = max(
        trajectories["actual_fuel_l"].max(),
        trajectories["predicted_fuel_l"].max()
    ) * 1.05

    axes[0].plot(
        [0, limit],
        [0, limit],
        "--",
        color=AMBER
    )

    axes[0].set_xlabel("Actual integrated fuel (L)")
    axes[0].set_ylabel("Predicted integrated fuel (L)")
    axes[0].set_title("Trajectory Fuel Accuracy")
    axes[0].legend()

    errors = trajectories[
        "fuel_error_percent"
    ].dropna()

    axes[1].hist(
        errors,
        bins=25,
        color=GREEN,
        alpha=0.85
    )

    axes[1].axvline(
        0,
        color=AMBER,
        linestyle="--"
    )

    axes[1].set_title("Trajectory Error Distribution")
    axes[1].set_xlabel("Relative fuel error (%)")
    axes[1].set_ylabel("Trajectories")

    fig.suptitle(
        "EcoRoute AI | Journey-Level Evaluation",
        fontsize=19
    )

    save_plot(
        fig,
        "trajectory_fuel_prediction_accuracy.png"
    )


def plot_telemetry_examples(data):

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 10)
    )

    for ax, scenario in zip(
        axes, ["baseline", "eco"]
    ):

        subset = data[
            data["scenario"] == scenario
        ]

        if subset.empty:
            ax.set_visible(False)
            continue

        sizes = subset.groupby(
            "trajectory_id"
        ).size().sort_values()

        selected_id = sizes.index[len(sizes) // 2]

        journey = subset[
            subset["trajectory_id"] == selected_id
        ].sort_values("sample_index")

        ax.set_facecolor(SURFACE)

        seconds = (
            journey["sample_index"]
            * SAMPLING_INTERVAL
        )

        ax.plot(
            seconds,
            journey["actual_fuel_lps"] * 3600,
            color=BLUE,
            label="Actual",
            linewidth=1.6
        )

        ax.plot(
            seconds,
            journey["predicted_fuel_lps"] * 3600,
            color=GREEN,
            label="Predicted",
            linewidth=1.5,
            alpha=0.85
        )

        ax.set_title(
            f"{scenario.title()} | Representative Journey"
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Fuel rate (L/h)")
        ax.legend()
        ax.grid(alpha=0.12)

    save_plot(
        fig,
        "final_model_telemetry_predictions.png"
    )


def plot_scenario_metrics(scenarios):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 6)
    )

    colors = [
        GREEN if name == "eco" else BLUE
        for name in scenarios["scenario"]
    ]

    for ax, metric in zip(
        axes, ["rmse", "r2"]
    ):

        ax.set_facecolor(SURFACE)

        bars = ax.bar(
            scenarios["scenario"].str.title(),
            scenarios[metric],
            color=colors,
            width=0.55
        )

        ax.bar_label(
            bars,
            fmt="%.5f",
            padding=5
        )

        ax.set_title(metric.upper())
        ax.margins(y=0.2)
        ax.grid(axis="y", alpha=0.12)

    fig.suptitle(
        "Final Performance by Driving Scenario",
        fontsize=18
    )

    save_plot(
        fig,
        "final_model_scenario_performance.png"
    )


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    start = time.perf_counter()

    marker = (
        REPORT_DIR / "final_test_started.json"
    )

    if marker.exists():

        raise RuntimeError(
            "Held-out evaluation was already started. "
            "Inspect previous results before proceeding."
        )

    logger.info("=" * 60)
    logger.info("ECOROUTE AI - MODEL OPTIMIZATION")
    logger.info("=" * 60)

    config = load_config()

    train = load_partition("train", config)

    validation = load_partition(
        "validation", config
    )

    train_ids = set(train["trajectory_id"])

    validation_ids = set(
        validation["trajectory_id"]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Training-validation trajectory overlap."
        )

    # Hyperparameter searches do not load test data.

    selected, comparison = optimize(
        train,
        validation,
        config
    )

    print("\nOPTIMIZATION LEADERBOARD")

    print(
        comparison[
            [
                "rank",
                "candidate",
                "cv_rmse",
                "validation_rmse",
                "validation_r2"
            ]
        ].round(7).to_string(index=False)
    )

    final_model, combined = train_final_model(
        selected,
        train,
        validation,
        config
    )

    # Model selection is finalized.

    marker.write_text(
        json.dumps({
            "selected_candidate":
                str(comparison.iloc[0]["candidate"]),

            "status":
                "selection_finalized"
        }, indent=4),
        encoding="utf-8"
    )

    logger.info(
        "Starting held-out test evaluation."
    )

    test = load_partition("test", config)

    if set(combined["trajectory_id"]) & set(
        test["trajectory_id"]
    ):
        raise ValueError(
            "Held-out trajectory overlap detected."
        )

    predictions, test_metrics, scenarios = (
        evaluate_test(
            final_model,
            test,
            config
        )
    )

    trajectories, trajectory_metrics = (
        evaluate_trajectories(predictions)
    )

    save_final(
        final_model,
        comparison,
        config,
        combined,
        test_metrics,
        trajectory_metrics
    )

    plot_optimization(comparison)
    plot_test_predictions(predictions)

    plot_trajectory_performance(
        trajectories
    )

    plot_telemetry_examples(predictions)
    plot_scenario_metrics(scenarios)

    print("\n" + "=" * 65)

    print("FINAL MODEL:",
          comparison.iloc[0]["candidate"])

    print("\nHELD-OUT TEST METRICS")

    for key, value in test_metrics.items():

        print(
            f"{key}: {value:.8f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    print("\nTRAJECTORY-LEVEL METRICS")

    for key, value in trajectory_metrics.items():

        print(
            f"{key}: {value:.6f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    print(
        "\nTotal execution time:",
        round(
            time.perf_counter() - start,
            2
        ),
        "seconds"
    )

    print("\nFINAL MODEL SAVED SUCCESSFULLY")
    print("MODEL OPTIMIZATION COMPLETED!")

    print("=" * 65)


if __name__ == "__main__":
    main()
