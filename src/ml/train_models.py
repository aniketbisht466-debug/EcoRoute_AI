
"""
EcoRoute AI
Unified Machine Learning Training Pipeline
Version 1.0
"""

from pathlib import Path
import json
import logging
import time
import os
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn
import xgboost

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
try:
    from sklearn.ensemble import RandomForestRegressor
    RANDOM_FOREST_AVAILABLE = True
except ImportError:
    RandomForestRegressor = None
    RANDOM_FOREST_AVAILABLE = False
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            REPORT_DIR / "training.log",
            mode="w",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("EcoRoute AI")

plt.style.use("dark_background")

GREEN = "#10B981"
BLUE = "#38BDF8"
PURPLE = "#A78BFA"
AMBER = "#F59E0B"
BACKGROUND = "#080F19"
SURFACE = "#111D2C"


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    config = json.loads(
        (DATA_DIR / "ml_config.json").read_text(
            encoding="utf-8"
        )
    )

    features = config["features"]
    target = config["target"]

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    allowed_features = {
        "speed_kmh",
        "acceleration_ms2",
        "jerk_ms3",
        "distance_to_intersection_m"
    }

    if (
        set(features) != allowed_features
        or len(features) != len(allowed_features)
        or target != "fuel_rate_lps"
    ):
        raise ValueError(
            "Unexpected feature or target configuration."
        )

    required = features + [
        target,
        "trajectory_id",
        "scenario",
        "sample_index"
    ]

    for name, data in [
        ("Training", train),
        ("Validation", validation)
    ]:

        missing = set(required) - set(data.columns)

        if missing:
            raise ValueError(
                f"{name}: missing columns {missing}"
            )

        if data.empty:
            raise ValueError(
                f"{name} dataset is empty."
            )

        if not np.isfinite(
            data[features + [target]].to_numpy()
        ).all():
            raise ValueError(
                f"{name}: non-finite model values."
            )

        if (data[target] < 0).any():
            raise ValueError(
                f"{name}: negative fuel measurements."
            )

    overlap = set(
        train["trajectory_id"]
    ) & set(
        validation["trajectory_id"]
    )

    if overlap:
        raise ValueError(
            "Trajectory leakage detected."
        )

    logger.info("Training rows: %s", len(train))
    logger.info("Validation rows: %s", len(validation))
    logger.info("Features: %s", features)
    logger.info("Target: %s", target)

    return train, validation, features, target


# ==========================================
# MODEL REGISTRY
# ==========================================

def build_models():

    models = {
        "dummy": DummyRegressor(
            strategy="mean"
        ),

        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "polynomial_ridge": Pipeline([
            (
                "polynomial",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False
                )
            ),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),

        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_alpha=0.01,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=SEED,
            n_jobs=N_JOBS,
            verbosity=0
        )
    }

    if RANDOM_FOREST_AVAILABLE:

        models["random_forest"] = (
            RandomForestRegressor(
                n_estimators=150,
                max_depth=16,
                min_samples_leaf=5,
                max_features=1.0,
                random_state=SEED,
                n_jobs=N_JOBS
            )
        )

    else:

        logger.warning(
            "Random Forest unavailable. "
            "Continuing with supported models."
        )

    return models

# METRICS
# ==========================================

def evaluate(y_true, y_pred):

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),

        "mse": float(
            mean_squared_error(y_true, y_pred)
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),

        "r2": float(
            r2_score(y_true, y_pred)
        )
    }


# ==========================================
# TRAIN ALL MODELS
# ==========================================

def train_all(
    train,
    validation,
    features,
    target
):

    X_train = train[features]
    y_train = train[target]

    X_val = validation[features]
    y_val = validation[target]

    models = build_models()

    results = []
    predictions = {}
    scenario_results = []

    for name, model in models.items():

        logger.info("=" * 55)
        logger.info("Training: %s", name)

        start = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = (
            time.perf_counter() - start
        )

        train_pred = np.maximum(
            model.predict(X_train),
            0
        )

        start = time.perf_counter()

        val_pred = np.maximum(
            model.predict(X_val),
            0
        )

        inference_time = (
            time.perf_counter() - start
        )

        train_metrics = evaluate(
            y_train, train_pred
        )

        val_metrics = evaluate(
            y_val, val_pred
        )

        results.append({
            "model": name,

            "train_mae":
                train_metrics["mae"],

            "train_rmse":
                train_metrics["rmse"],

            "train_r2":
                train_metrics["r2"],

            "validation_mae":
                val_metrics["mae"],

            "validation_mse":
                val_metrics["mse"],

            "validation_rmse":
                val_metrics["rmse"],

            "validation_r2":
                val_metrics["r2"],

            "rmse_gap":
                val_metrics["rmse"]
                - train_metrics["rmse"],

            "training_seconds":
                training_time,

            "validation_inference_seconds":
                inference_time
        })

        predictions[name] = val_pred

        # Scenario-level validation metrics

        for scenario in ["baseline", "eco"]:

            mask = (
                validation["scenario"]
                == scenario
            ).to_numpy()

            if not mask.any():
                continue

            metrics = evaluate(
                y_val.to_numpy()[mask],
                val_pred[mask]
            )

            scenario_results.append({
                "model": name,
                "scenario": scenario,
                "observations": int(mask.sum()),
                **metrics
            })

        joblib.dump(
            model,
            MODEL_DIR / f"{name}.joblib",
            compress=3
        )

        logger.info(
            "Validation MAE: %.8f",
            val_metrics["mae"]
        )

        logger.info(
            "Validation RMSE: %.8f",
            val_metrics["rmse"]
        )

        logger.info(
            "Validation R2: %.5f",
            val_metrics["r2"]
        )

        logger.info(
            "Training time: %.2f seconds",
            training_time
        )

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "validation_rmse"
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "rank",
        np.arange(1, len(comparison) + 1)
    )

    dummy_rmse = comparison.loc[
        comparison["model"] == "dummy",
        "validation_rmse"
    ].iloc[0]

    comparison["rmse_improvement_percent"] = (
        (dummy_rmse - comparison["validation_rmse"])
        / dummy_rmse * 100
        if dummy_rmse > 0
        else np.nan
    )

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    pd.DataFrame(
        scenario_results
    ).to_csv(
        REPORT_DIR / "model_scenario_metrics.csv",
        index=False
    )

    return comparison, predictions


# ==========================================
# SAVE BEST CANDIDATE
# ==========================================

def save_best_model(
    comparison,
    features,
    target
):

    best = comparison.iloc[0]

    model_name = str(best["model"])

    model = joblib.load(
        MODEL_DIR / f"{model_name}.joblib"
    )

    bundle = {
        "model": model,
        "model_name": model_name,
        "features": features,
        "target": target,
        "target_unit": "litres_per_second",
        "clip_minimum": 0.0
    }

    joblib.dump(
        bundle,
        MODEL_DIR / "best_candidate.joblib",
        compress=3
    )

    metadata = {
        "project": "EcoRoute AI",
        "status": "validation_selected_candidate",
        "model_name": model_name,
        "features": features,
        "target": target,

        "selection_metric":
            "validation_rmse",

        "validation_mae":
            float(best["validation_mae"]),

        "validation_rmse":
            float(best["validation_rmse"]),

        "validation_r2":
            float(best["validation_r2"]),

        "rmse_improvement_percent":
            float(best["rmse_improvement_percent"])
            if pd.notna(
                best["rmse_improvement_percent"]
            )
            else None,

        "random_seed": SEED,

        "test_evaluated": False,

        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__
        }
    }

    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    return model_name


# ==========================================
# SAVE VALIDATION PREDICTIONS
# ==========================================

def save_predictions(
    validation,
    predictions,
    best_name,
    target
):

    output = validation[
        [
            "trajectory_id",
            "scenario",
            "sample_index"
        ]
    ].copy()

    output["actual_fuel_lps"] = validation[target]

    for name, values in predictions.items():

        output[f"predicted_{name}_lps"] = values

    output["best_prediction_lps"] = (
        predictions[best_name]
    )

    output["residual_lps"] = (
        output["actual_fuel_lps"]
        - output["best_prediction_lps"]
    )

    output.to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False
    )


# ==========================================
# VISUALIZATION UTILITIES
# ==========================================

def save_figure(fig, name):

    fig.patch.set_facecolor(BACKGROUND)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / name,
        dpi=220,
        bbox_inches="tight",
        facecolor=BACKGROUND
    )

    plt.close(fig)


# ==========================================
# MODEL COMPARISON CHART
# ==========================================

def plot_comparison(comparison):

    data = comparison.copy()

    labels = (
        data["model"]
        .str.replace("_", " ")
        .str.title()
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    metrics = [
        (
            "validation_mae",
            "Validation MAE",
            "L/s"
        ),
        (
            "validation_rmse",
            "Validation RMSE",
            "L/s"
        ),
        (
            "validation_r2",
            "Validation R-squared",
            "R2"
        )
    ]

    for ax, (metric, title, unit) in zip(
        axes, metrics
    ):

        ax.set_facecolor(SURFACE)

        colors = [
            GREEN if i == 0 else BLUE
            for i in range(len(data))
        ]

        bars = ax.barh(
            labels,
            data[metric],
            color=colors
        )

        ax.invert_yaxis()

        ax.bar_label(
            bars,
            fmt="%.4g",
            padding=4,
            fontsize=9
        )

        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.margins(x=0.18)
        ax.grid(axis="x", alpha=0.12)

    fig.suptitle(
        "EcoRoute AI | Model Performance Benchmark",
        fontsize=20
    )

    save_figure(
        fig,
        "model_performance_comparison.png"
    )


# ==========================================
# TRAINING VS VALIDATION
# ==========================================

def plot_generalization(comparison):

    positions = np.arange(len(comparison))

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.set_facecolor(SURFACE)

    ax.bar(
        positions - 0.18,
        comparison["train_rmse"],
        width=0.36,
        color=BLUE,
        label="Training RMSE"
    )

    ax.bar(
        positions + 0.18,
        comparison["validation_rmse"],
        width=0.36,
        color=PURPLE,
        label="Validation RMSE"
    )

    ax.set_xticks(positions)

    ax.set_xticklabels(
        comparison["model"]
        .str.replace("_", " ")
        .str.title(),
        rotation=20
    )

    ax.set_title(
        "Training vs Validation Generalization"
    )

    ax.set_ylabel("RMSE (L/s)")
    ax.legend()
    ax.grid(axis="y", alpha=0.15)

    save_figure(
        fig,
        "model_generalization_gap.png"
    )


# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

def plot_predictions(
    y_true,
    y_pred,
    model_name
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.set_facecolor(SURFACE)

    density = ax.hexbin(
        y_true,
        y_pred,
        gridsize=65,
        cmap="viridis",
        mincnt=1,
        bins="log"
    )

    limit = max(
        y_true.max(),
        y_pred.max()
    ) * 1.05

    ax.plot(
        [0, limit],
        [0, limit],
        "--",
        color=AMBER,
        label="Perfect prediction"
    )

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)

    ax.set_xlabel("Actual fuel rate (L/s)")
    ax.set_ylabel("Predicted fuel rate (L/s)")

    ax.set_title(
        f"Actual vs Predicted | {model_name}"
    )

    ax.legend()

    fig.colorbar(
        density,
        ax=ax,
        label="Observation count (log scale)"
    )

    save_figure(
        fig,
        "best_model_actual_vs_predicted.png"
    )


# ==========================================
# RESIDUAL DISTRIBUTION
# ==========================================

def plot_residuals(y_true, y_pred):

    residuals = (
        np.asarray(y_true)
        - np.asarray(y_pred)
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6)
    )

    for ax in axes:
        ax.set_facecolor(SURFACE)

    axes[0].hist(
        residuals,
        bins=70,
        color=GREEN,
        alpha=0.85
    )

    axes[0].axvline(
        0,
        color=AMBER,
        linestyle="--"
    )

    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Actual - Predicted (L/s)")
    axes[0].set_ylabel("Frequency")

    axes[1].hexbin(
        y_pred,
        residuals,
        gridsize=50,
        cmap="viridis",
        mincnt=1,
        bins="log"
    )

    axes[1].axhline(
        0,
        color=AMBER,
        linestyle="--"
    )

    axes[1].set_title("Residual vs Predicted")
    axes[1].set_xlabel("Predicted fuel rate (L/s)")
    axes[1].set_ylabel("Residual (L/s)")

    save_figure(
        fig,
        "best_model_residual_analysis.png"
    )


# ==========================================
# TRAINING TIME COMPARISON
# ==========================================

def plot_training_time(comparison):

    data = comparison.sort_values(
        "training_seconds"
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.set_facecolor(SURFACE)

    bars = ax.barh(
        data["model"]
        .str.replace("_", " ")
        .str.title(),
        data["training_seconds"],
        color=PURPLE
    )

    ax.bar_label(
        bars,
        fmt="%.2f s",
        padding=5
    )

    ax.set_title("Model Training Efficiency")
    ax.set_xlabel("Training time (seconds)")
    ax.margins(x=0.2)

    save_figure(
        fig,
        "model_training_time.png"
    )


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("ECOROUTE AI - MODEL TRAINING")
    logger.info("=" * 60)

    train, validation, features, target = (
        load_data()
    )

    comparison, predictions = train_all(
        train,
        validation,
        features,
        target
    )

    best_name = save_best_model(
        comparison,
        features,
        target
    )

    save_predictions(
        validation,
        predictions,
        best_name,
        target
    )

    plot_comparison(comparison)

    plot_generalization(comparison)

    plot_predictions(
        validation[target],
        predictions[best_name],
        best_name
    )

    plot_residuals(
        validation[target],
        predictions[best_name]
    )

    plot_training_time(comparison)

    print("\n" + "=" * 75)
    print("FINAL MODEL LEADERBOARD")
    print("=" * 75)

    columns = [
        "rank",
        "model",
        "validation_mae",
        "validation_rmse",
        "validation_r2",
        "rmse_improvement_percent",
        "training_seconds"
    ]

    print(
        comparison[columns]
        .round(6)
        .to_string(index=False)
    )

    print("\nBEST CANDIDATE:", best_name.upper())

    print(
        "Best validation RMSE:",
        round(
            comparison.iloc[0]["validation_rmse"],
            8
        )
    )

    print(
        "Best validation R2:",
        round(
            comparison.iloc[0]["validation_r2"],
            6
        )
    )

    print(
        "Total execution time:",
        round(
            time.perf_counter() - start,
            2
        ),
        "seconds"
    )

    print("\nModels saved:", MODEL_DIR)
    print("Reports saved:", REPORT_DIR)
    print("Figures saved:", FIGURE_DIR)

    print("\nTEST DATASET REMAINS UNTOUCHED")

    print("\nMODEL TRAINING COMPLETED!")


if __name__ == "__main__":
    main()
