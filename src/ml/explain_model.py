
"""
EcoRoute AI
Explainable Machine Learning
"""

from pathlib import Path
import sys
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from inference.fuel_predictor import FuelPredictor


REPORT_DIR = ROOT / "reports" / "xai"
FIGURE_DIR = ROOT / "assets" / "figures" / "xai"
DATA_DIR = ROOT / "data" / "ml"

for directory in [REPORT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


SEED = 42

BACKGROUND = "#080F19"
SURFACE = "#111D2C"
GREEN = "#10B981"
BLUE = "#38BDF8"
PURPLE = "#A78BFA"

plt.style.use("dark_background")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger("EcoRoute XAI")


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


def permutation_analysis(predictor, data):

    features = predictor.features
    X = data[features].copy()
    y = data["fuel_rate_lps"].to_numpy()

    baseline = predictor.predict(X)[
        "fuel_rate_lps"
    ].to_numpy()

    baseline_rmse = np.sqrt(
        mean_squared_error(y, baseline)
    )

    rng = np.random.default_rng(SEED)

    records = []

    for feature in features:

        scores = []

        for _ in range(5):

            shuffled = X.copy()

            shuffled[feature] = rng.permutation(
                shuffled[feature].to_numpy()
            )

            predictions = predictor.predict(
                shuffled
            )["fuel_rate_lps"]

            error = np.sqrt(
                mean_squared_error(y, predictions)
            )

            scores.append(
                float(error - baseline_rmse)
            )

        records.append({
            "feature": feature,
            "importance_mean": np.mean(scores),
            "importance_std": np.std(scores),
            "baseline_rmse": baseline_rmse
        })

    results = pd.DataFrame(
        records
    ).sort_values(
        "importance_mean",
        ascending=False
    )

    results.to_csv(
        REPORT_DIR / "permutation_importance.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.set_facecolor(SURFACE)

    bars = ax.barh(
        results["feature"],
        results["importance_mean"],
        xerr=results["importance_std"],
        color=GREEN,
        capsize=4
    )

    ax.bar_label(
        bars,
        fmt="%.6f",
        padding=6
    )

    ax.invert_yaxis()

    ax.set_title(
        "Permutation Feature Importance",
        fontsize=18
    )

    ax.set_xlabel("Increase in RMSE (L/s)")
    ax.margins(x=0.25)
    ax.grid(axis="x", alpha=0.12)

    save_plot(
        fig,
        "permutation_feature_importance.png"
    )

    return results


def native_feature_importance(predictor):

    if not hasattr(
        predictor.model,
        "feature_importances_"
    ):
        return

    values = np.asarray(
        predictor.model.feature_importances_
    )

    if len(values) != len(predictor.features):
        return

    results = pd.DataFrame({
        "feature": predictor.features,
        "importance": values
    }).sort_values(
        "importance",
        ascending=False
    )

    results.to_csv(
        REPORT_DIR / "native_feature_importance.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_facecolor(SURFACE)

    ax.barh(
        results["feature"],
        results["importance"],
        color=BLUE
    )

    ax.invert_yaxis()
    ax.set_title("Native Model Feature Importance")
    ax.set_xlabel("Model-specific importance")

    save_plot(
        fig,
        "native_feature_importance.png"
    )


def shap_analysis(predictor, data):

    import shap

    X = data[predictor.features].sample(
        n=min(1000, len(data)),
        random_state=SEED
    )

    explainer = shap.TreeExplainer(
        predictor.model
    )

    explanation = explainer(X)

    values = np.asarray(
        explanation.values
    )

    if values.shape != X.shape:
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    results = pd.DataFrame({
        "feature": predictor.features,

        "mean_absolute_shap":
            np.abs(values).mean(axis=0),

        "mean_shap":
            values.mean(axis=0)
    }).sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    results.to_csv(
        REPORT_DIR / "shap_feature_importance.csv",
        index=False
    )

    values_df = pd.DataFrame(
        values,
        columns=predictor.features
    )

    values_df.to_csv(
        REPORT_DIR / "shap_values.csv",
        index=False
    )

    X.to_csv(
        REPORT_DIR / "shap_input_sample.csv",
        index=False
    )

    shap.plots.beeswarm(
        explanation,
        show=False
    )

    save_plot(
        plt.gcf(),
        "shap_summary_beeswarm.png"
    )

    shap.plots.bar(
        explanation,
        show=False
    )

    save_plot(
        plt.gcf(),
        "shap_global_importance.png"
    )

    shap.plots.waterfall(
        explanation[0],
        show=False
    )

    save_plot(
        plt.gcf(),
        "shap_prediction_waterfall.png"
    )

    for feature in predictor.features:

        shap.plots.scatter(
            explanation[:, feature],
            show=False
        )

        save_plot(
            plt.gcf(),
            f"shap_dependence_{feature}.png"
        )

    return results


def feature_response_analysis(
    predictor,
    data
):

    features = predictor.features

    reference = data[features].sample(
        n=min(500, len(data)),
        random_state=SEED
    ).copy()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 11)
    )

    records = []

    for ax, feature in zip(
        axes.flatten(), features
    ):

        ax.set_facecolor(SURFACE)

        lower = reference[feature].quantile(0.05)
        upper = reference[feature].quantile(0.95)

        grid = np.linspace(lower, upper, 35)

        means = []
        lower_band = []
        upper_band = []

        for value in grid:

            modified = reference.copy()

            modified[feature] = value

            fuel = predictor.predict(
                modified
            )["fuel_rate_lph"].to_numpy()

            means.append(fuel.mean())

            lower_band.append(
                np.quantile(fuel, 0.10)
            )

            upper_band.append(
                np.quantile(fuel, 0.90)
            )

            records.append({
                "feature": feature,
                "feature_value": value,
                "mean_fuel_lph": fuel.mean(),
                "p10_fuel_lph": lower_band[-1],
                "p90_fuel_lph": upper_band[-1]
            })

        ax.plot(
            grid,
            means,
            color=GREEN,
            linewidth=2.5
        )

        ax.fill_between(
            grid,
            lower_band,
            upper_band,
            color=GREEN,
            alpha=0.15
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Predicted fuel rate (L/h)")

        ax.set_title(
            feature.replace("_", " ").title()
        )

        ax.grid(alpha=0.12)

    fig.suptitle(
        "EcoRoute AI | Model Response Profiles",
        fontsize=19
    )

    save_plot(
        fig,
        "feature_response_profiles.png"
    )

    pd.DataFrame(records).to_csv(
        REPORT_DIR / "feature_response_profiles.csv",
        index=False
    )


def main():

    predictor = FuelPredictor()

    data = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    sample = data.sample(
        n=min(5000, len(data)),
        random_state=SEED
    )

    status = {
        "model": predictor.model_name,
        "explanation_dataset": "validation",
        "shap_status": "pending",
        "permutation_status": "pending"
    }

    logger.info("Generating permutation importance")

    importance = permutation_analysis(
        predictor,
        sample
    )

    status["permutation_status"] = "completed"

    native_feature_importance(predictor)

    logger.info("Generating model response profiles")

    feature_response_analysis(
        predictor,
        sample
    )

    logger.info("Generating SHAP explanations")

    try:

        shap_results = shap_analysis(
            predictor,
            sample
        )

        status["shap_status"] = "completed"

        print("\nSHAP IMPORTANCE")

        print(
            shap_results.to_string(index=False)
        )

    except Exception as error:

        status["shap_status"] = "failed"
        status["shap_error"] = str(error)

        logger.warning(
            "SHAP unavailable: %s",
            error
        )

    (REPORT_DIR / "xai_status.json").write_text(
        json.dumps(status, indent=4),
        encoding="utf-8"
    )

    print("\nPERMUTATION IMPORTANCE")

    print(
        importance.to_string(index=False)
    )

    print("\nXAI REPORTS GENERATED")
    print("SHAP status:", status["shap_status"])


if __name__ == "__main__":
    main()

