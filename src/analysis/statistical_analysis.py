
"""
EcoRoute AI
Statistical Hypothesis Testing
Version 1.0

Purpose:
    Evaluate fuel consumption differences,
    effect sizes, confidence intervals and
    sensitivity to analytical decisions.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from visualization.plot_theme import (
    COLORS,
    apply_theme,
    save_plot
)


REPORT_DIR = ROOT / "reports" / "eda"
FIGURE_DIR = ROOT / "assets" / "figures" / "eda"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = REPORT_DIR / "trajectory_summary.csv"

CONDITION_FILE = REPORT_DIR / "comparable_conditions.csv"

ALPHA = 0.05
N_BOOTSTRAP = 5000
RANDOM_SEED = 42
MIN_DISTANCE_KM = 0.1

apply_theme()


def load_data():

    df = pd.read_csv(SUMMARY_FILE)

    columns = [
        "distance_km",
        "duration_s",
        "total_fuel_l",
        "fuel_l_per_100km"
    ]

    df[columns] = df[columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(subset=columns)

    df = df[
        (df["duration_s"] > 0)
        & (df["distance_km"] > 0)
        & (df["total_fuel_l"] >= 0)
        & df["scenario"].isin(["baseline", "eco"])
    ].copy()

    return df


def filter_distance(df, threshold):

    return df[
        df["distance_km"] >= threshold
    ].copy()


def weighted_fuel_rate(df):

    return (
        df["total_fuel_l"].sum()
        / df["distance_km"].sum()
        * 100
    )


def calculate_savings(baseline, eco):

    if baseline <= 0:
        return np.nan

    return (baseline - eco) / baseline * 100


def cohens_d(a, b):

    n1 = len(a)
    n2 = len(b)

    pooled_variance = (
        (n1 - 1) * np.var(a, ddof=1)
        + (n2 - 1) * np.var(b, ddof=1)
    ) / (n1 + n2 - 2)

    if pooled_variance <= 0:
        return np.nan

    return (
        np.mean(a) - np.mean(b)
    ) / np.sqrt(pooled_variance)


def perform_statistical_tests(df):

    baseline = df.loc[
        df["scenario"] == "baseline",
        "fuel_l_per_100km"
    ].to_numpy()

    eco = df.loc[
        df["scenario"] == "eco",
        "fuel_l_per_100km"
    ].to_numpy()

    if len(baseline) < 2 or len(eco) < 2:
        raise ValueError(
            "Insufficient trajectories for testing."
        )

    welch = stats.ttest_ind(
        baseline,
        eco,
        equal_var=False
    )

    mann_whitney = stats.mannwhitneyu(
        baseline,
        eco,
        alternative="two-sided"
    )

    effect_size = cohens_d(baseline, eco)

    rank_biserial = (
        2 * mann_whitney.statistic
        / (len(baseline) * len(eco))
        - 1
    )

    results = pd.DataFrame([
        {
            "test": "Welch t-test",
            "statistic": welch.statistic,
            "p_value": welch.pvalue,
            "effect_size": effect_size,
            "effect_measure": "Cohen d"
        },
        {
            "test": "Mann-Whitney U",
            "statistic": mann_whitney.statistic,
            "p_value": mann_whitney.pvalue,
            "effect_size": rank_biserial,
            "effect_measure": "Rank biserial"
        }
    ])

    results["significant_at_005"] = (
        results["p_value"] < ALPHA
    )

    results.to_csv(
        REPORT_DIR / "statistical_test_results.csv",
        index=False
    )

    print("\nTRAJECTORY-LEVEL STATISTICS")
    print("Baseline trajectories:", len(baseline))
    print("Eco trajectories:", len(eco))

    print(
        "Baseline mean:",
        round(baseline.mean(), 4)
    )

    print(
        "Eco mean:",
        round(eco.mean(), 4)
    )

    print("\nHYPOTHESIS TEST RESULTS")

    print(results.to_string(index=False))

    print(
        "\nPositive effect sizes indicate "
        "higher consumption in baseline recordings."
    )

    print(
        "Tests assume independent trajectories; "
        "interpret as exploratory."
    )

    return results


def bootstrap_fuel_savings(df):

    rng = np.random.default_rng(RANDOM_SEED)

    baseline = df.loc[
        df["scenario"] == "baseline",
        ["total_fuel_l", "distance_km"]
    ].to_numpy()

    eco = df.loc[
        df["scenario"] == "eco",
        ["total_fuel_l", "distance_km"]
    ].to_numpy()

    estimates = []

    for _ in range(N_BOOTSTRAP):

        baseline_sample = baseline[
            rng.integers(
                0, len(baseline), len(baseline)
            )
        ]

        eco_sample = eco[
            rng.integers(
                0, len(eco), len(eco)
            )
        ]

        baseline_rate = (
            baseline_sample[:, 0].sum()
            / baseline_sample[:, 1].sum()
            * 100
        )

        eco_rate = (
            eco_sample[:, 0].sum()
            / eco_sample[:, 1].sum()
            * 100
        )

        estimates.append(
            calculate_savings(
                baseline_rate,
                eco_rate
            )
        )

    estimates = np.asarray(estimates)
    estimates = estimates[np.isfinite(estimates)]

    if len(estimates) == 0:
        raise ValueError(
            "No valid bootstrap estimates."
        )

    lower, upper = np.percentile(
        estimates,
        [2.5, 97.5]
    )

    observed = calculate_savings(
        weighted_fuel_rate(
            df[df["scenario"] == "baseline"]
        ),
        weighted_fuel_rate(
            df[df["scenario"] == "eco"]
        )
    )

    summary = pd.DataFrame([{
        "observed_savings_percent": observed,
        "bootstrap_mean_percent": estimates.mean(),
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "valid_iterations": len(estimates),
        "random_seed": RANDOM_SEED
    }])

    summary.to_csv(
        REPORT_DIR /
        "bootstrap_confidence_interval.csv",
        index=False
    )

    pd.DataFrame({
        "fuel_savings_percent": estimates
    }).to_csv(
        REPORT_DIR / "bootstrap_samples.csv",
        index=False
    )

    print("\nBOOTSTRAP CONFIDENCE INTERVAL")

    print(summary.round(4).to_string(index=False))

    plot_bootstrap_distribution(
        estimates, observed, lower, upper
    )

    return summary


def plot_bootstrap_distribution(
    estimates, observed, lower, upper
):

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        estimates,
        bins=50,
        color=COLORS["eco"],
        alpha=0.85
    )

    ax.axvspan(
        lower,
        upper,
        color=COLORS["eco"],
        alpha=0.10,
        label="95% confidence interval"
    )

    ax.axvline(
        observed,
        color=COLORS["baseline"],
        linewidth=2.5,
        label=f"Observed: {observed:.2f}%"
    )

    for value in [lower, upper]:

        ax.axvline(
            value,
            color=COLORS["muted"],
            linestyle="--",
            linewidth=1.5
        )

    ax.axvline(
        0,
        color="#EF4444",
        linestyle=":",
        label="No fuel savings"
    )

    ax.set_title(
        "Bootstrap Distribution of Fuel Savings"
    )

    ax.set_xlabel("Fuel savings (%)")
    ax.set_ylabel("Frequency")

    ax.legend()

    save_plot(
        fig,
        FIGURE_DIR / "bootstrap_fuel_savings.png"
    )


def analyse_comparable_conditions():

    if not CONDITION_FILE.exists():

        print(
            "\nComparable-condition file unavailable."
        )

        return

    conditions = pd.read_csv(CONDITION_FILE)

    required = [
        "baseline_l_per_100km",
        "eco_l_per_100km"
    ]

    conditions = conditions.dropna(
        subset=required
    ).copy()

    if len(conditions) < 2:

        print(
            "\nInsufficient comparable groups "
            "for condition-level analysis."
        )

        return

    conditions["difference_l_per_100km"] = (
        conditions["baseline_l_per_100km"]
        - conditions["eco_l_per_100km"]
    )

    differences = conditions[
        "difference_l_per_100km"
    ].to_numpy()

    print("\nCOMPARABLE CONDITION ANALYSIS")

    print("Comparable groups:", len(conditions))

    print(
        "Groups favouring eco-driving:",
        int((differences > 0).sum())
    )

    print(
        "Groups favouring baseline:",
        int((differences < 0).sum())
    )

    print(
        "Mean group difference:",
        round(differences.mean(), 4)
    )

    # Exploratory matched-group comparison.
    if np.all(differences == 0):

        statistic = 0.0
        p_value = 1.0

    else:

        test = stats.wilcoxon(
            differences,
            alternative="two-sided"
        )

        statistic = test.statistic
        p_value = test.pvalue

    result = pd.DataFrame([{
        "groups": len(conditions),
        "mean_difference_l_per_100km":
            differences.mean(),
        "median_difference_l_per_100km":
            np.median(differences),
        "wilcoxon_statistic": statistic,
        "p_value": p_value
    }])

    result.to_csv(
        REPORT_DIR /
        "condition_statistical_results.csv",
        index=False
    )

    print(result.round(6).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 7))

    x = conditions["baseline_l_per_100km"]
    y = conditions["eco_l_per_100km"]

    ax.scatter(
        x,
        y,
        color=COLORS["eco"],
        alpha=0.8,
        s=70
    )

    limit = max(x.max(), y.max()) * 1.1

    ax.plot(
        [0, limit],
        [0, limit],
        linestyle="--",
        color=COLORS["muted"],
        label="Equal fuel consumption"
    )

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)

    ax.set_xlabel("Baseline fuel (L/100 km)")
    ax.set_ylabel("Eco-driving fuel (L/100 km)")

    ax.set_title(
        "Comparable Condition Fuel Consumption"
    )

    ax.legend()

    save_plot(
        fig,
        FIGURE_DIR /
        "condition_level_statistical_comparison.png"
    )


def perform_sensitivity_analysis(df):

    records = []

    thresholds = [
        0.0,
        0.1,
        0.25,
        0.5,
        1.0
    ]

    for threshold in thresholds:

        data = filter_distance(df, threshold)

        baseline = data[
            data["scenario"] == "baseline"
        ]

        eco = data[
            data["scenario"] == "eco"
        ]

        if baseline.empty or eco.empty:
            continue

        baseline_rate = weighted_fuel_rate(baseline)
        eco_rate = weighted_fuel_rate(eco)

        records.append({
            "minimum_distance_km": threshold,
            "baseline_trajectories": len(baseline),
            "eco_trajectories": len(eco),
            "baseline_l_per_100km": baseline_rate,
            "eco_l_per_100km": eco_rate,
            "fuel_savings_percent":
                calculate_savings(
                    baseline_rate,
                    eco_rate
                )
        })

    summary = pd.DataFrame(records)

    summary.to_csv(
        REPORT_DIR / "sensitivity_analysis.csv",
        index=False
    )

    print("\nDISTANCE THRESHOLD SENSITIVITY")

    print(summary.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(
        summary["minimum_distance_km"],
        summary["fuel_savings_percent"],
        color=COLORS["eco"],
        marker="o",
        linewidth=2.5
    )

    ax.axhline(
        0,
        color="#EF4444",
        linestyle="--"
    )

    ax.set_title(
        "Fuel Savings Sensitivity to Distance Filtering"
    )

    ax.set_xlabel(
        "Minimum trajectory distance (km)"
    )

    ax.set_ylabel("Observed fuel savings (%)")

    save_plot(
        fig,
        FIGURE_DIR /
        "distance_threshold_sensitivity.png"
    )


def plot_fuel_distribution(df):

    fig, ax = plt.subplots(figsize=(12, 7))

    for scenario in ["baseline", "eco"]:

        values = np.sort(
            df.loc[
                df["scenario"] == scenario,
                "fuel_l_per_100km"
            ].to_numpy()
        )

        cumulative = (
            np.arange(1, len(values) + 1)
            / len(values)
        )

        ax.step(
            values,
            cumulative,
            where="post",
            linewidth=2.2,
            color=COLORS[scenario],
            label=scenario.title()
        )

    ax.set_title(
        "Cumulative Fuel Consumption Distributions"
    )

    ax.set_xlabel("Fuel consumption (L/100 km)")
    ax.set_ylabel("Cumulative proportion")

    ax.legend()

    save_plot(
        fig,
        FIGURE_DIR / "fuel_consumption_ecdf.png"
    )


def main():

    print("=" * 65)
    print("ECOROUTE AI - STATISTICAL ANALYSIS")
    print("=" * 65)

    raw = load_data()

    df = filter_distance(
        raw,
        MIN_DISTANCE_KM
    )

    print("\nValid trajectories:", len(raw))
    print("Eligible trajectories:", len(df))

    test_results = perform_statistical_tests(df)

    bootstrap_summary = bootstrap_fuel_savings(df)

    analyse_comparable_conditions()

    perform_sensitivity_analysis(raw)

    plot_fuel_distribution(df)

    result = bootstrap_summary.iloc[0]

    print("\n" + "=" * 65)
    print("FINAL STATISTICAL FINDINGS")
    print("=" * 65)

    print(
        "Observed fuel savings:",
        round(result["observed_savings_percent"], 4),
        "%"
    )

    print(
        "95% confidence interval:",
        round(result["ci_lower_95"], 4),
        "to",
        round(result["ci_upper_95"], 4),
        "%"
    )

    if result["ci_lower_95"] > 0:

        print(
            "Bootstrap interval supports "
            "positive observed fuel savings."
        )

    elif result["ci_upper_95"] < 0:

        print(
            "Bootstrap interval supports "
            "increased fuel consumption."
        )

    else:

        print(
            "Bootstrap interval includes zero."
        )

    print(
        "\nResults are exploratory associations, "
        "not proof of causality."
    )

    print("\nSTATISTICAL ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
