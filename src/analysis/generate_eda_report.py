
"""
EcoRoute AI
Automated Scientific EDA Report
Version 1.0

Purpose:
    Consolidate dataset statistics,
    scientific analysis, statistical results,
    visualizations and ML recommendations.
"""

from pathlib import Path
from datetime import datetime
from html import escape
import base64

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "truck_telemetry_features.csv"
)

REPORT_DIR = ROOT / "reports" / "eda"
FIGURE_DIR = ROOT / "assets" / "figures" / "eda"

OUTPUT_FILE = REPORT_DIR / "scientific_eda_report.html"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# REPORT DESIGN
# ==========================================

CSS = """
:root {
    --background: #080F19;
    --surface: #111D2C;
    --border: #253344;
    --green: #34D399;
    --blue: #38BDF8;
    --warning: #F59E0B;
    --text: #F1F5F9;
    --muted: #94A3B8;
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    background: var(--background);
    color: var(--text);
    font-family: Inter, Segoe UI, Arial, sans-serif;
    line-height: 1.7;
}

header {
    padding: 65px 30px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(
        135deg, #080F19, #103329
    );
}

.container {
    max-width: 1350px;
    margin: auto;
    padding: 30px;
}

.brand {
    color: var(--green);
    font-size: 26px;
    font-weight: 750;
}

.eyebrow {
    color: var(--green);
    letter-spacing: 3px;
    font-size: 12px;
    text-transform: uppercase;
}

h1 {
    font-size: clamp(32px, 5vw, 55px);
    line-height: 1.2;
    margin: 20px 0;
}

h2 {
    margin-top: 0;
    font-size: 27px;
}

h3 {
    color: var(--blue);
}

p {
    color: var(--muted);
}

nav {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    padding: 20px 0;
}

nav a {
    background: var(--surface);
    padding: 8px 16px;
    border-radius: 8px;
    text-decoration: none;
    color: var(--muted);
    border: 1px solid var(--border);
}

nav a:hover {
    color: var(--green);
}

section {
    margin: 35px 0;
    padding: 30px;
    border-radius: 16px;
    background: var(--surface);
    border: 1px solid var(--border);
}

.metrics {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(210px, 1fr));
    gap: 18px;
}

.metric {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 24px;
    border-radius: 14px;
}

.metric-label {
    color: var(--muted);
    font-size: 13px;
}

.metric-value {
    color: var(--green);
    font-size: 34px;
    font-weight: bold;
    margin-top: 8px;
}

.gallery {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(360px, 1fr));
    gap: 20px;
}

.figure {
    background: var(--background);
    border: 1px solid var(--border);
    padding: 14px;
    border-radius: 12px;
}

.figure img {
    width: 100%;
    height: auto;
    border-radius: 7px;
}

.figure-title {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
}

.table-wrapper {
    width: 100%;
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

th {
    background: #1D3041;
    color: var(--blue);
    text-align: left;
}

td, th {
    padding: 12px;
    border-bottom: 1px solid var(--border);
}

tr:hover td {
    background: #192B3A;
}

.warning {
    border-left: 3px solid var(--warning);
    background: rgba(245, 158, 11, 0.07);
    padding: 18px;
    margin: 20px 0;
    color: #FCD34D;
}

li {
    margin: 10px 0;
    color: var(--muted);
}

footer {
    border-top: 1px solid var(--border);
    text-align: center;
    padding: 35px;
    color: var(--muted);
}

@media(max-width: 600px) {
    .container {
        padding: 16px;
    }

    section {
        padding: 18px;
    }

    .gallery {
        grid-template-columns: 1fr;
    }
}
"""


# ==========================================
# DATA LOADING
# ==========================================

def load_csv(filename):

    path = REPORT_DIR / filename

    if not path.exists():

        print("[MISSING]", filename)

        return None

    data = pd.read_csv(path)

    print("[LOADED]", filename)

    return data


def load_telemetry():

    if not DATA_FILE.exists():

        raise FileNotFoundError(DATA_FILE)

    return pd.read_csv(DATA_FILE)


def metric(label, value):

    return f"""
    <div class="metric">
        <div class="metric-label">
            {escape(str(label))}
        </div>

        <div class="metric-value">
            {escape(str(value))}
        </div>
    </div>
    """


def format_number(value, decimals=2):

    if pd.isna(value):
        return "Unavailable"

    return f"{value:,.{decimals}f}"


def render_table(df):

    if df is None or df.empty:

        return """
        <div class="warning">
        Report unavailable or empty.
        </div>
        """

    table = df.to_html(
        index=False,
        border=0,
        escape=True,
        float_format=lambda x: f"{x:.6g}"
    )

    return f"""
    <div class="table-wrapper">
        {table}
    </div>
    """


# ==========================================
# DATASET OVERVIEW
# ==========================================

def dataset_overview(df):

    trajectories = df["trajectory_id"].nunique()

    baseline = df.loc[
        df["scenario"] == "baseline",
        "trajectory_id"
    ].nunique()

    eco = df.loc[
        df["scenario"] == "eco",
        "trajectory_id"
    ].nunique()

    features = [
        "speed_kmh",
        "acceleration_ms2",
        "jerk_ms3",
        "distance_to_intersection_m",
        "fuel_rate_lps"
    ]

    available = [
        col for col in features
        if col in df.columns
    ]

    summary = (
        df[available]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    cards = "".join([
        metric("Telemetry observations",
               f"{len(df):,}"),

        metric("Recorded trajectories",
               f"{trajectories:,}"),

        metric("Baseline trajectories",
               f"{baseline:,}"),

        metric("Eco-driving trajectories",
               f"{eco:,}"),

        metric("Dataset columns",
               len(df.columns)),

        metric("Generated EDA figures",
               len(list(FIGURE_DIR.glob("*.png"))))
    ])

    return f"""
    <section id="dataset">

        <div class="eyebrow">01 / Dataset</div>

        <h2>Dataset Overview</h2>

        <div class="metrics">
            {cards}
        </div>

        <h3>Descriptive Statistics</h3>

        {render_table(summary)}

    </section>
    """


# ==========================================
# FEATURE CORRELATIONS
# ==========================================

def correlation_analysis(df):

    features = [
        "speed_kmh",
        "acceleration_ms2",
        "jerk_ms3",
        "distance_to_intersection_m"
    ]

    target = "fuel_rate_lps"

    records = []

    for feature in features:

        if feature not in df.columns:
            continue

        pair = df[[feature, target]].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if (
            len(pair) < 2
            or pair[feature].nunique() < 2
            or pair[target].nunique() < 2
        ):
            pearson = np.nan
            spearman = np.nan

        else:
            pearson = pair[feature].corr(
                pair[target],
                method="pearson"
            )

            spearman = pair[feature].corr(
                pair[target],
                method="spearman"
            )

        records.append({
            "feature": feature,
            "pearson": pearson,
            "spearman": spearman,
            "valid_observations": len(pair)
        })

    results = pd.DataFrame(records)

    results["absolute_pearson"] = (
        results["pearson"].abs()
    )

    results = results.sort_values(
        "absolute_pearson",
        ascending=False
    ).drop(columns="absolute_pearson")

    results.to_csv(
        REPORT_DIR / "final_feature_correlations.csv",
        index=False
    )

    valid = results.dropna(subset=["pearson"])

    if valid.empty:

        finding = "No valid correlations available."

    else:

        strongest = valid.iloc[0]

        finding = (
            f"Among evaluated features, "
            f"{strongest['feature']} has the "
            f"largest absolute Pearson correlation "
            f"with fuel rate: "
            f"{strongest['pearson']:.4f}."
        )

    return f"""
    <section id="correlations">

        <div class="eyebrow">02 / Relationships</div>

        <h2>Feature Correlation Analysis</h2>

        <p>{escape(finding)}</p>

        {render_table(results)}

        <div class="warning">
            Correlation does not establish causation
            or out-of-sample predictive performance.
            These statistics may be influenced by
            dependence within trajectories.
        </div>

    </section>
    """


# ==========================================
# SCENARIO RESULTS
# ==========================================

def scenario_analysis():

    summary = load_csv(
        "scenario_comparison.csv"
    )

    behaviour = load_csv(
        "scenario_driving_behaviour.csv"
    )

    conditions = load_csv(
        "comparable_conditions.csv"
    )

    cards = ""

    if summary is not None:

        baseline = summary[
            summary["scenario"] == "baseline"
        ]

        eco = summary[
            summary["scenario"] == "eco"
        ]

        if len(baseline) and len(eco):

            baseline_rate = baseline.iloc[0][
                "weighted_fuel_l_per_100km"
            ]

            eco_rate = eco.iloc[0][
                "weighted_fuel_l_per_100km"
            ]

            savings = (
                (baseline_rate - eco_rate)
                / baseline_rate * 100
                if baseline_rate > 0
                else np.nan
            )

            cards = "".join([
                metric(
                    "Baseline L/100 km",
                    format_number(baseline_rate)
                ),

                metric(
                    "Eco-driving L/100 km",
                    format_number(eco_rate)
                ),

                metric(
                    "Observed fuel savings",
                    format_number(savings) + "%"
                )
            ])

    behaviour_summary = None

    if behaviour is not None:

        behaviour_summary = (
            behaviour
            .groupby("scenario")
            .mean(numeric_only=True)
            .reset_index()
        )

    return f"""
    <section id="scenarios">

        <div class="eyebrow">03 / Driving strategies</div>

        <h2>Baseline vs Eco-driving</h2>

        <div class="metrics">
            {cards}
        </div>

        <h3>Scenario Comparison</h3>

        {render_table(summary)}

        <h3>Driving Behaviour Summary</h3>

        {render_table(behaviour_summary)}

        <h3>Comparable Conditions</h3>

        {render_table(conditions)}

        <p>
            Comparisons are descriptive.
            Vehicle, location, payload, weather
            and traffic differences may influence
            observed fuel consumption.
        </p>

    </section>
    """


# ==========================================
# STATISTICAL FINDINGS
# ==========================================

def statistical_analysis():

    tests = load_csv(
        "statistical_test_results.csv"
    )

    bootstrap = load_csv(
        "bootstrap_confidence_interval.csv"
    )

    sensitivity = load_csv(
        "sensitivity_analysis.csv"
    )

    condition_tests = load_csv(
        "condition_statistical_results.csv"
    )

    interpretation = (
        "Bootstrap results are unavailable."
    )

    if bootstrap is not None and not bootstrap.empty:

        result = bootstrap.iloc[0]

        lower = result["ci_lower_95"]
        upper = result["ci_upper_95"]

        if not np.isfinite([lower, upper]).all():

            interpretation = (
                "Confidence interval unavailable."
            )

        elif lower > 0:

            interpretation = (
                "The exploratory bootstrap interval "
                "supports positive observed fuel savings."
            )

        elif upper < 0:

            interpretation = (
                "The exploratory bootstrap interval "
                "supports increased fuel consumption."
            )

        else:

            interpretation = (
                "The bootstrap confidence interval "
                "includes zero. No difference remains "
                "compatible with this interval."
            )

    return f"""
    <section id="statistics">

        <div class="eyebrow">04 / Statistical evidence</div>

        <h2>Statistical Hypothesis Testing</h2>

        <h3>Hypothesis Tests and Effect Sizes</h3>

        {render_table(tests)}

        <h3>Bootstrap Confidence Interval</h3>

        {render_table(bootstrap)}

        <p>{escape(interpretation)}</p>

        <h3>Condition-Level Tests</h3>

        {render_table(condition_tests)}

        <h3>Distance Threshold Sensitivity</h3>

        {render_table(sensitivity)}

        <div class="warning">
            These are exploratory statistical results.
            Repeated trajectories from the same vehicle
            or session may violate independence.
            Statistical significance does not establish
            causality or operational importance.
        </div>

    </section>
    """


# ==========================================
# VISUALIZATION GALLERY
# ==========================================

def build_gallery():

    files = sorted(
        FIGURE_DIR.glob("*.png")
    )

    cards = []

    for path in files:

        image_data = base64.b64encode(
            path.read_bytes()
        ).decode("ascii")

        title = path.stem.replace(
            "_", " "
        ).title()

        cards.append(f"""
        <div class="figure">

            <div class="figure-title">
                {escape(title)}
            </div>

            <img
                loading="lazy"
                src="data:image/png;base64,{image_data}"
                alt="{escape(title)}"
            >

        </div>
        """)

        print("[EMBEDDED]", path.name)

    gallery = "".join(cards)

    return f"""
    <section id="visualizations">

        <div class="eyebrow">05 / Visual intelligence</div>

        <h2>Scientific Visualization Gallery</h2>

        <p>{len(files)} generated visualizations.</p>

        <div class="gallery">
            {gallery}
        </div>

    </section>
    """


# ==========================================
# DATA LIMITATIONS
# ==========================================

def limitations():

    return """
    <section id="limitations">

        <div class="eyebrow">06 / Research integrity</div>

        <h2>Dataset and Research Limitations</h2>

        <ul>

            <li>
                The dataset represents a specific
                collection of heavy-duty truck recordings.
                Generalization to other vehicles requires
                independent validation.
            </li>

            <li>
                Baseline and eco-driving sample sizes
                are unequal.
            </li>

            <li>
                Adjacent telemetry observations are
                temporally dependent.
            </li>

            <li>
                Repeated recordings from the same truck
                or experimental session may also be
                dependent.
            </li>

            <li>
                CO2 emissions calculated using fuel
                conversion factors are estimates,
                not direct emission measurements.
            </li>

            <li>
                Payload, road gradient, traffic and
                weather are not established as
                available model inputs in this dataset.
            </li>

            <li>
                Distance and fuel calculations depend
                on correct measurement units and
                sampling interval assumptions.
            </li>

            <li>
                Intersection distance does not provide
                complete geographic road geometry.
            </li>

            <li>
                Descriptive and statistical associations
                do not establish causal fuel savings.
            </li>

        </ul>

    </section>
    """


# ==========================================
# MACHINE LEARNING RECOMMENDATIONS
# ==========================================

def ml_recommendations():

    features = pd.DataFrame([
        {
            "feature": "speed_kmh",
            "role": "Candidate input",
            "reason": "Vehicle movement"
        },
        {
            "feature": "acceleration_ms2",
            "role": "Candidate input",
            "reason": "Driving dynamics"
        },
        {
            "feature": "jerk_ms3",
            "role": "Candidate input",
            "reason": "Driving smoothness"
        },
        {
            "feature": "distance_to_intersection_m",
            "role": "Candidate input",
            "reason": "Intersection approach"
        },
        {
            "feature": "fuel_rate_lps",
            "role": "Prediction target",
            "reason": "Instantaneous fuel consumption"
        },
        {
            "feature": "trajectory_id",
            "role": "Grouping only",
            "reason": "Prevent trajectory leakage"
        },
        {
            "feature": "Fuel-derived measurements",
            "role": "Exclude from predictors",
            "reason": "Target leakage prevention"
        }
    ])

    return f"""
    <section id="machine-learning">

        <div class="eyebrow">07 / Predictive intelligence</div>

        <h2>Machine Learning Recommendations</h2>

        {render_table(features)}

        <h3>Proposed Modelling Strategy</h3>

        <ol>

            <li>
                Audit predictor availability at
                inference time.
            </li>

            <li>
                Create grouped training,
                validation and test partitions.
            </li>

            <li>
                Train a simple mean-prediction baseline.
            </li>

            <li>
                Train Linear Regression.
            </li>

            <li>
                Train Random Forest and XGBoost.
            </li>

            <li>
                Compare MAE, RMSE and R-squared.
            </li>

            <li>
                Evaluate trajectory-level integrated
                fuel prediction errors.
            </li>

            <li>
                Analyse model performance across
                driving scenarios and vehicle groups.
            </li>

            <li>
                Integrate the validated model with
                the vehicle physics and routing engine.
            </li>

        </ol>

        <div class="warning">
            Instantaneous fuel prediction does not
            automatically solve route-level fuel
            estimation. A future driving profile
            must be estimated from road conditions.
        </div>

    </section>
    """


# ==========================================
# GENERATE FINAL HTML
# ==========================================

def main():

    print("=" * 65)
    print("ECOROUTE AI - SCIENTIFIC EDA REPORT")
    print("=" * 65)

    df = load_telemetry()

    timestamp = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    content = "".join([
        dataset_overview(df),
        correlation_analysis(df),
        scenario_analysis(),
        statistical_analysis(),
        build_gallery(),
        limitations(),
        ml_recommendations()
    ])

    html = f"""
    <!DOCTYPE html>

    <html lang="en">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width,
              initial-scale=1.0">

        <title>
            EcoRoute AI | Scientific EDA Report
        </title>

        <style>{CSS}</style>

    </head>

    <body>

        <header>

            <div class="container">

                <div class="brand">
                    EcoRoute AI
                </div>

                <div class="eyebrow">
                    Research and Telemetry Intelligence
                </div>

                <h1>
                    Driving towards a more
                    efficient future.
                </h1>

                <p>
                    Scientific exploratory analysis
                    of heavy-duty truck telemetry,
                    driving dynamics and
                    fuel consumption.
                </p>

                <p>Generated: {timestamp}</p>

            </div>

        </header>

        <main class="container">

            <nav>

                <a href="#dataset">Dataset</a>

                <a href="#correlations">
                    Correlations
                </a>

                <a href="#scenarios">
                    Scenarios
                </a>

                <a href="#statistics">
                    Statistics
                </a>

                <a href="#visualizations">
                    Visualizations
                </a>

                <a href="#limitations">
                    Limitations
                </a>

                <a href="#machine-learning">
                    Machine Learning
                </a>

            </nav>

            {content}

        </main>

        <footer>

            EcoRoute AI |
            Scientific Exploratory Data Analysis |
            Version 1.0

        </footer>

    </body>

    </html>
    """

    OUTPUT_FILE.write_text(
        html,
        encoding="utf-8"
    )

    size_mb = (
        OUTPUT_FILE.stat().st_size
        / (1024 * 1024)
    )

    print("\n" + "=" * 65)

    print("REPORT GENERATED SUCCESSFULLY")

    print("Location:", OUTPUT_FILE)

    print(f"Report size: {size_mb:.2f} MB")

    print(
        "Embedded figures:",
        len(list(FIGURE_DIR.glob("*.png")))
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
