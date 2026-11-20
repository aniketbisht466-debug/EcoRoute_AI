
"""
EcoRoute AI
Scientific Analytics API
"""

from pathlib import Path
import json

import pandas as pd

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "reports"

FIGURES = ROOT / "assets" / "figures"

router = APIRouter(tags=["Scientific Analytics"])


ALLOWED_GROUPS = {
    "eda",
    "ml",
    "routing",
    "xai"
}


def validate_group(group):

    if group not in ALLOWED_GROUPS:

        raise HTTPException(
            status_code=404,
            detail="Unknown analytics category."
        )


def resolve_file(directory, filename, extensions):

    path = (directory / filename).resolve()

    if (
        path.parent != directory.resolve()
        or path.suffix.lower() not in extensions
        or not path.is_file()
    ):

        raise HTTPException(
            status_code=404,
            detail="Requested artifact not found."
        )

    return path


@router.get("/analytics/overview")
def analytics_overview():

    categories = {}

    for group in sorted(ALLOWED_GROUPS):

        report_dir = REPORTS / group
        figure_dir = FIGURES / group

        reports = sorted([
            file.name
            for file in report_dir.glob("*")
            if file.is_file()
            and file.suffix.lower() in {
                ".csv", ".json"
            }
        ])

        figures = sorted([
            file.name
            for file in figure_dir.glob("*")
            if file.is_file()
            and file.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".webp"
            }
        ])

        categories[group] = {
            "report_count": len(reports),
            "figure_count": len(figures),
            "reports": reports,
            "figures": figures
        }

    return {
        "project": "EcoRoute AI",
        "categories": categories,
        "total_reports": sum(
            group["report_count"]
            for group in categories.values()
        ),
        "total_figures": sum(
            group["figure_count"]
            for group in categories.values()
        )
    }


@router.get("/analytics/dataset")
def dataset_summary():

    path = (
        ROOT
        / "data"
        / "processed"
        / "truck_telemetry_features.csv"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Feature dataset unavailable."
        )

    df = pd.read_csv(
        path,
        usecols=[
            "trajectory_id",
            "scenario",
            "speed_kmh",
            "fuel_rate_lps",
            "acceleration_ms2"
        ]
    )

    return {
        "observations": int(len(df)),
        "trajectories": int(
            df["trajectory_id"].nunique()
        ),
        "baseline_trajectories": int(
            df.loc[
                df["scenario"] == "baseline",
                "trajectory_id"
            ].nunique()
        ),
        "eco_trajectories": int(
            df.loc[
                df["scenario"] == "eco",
                "trajectory_id"
            ].nunique()
        ),
        "mean_speed_kmh": round(
            float(df["speed_kmh"].mean()), 3
        ),
        "maximum_speed_kmh": round(
            float(df["speed_kmh"].max()), 3
        ),
        "mean_fuel_rate_lps": round(
            float(df["fuel_rate_lps"].mean()), 6
        ),
        "mean_absolute_acceleration": round(
            float(
                df["acceleration_ms2"].abs().mean()
            ), 4
        )
    }


@router.get("/analytics/reports/{group}/{filename}")
def get_report(group: str, filename: str):

    validate_group(group)

    path = resolve_file(
        REPORTS / group,
        filename,
        {".csv", ".json"}
    )

    if path.suffix.lower() == ".json":

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    df = pd.read_csv(path)

    preview = df.head(500)

    rows = json.loads(
        preview.to_json(orient="records")
    )

    return {
        "filename": filename,
        "columns": list(df.columns),
        "total_rows": int(len(df)),
        "displayed_rows": len(rows),
        "rows": rows
    }


@router.get("/analytics/figures/{group}/{filename}")
def get_figure(group: str, filename: str):

    validate_group(group)

    path = resolve_file(
        FIGURES / group,
        filename,
        {".png", ".jpg", ".jpeg", ".webp"}
    )

    return FileResponse(path)


@router.get(
    "/analytics",
    include_in_schema=False
)
def analytics_page():

    path = ROOT / "dashboard" / "analytics.html"

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Analytics page not found."
        )

    return FileResponse(
        path,
        media_type="text/html"
    )
