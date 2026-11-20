
"""
EcoRoute AI
REST Prediction API
"""

from pathlib import Path
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from inference.fuel_predictor import FuelPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.predictor = FuelPredictor()

    yield

    app.state.predictor = None


app = FastAPI(
    title="EcoRoute AI",
    description=(
        "Vehicle fuel consumption and "
        "estimated CO2 prediction API."
    ),
    version="1.0.0",
    lifespan=lifespan
)


class TelemetryInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    speed_kmh: float = Field(ge=0)

    acceleration_ms2: float

    jerk_ms3: float

    distance_to_intersection_m: float = Field(
        ge=0
    )


class TrajectoryInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    observations: list[TelemetryInput] = Field(
        min_length=2,
        max_length=10000
    )

    strict: bool = False


@app.get("/")
def root():

    return {
        "service": "EcoRoute AI",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded":
            app.state.predictor is not None,
        "model_name":
            app.state.predictor.model_name
    }


@app.get("/model/info")
def model_info():

    return app.state.predictor.get_model_info()


@app.post("/predict")
def predict(
    telemetry: TelemetryInput,
    strict: bool = False
):

    try:

        return app.state.predictor.predict_single(
            **telemetry.model_dump(),
            strict=strict
        )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )


@app.post("/predict/trajectory")
def predict_trajectory(request: TrajectoryInput):

    try:

        records = [
            row.model_dump()
            for row in request.observations
        ]

        result = (
            app.state.predictor.predict_trajectory(
                records,
                strict=request.strict
            )
        )

        predictions = result.pop("predictions")

        result["predictions"] = predictions.to_dict(
            orient="records"
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )
from api.routing_api import router as routing_router

app.include_router(routing_router)       
from api.analytics_api import router as analytics_router

app.include_router(analytics_router)
from fastapi.responses import FileResponse


@app.get("/fuel-lab", include_in_schema=False)
def fuel_lab():

    path = ROOT / "dashboard" / "fuel_lab.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Fuel laboratory unavailable."
        )

    return FileResponse(
        path,
        media_type="text/html"
    )

from fastapi.responses import FileResponse


@app.get("/fuel-lab", include_in_schema=False)
def fuel_lab():

    path = ROOT / "dashboard" / "fuel_lab.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Fuel laboratory unavailable."
        )

    return FileResponse(
        path,
        media_type="text/html"
    )
