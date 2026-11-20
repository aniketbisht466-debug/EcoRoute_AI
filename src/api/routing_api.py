
"""
EcoRoute AI
Interactive Routing API
"""

from pathlib import Path
from threading import Lock
import logging

import osmnx as ox

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict

from routing.route_optimizer import EcoRouteOptimizer
from routing.driving_profile import (
    DrivingProfileSimulator,
    OBJECTIVES
)

ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(tags=["Route Optimization"])

logger = logging.getLogger("EcoRoute API")

optimizer = None
network_geojson = None
routing_lock = Lock()


class Location(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class RouteRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    origin: Location
    destination: Location
    include_dynamic: bool = False


def get_optimizer():

    global optimizer

    if optimizer is None:
        optimizer = EcoRouteOptimizer()

    return optimizer


@router.get("/dashboard", include_in_schema=False)
def dashboard():

    path = ROOT / "dashboard" / "index.html"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dashboard file not found."
        )

    return FileResponse(path, media_type="text/html")


@router.get("/routing/network")
def get_network():

    global network_geojson

    try:

        with routing_lock:

            engine = get_optimizer()

            if network_geojson is None:

                _, edges = ox.graph_to_gdfs(
                    engine.graph
                )

                roads = edges[
                    ["geometry"]
                ].to_crs(epsg=4326)

                network_geojson = (
                    roads.reset_index(drop=True)
                    .to_json(drop_id=True)
                )

        return Response(
            content=network_geojson,
            media_type="application/geo+json"
        )

    except Exception:

        logger.exception("Network loading failed")

        raise HTTPException(
            status_code=503,
            detail="Road network unavailable."
        )



@router.post("/routing/optimize")
def optimize_route(request: RouteRequest):

    origin = (
        request.origin.latitude,
        request.origin.longitude
    )

    destination = (
        request.destination.latitude,
        request.destination.longitude
    )

    try:

        with routing_lock:

            engine = get_optimizer()

            result = engine.optimize(
                origin,
                destination
            )

            result["dynamic_simulation"] = None

            if request.include_dynamic:

                simulator = DrivingProfileSimulator(
                    engine
                )

                dynamic_results = {}

                for name, route in result["routes"].items():

                    try:

                        telemetry, summary = simulator.simulate(
                            route["nodes"],
                            OBJECTIVES[name]
                        )

                        dynamic_results[name] = {
                            "status": "completed",
                            **summary
                        }

                    except ValueError as error:

                        dynamic_results[name] = {
                            "status": "unavailable",
                            "reason": str(error)
                        }

                result["dynamic_simulation"] = {
                    "mode": "dynamic_driving_simulation",

                    "assumptions": {
                        "acceleration_ms2":
                            simulator.acceleration,

                        "deceleration_ms2":
                            simulator.deceleration,

                        "signal_wait_seconds":
                            simulator.signal_wait
                    },

                    "routes": dynamic_results
                }

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except RuntimeError as error:

        raise HTTPException(
            status_code=409,
            detail=str(error)
        )

    except Exception:

        logger.exception("Route optimization failed")

        raise HTTPException(
            status_code=500,
            detail="Route optimization failed."
        )



@router.get("/routing/status")
def routing_status():

    with routing_lock:

        return {
            "status": "available",
            "network_loaded": optimizer is not None,
            "fuel_costs_ready": (
                optimizer.costs_ready
                if optimizer is not None
                else False
            ),
            "estimation_mode":
                "steady_speed_ml_simulation"
        }
