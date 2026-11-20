
"""
EcoRoute AI
Automated System Integration Tests
Version 1.0
"""

from pathlib import Path
import sys
import unittest
import math
import json

import pandas as pd

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.api.main import app
from routing.network_manager import RoadNetworkManager


class EcoRouteSystemTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.client = TestClient(app)

        cls.client.__enter__()

        cls.addClassCleanup(
            cls.client.__exit__, None, None, None
        )

        response = cls.client.get("/model/info")

        if response.status_code != 200:
            raise RuntimeError(response.text)

        cls.model_info = response.json()

        cls.features = cls.model_info["features"]

        cls.domain = cls.model_info["feature_domain"]

        train = pd.read_csv(
            ROOT / "data" / "ml" / "train.csv"
        )

        cls.telemetry = {
            feature: float(train.iloc[0][feature])
            for feature in cls.features
        }

        cls.journey = {
            "origin": {
                "latitude": 28.5708,
                "longitude": 77.3261
            },
            "destination": {
                "latitude": 28.5355,
                "longitude": 77.3910
            }
        }

    def require_cached_network(self):

        manager = RoadNetworkManager()

        path = manager.get_cache_path(
            manager.config["default_city"]
        )

        self.assertTrue(
            path.exists(),
            "Cached road network missing."
        )

    # ======================================
    # FILE VALIDATION
    # ======================================

    def test_01_required_files(self):

        files = [
            "models/final_model.joblib",
            "models/final_model_metadata.json",
            "data/ml/ml_config.json",
            "data/ml/train.csv",
            "data/ml/validation.csv",
            "data/ml/test.csv",
            "dashboard/index.html",
            "dashboard/analytics.html",
            "dashboard/fuel_lab.html",
            "configs/routing_config.json"
        ]

        for filename in files:

            with self.subTest(file=filename):

                self.assertTrue(
                    (ROOT / filename).is_file(),
                    f"Missing: {filename}"
                )

        self.require_cached_network()

    # ======================================
    # BACKEND HEALTH
    # ======================================

    def test_02_health_endpoint(self):

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(
            data["status"], "healthy"
        )

        self.assertTrue(data["model_loaded"])

    # ======================================
    # MODEL METADATA
    # ======================================

    def test_03_model_information(self):

        data = self.model_info

        self.assertEqual(
            len(data["features"]), 4
        )

        self.assertEqual(
            set(data["features"]),
            {
                "speed_kmh",
                "acceleration_ms2",
                "jerk_ms3",
                "distance_to_intersection_m"
            }
        )

        self.assertGreater(data["co2_factor"], 0)

        for feature in data["features"]:

            domain = data["feature_domain"][feature]

            self.assertLessEqual(
                domain["min"], domain["median"]
            )

            self.assertLessEqual(
                domain["median"], domain["max"]
            )

    # ======================================
    # FUEL PREDICTION
    # ======================================

    def test_04_fuel_prediction(self):

        response = self.client.post(
            "/predict?strict=true",
            json=self.telemetry
        )

        self.assertEqual(
            response.status_code,
            200,
            response.text
        )

        result = response.json()

        fuel = result["fuel_rate_lps"]

        self.assertTrue(math.isfinite(fuel))

        self.assertGreaterEqual(fuel, 0)

        self.assertTrue(
            result["within_training_range"]
        )

        self.assertAlmostEqual(
            result["fuel_rate_lph"],
            fuel * 3600,
            places=6
        )

    # ======================================
    # CO2 CALCULATION
    # ======================================

    def test_05_emission_calculation(self):

        result = self.client.post(
            "/predict",
            json=self.telemetry
        )

        self.assertEqual(result.status_code, 200)

        data = result.json()

        expected = (
            data["fuel_rate_lps"]
            * self.model_info["co2_factor"]
            * 1000
        )

        self.assertAlmostEqual(
            data["co2_rate_gps"],
            expected,
            places=6
        )

    # ======================================
    # INVALID TELEMETRY
    # ======================================

    def test_06_negative_speed_rejected(self):

        payload = {
            **self.telemetry,
            "speed_kmh": -10
        }

        response = self.client.post(
            "/predict",
            json=payload
        )

        self.assertEqual(
            response.status_code, 422
        )

    def test_07_missing_feature_rejected(self):

        payload = self.telemetry.copy()

        payload.pop("jerk_ms3")

        response = self.client.post(
            "/predict",
            json=payload
        )

        self.assertEqual(
            response.status_code, 422
        )

    # ======================================
    # MODEL DOMAIN VALIDATION
    # ======================================

    def test_08_strict_domain_validation(self):

        maximum = self.domain["speed_kmh"]["max"]

        payload = {
            **self.telemetry,
            "speed_kmh": maximum + 20
        }

        response = self.client.post(
            "/predict?strict=true",
            json=payload
        )

        self.assertEqual(
            response.status_code, 422
        )

    def test_09_domain_warning(self):

        maximum = self.domain["speed_kmh"]["max"]

        payload = {
            **self.telemetry,
            "speed_kmh": maximum + 20
        }

        response = self.client.post(
            "/predict?strict=false",
            json=payload
        )

        self.assertEqual(
            response.status_code, 200
        )

        data = response.json()

        self.assertFalse(
            data["within_training_range"]
        )

        self.assertTrue(data["warnings"])

    # ======================================
    # TRAJECTORY PREDICTION
    # ======================================

    def test_10_trajectory_prediction(self):

        response = self.client.post(
            "/predict/trajectory",
            json={
                "observations": [
                    self.telemetry.copy()
                    for _ in range(5)
                ],
                "strict": True
            }
        )

        self.assertEqual(
            response.status_code,
            200,
            response.text
        )

        data = response.json()

        self.assertEqual(data["observations"], 5)

        expected_duration = (
            4
            * self.model_info[
                "sampling_interval_seconds"
            ]
        )

        self.assertAlmostEqual(
            data["duration_s"],
            expected_duration
        )

        self.assertGreaterEqual(
            data["estimated_fuel_l"], 0
        )

        self.assertAlmostEqual(
            data["estimated_co2_kg"],
            data["estimated_fuel_l"]
            * self.model_info["co2_factor"],
            places=6
        )

    # ======================================
    # DASHBOARD PAGES
    # ======================================

    def test_11_dashboard_pages(self):

        pages = [
            "/dashboard",
            "/analytics",
            "/fuel-lab",
            "/docs"
        ]

        for page in pages:

            with self.subTest(page=page):

                response = self.client.get(page)

                self.assertEqual(
                    response.status_code,
                    200,
                    response.text[:200]
                )

                self.assertIn(
                    "text/html",
                    response.headers["content-type"]
                )

    # ======================================
    # ANALYTICS ENGINE
    # ======================================

    def test_12_dataset_analytics(self):

        response = self.client.get(
            "/analytics/dataset"
        )

        self.assertEqual(
            response.status_code, 200
        )

        data = response.json()

        self.assertGreater(
            data["observations"], 0
        )

        self.assertGreater(
            data["trajectories"], 0
        )

        self.assertEqual(
            data["baseline_trajectories"]
            + data["eco_trajectories"],
            data["trajectories"]
        )

    def test_13_analytics_categories(self):

        response = self.client.get(
            "/analytics/overview"
        )

        self.assertEqual(
            response.status_code, 200
        )

        data = response.json()

        for group in [
            "eda", "ml", "routing", "xai"
        ]:

            self.assertIn(
                group, data["categories"]
            )

        self.assertGreater(
            data["total_reports"], 0
        )

    # ======================================
    # GEOSPATIAL NETWORK
    # ======================================

    def test_14_road_network(self):

        self.require_cached_network()

        response = self.client.get(
            "/routing/network"
        )

        self.assertEqual(
            response.status_code,
            200,
            response.text[:200]
        )

        data = response.json()

        self.assertEqual(
            data["type"],
            "FeatureCollection"
        )

        self.assertGreater(
            len(data["features"]), 0
        )

    # ======================================
    # ROUTE OPTIMIZATION
    # ======================================

    def test_15_route_optimization(self):

        self.require_cached_network()

        response = self.client.post(
            "/routing/optimize",
            json={
                **self.journey,
                "include_dynamic": False
            }
        )

        self.assertEqual(
            response.status_code,
            200,
            response.text[:300]
        )

        data = response.json()

        routes = data["routes"]

        self.assertIn("shortest", routes)
        self.assertIn("fastest", routes)

        for name, route in routes.items():

            with self.subTest(route=name):

                metrics = route["metrics"]

                self.assertGreater(
                    metrics["distance_km"], 0
                )

                self.assertGreater(
                    metrics["duration_minutes"], 0
                )

                self.assertGreater(
                    len(route["geometry"]), 1
                )

                fuel = metrics["estimated_fuel_l"]

                if metrics["out_of_domain_edges"] > 0:

                    self.assertIsNone(fuel)

                    self.assertIsNone(
                        metrics["estimated_co2_kg"]
                    )

                else:

                    self.assertIsNotNone(fuel)

                    self.assertGreaterEqual(fuel, 0)

                    self.assertAlmostEqual(
                        metrics["estimated_co2_kg"],
                        fuel
                        * self.model_info["co2_factor"],
                        places=6
                    )

        shortest = routes["shortest"]["metrics"]
        fastest = routes["fastest"]["metrics"]

        self.assertLessEqual(
            shortest["distance_km"],
            fastest["distance_km"] + 1e-6
        )

        self.assertLessEqual(
            fastest["duration_minutes"],
            shortest["duration_minutes"] + 1e-6
        )

        recommended = data["recommended_route"]

        if recommended is not None:

            metrics = routes[recommended]["metrics"]

            self.assertTrue(
                metrics["within_detour_limit"]
            )

            self.assertIsNotNone(
                metrics["estimated_fuel_l"]
            )

    # ======================================
    # DYNAMIC DRIVING SIMULATION
    # ======================================

    def test_16_dynamic_simulation(self):

        self.require_cached_network()

        response = self.client.post(
            "/routing/optimize",
            json={
                **self.journey,
                "include_dynamic": True
            }
        )

        self.assertEqual(
            response.status_code,
            200,
            response.text[:300]
        )

        data = response.json()

        simulation = data["dynamic_simulation"]

        self.assertIsNotNone(simulation)

        self.assertEqual(
            set(simulation["routes"]),
            set(data["routes"])
        )

        for name, result in simulation["routes"].items():

            with self.subTest(route=name):

                self.assertEqual(
                    result["status"],
                    "completed",
                    result.get("reason", "")
                )

                self.assertGreater(
                    result["simulated_duration_minutes"],
                    0
                )

                self.assertGreater(
                    result["observations"], 1
                )

                self.assertGreaterEqual(
                    result["domain_coverage_percent"], 0
                )

                self.assertLessEqual(
                    result["domain_coverage_percent"], 100
                )

                self.assertAlmostEqual(
                    result["distance_km"],
                    data["routes"][name]["metrics"][
                        "distance_km"
                    ],
                    places=6
                )

                if result["out_of_domain_observations"] > 0:

                    self.assertIsNone(
                        result["estimated_fuel_l"]
                    )

                else:

                    self.assertGreaterEqual(
                        result["estimated_fuel_l"], 0
                    )

                    self.assertAlmostEqual(
                        result["estimated_co2_kg"],
                        result["estimated_fuel_l"]
                        * self.model_info["co2_factor"],
                        places=6
                    )

    # ======================================
    # INVALID ROUTING REQUESTS
    # ======================================

    def test_17_invalid_coordinates(self):

        response = self.client.post(
            "/routing/optimize",
            json={
                "origin": {
                    "latitude": 200,
                    "longitude": 77
                },
                "destination": {
                    "latitude": 28,
                    "longitude": 77
                }
            }
        )

        self.assertEqual(
            response.status_code, 422
        )

    def test_18_identical_locations(self):

        self.require_cached_network()

        response = self.client.post(
            "/routing/optimize",
            json={
                "origin": self.journey["origin"],
                "destination": self.journey["origin"]
            }
        )

        self.assertEqual(
            response.status_code, 422
        )

    # ======================================
    # FINAL EVALUATION ARTIFACTS
    # ======================================

    def test_19_model_evaluation(self):

        path = (
            ROOT
            / "reports"
            / "ml"
            / "final_test_metrics.json"
        )

        self.assertTrue(path.exists())

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        for metric in ["mae", "rmse", "r2"]:

            self.assertIn(metric, data)

            self.assertTrue(
                math.isfinite(data[metric])
            )

            self.assertAlmostEqual(
                data[metric],
                self.model_info["test_metrics"][metric],
                places=8
            )

        self.assertGreaterEqual(data["mae"], 0)
        self.assertGreaterEqual(data["rmse"], 0)
        self.assertLessEqual(data["r2"], 1)


if __name__ == "__main__":

    print("=" * 65)
    print("ECOROUTE AI - SYSTEM QUALITY ASSURANCE")
    print("=" * 65)

    unittest.main(verbosity=2)
