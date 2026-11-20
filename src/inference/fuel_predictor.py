
"""
EcoRoute AI
Production Fuel Prediction Engine
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "final_model.joblib"

METADATA_PATH = (
    ROOT / "models" / "final_model_metadata.json"
)


class FuelPredictor:

    def __init__(self, model_path=MODEL_PATH):

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

        actual_hash = hashlib.sha256(
            model_path.read_bytes()
        ).hexdigest()

        if actual_hash != self.metadata["model_sha256"]:
            raise ValueError("Model checksum mismatch.")

        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.features = bundle["features"]
        self.domain = bundle["feature_domain"]

        self.co2_factor = float(
            bundle["diesel_co2_kg_per_litre"]
        )

        self.sampling_interval = float(
            bundle["sampling_interval_seconds"]
        )

    def validate_inputs(self, data):

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        elif not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if data.empty:
            raise ValueError("Input data is empty.")

        missing = set(self.features) - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing features: {sorted(missing)}"
            )

        X = data[self.features].copy()

        for column in self.features:

            X[column] = pd.to_numeric(
                X[column],
                errors="raise"
            )

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                "Input contains non-finite values."
            )

        if (X["speed_kmh"] < 0).any():
            raise ValueError(
                "Speed cannot be negative."
            )

        if (X["distance_to_intersection_m"] < 0).any():
            raise ValueError(
                "Intersection distance cannot be negative."
            )

        return X

    def detect_out_of_domain(self, X):

        warnings = [[] for _ in range(len(X))]

        for feature in self.features:

            lower = self.domain[feature]["min"]
            upper = self.domain[feature]["max"]

            values = X[feature].to_numpy()

            invalid = (
                (values < lower)
                | (values > upper)
            )

            for index in np.flatnonzero(invalid):

                warnings[index].append(
                    f"{feature} outside training range "
                    f"[{lower:.4f}, {upper:.4f}]"
                )

        return warnings

    def predict(self, data, strict=False):

        X = self.validate_inputs(data)

        warnings = self.detect_out_of_domain(X)

        if strict and any(warnings):
            raise ValueError(
                "Input outside model training domain."
            )

        fuel = np.asarray(
            self.model.predict(X),
            dtype=float
        ).reshape(-1)

        if not np.isfinite(fuel).all():
            raise RuntimeError(
                "Non-finite model output."
            )

        fuel = np.maximum(fuel, 0)

        results = X.reset_index(drop=True).copy()

        results["fuel_rate_lps"] = fuel

        results["fuel_rate_lph"] = fuel * 3600

        results["co2_rate_gps"] = (
            fuel * self.co2_factor * 1000
        )

        results["within_training_range"] = [
            not bool(item) for item in warnings
        ]

        results["warnings"] = warnings

        return results

    def predict_single(
        self,
        speed_kmh,
        acceleration_ms2,
        jerk_ms3,
        distance_to_intersection_m,
        strict=False
    ):

        inputs = {
            "speed_kmh": speed_kmh,
            "acceleration_ms2": acceleration_ms2,
            "jerk_ms3": jerk_ms3,
            "distance_to_intersection_m":
                distance_to_intersection_m
        }

        result = self.predict(
            inputs,
            strict=strict
        )

        return result.to_dict(
            orient="records"
        )[0]

    def predict_trajectory(
        self,
        telemetry,
        time_column=None,
        strict=False
    ):

        data = pd.DataFrame(telemetry).copy()

        if len(data) < 2:
            raise ValueError(
                "At least two observations required."
            )

        if time_column is not None:

            if time_column not in data.columns:
                raise ValueError(
                    f"Missing time column: {time_column}"
                )

            timestamps = pd.to_numeric(
                data[time_column],
                errors="raise"
            ).to_numpy(dtype=float)

        else:

            timestamps = (
                np.arange(len(data))
                * self.sampling_interval
            )

        intervals = np.diff(timestamps)

        if (
            not np.isfinite(timestamps).all()
            or (intervals <= 0).any()
        ):
            raise ValueError(
                "Timestamps must be finite and increasing."
            )

        if not np.allclose(
            intervals,
            self.sampling_interval
        ):
            raise ValueError(
                "Trajectory sampling interval does not "
                "match the model configuration."
            )

        predictions = self.predict(
            data,
            strict=strict
        )

        fuel = predictions[
            "fuel_rate_lps"
        ].to_numpy()

        speed_ms = (
            predictions["speed_kmh"]
            .to_numpy() / 3.6
        )

        interval_fuel = (
            (fuel[:-1] + fuel[1:])
            / 2 * intervals
        )

        interval_distance = (
            (speed_ms[:-1] + speed_ms[1:])
            / 2 * intervals
        )

        total_fuel = float(interval_fuel.sum())

        distance_km = float(
            interval_distance.sum() / 1000
        )

        duration = float(
            timestamps[-1] - timestamps[0]
        )

        predictions["elapsed_time_s"] = (
            timestamps - timestamps[0]
        )

        predictions["cumulative_fuel_l"] = np.r_[
            0,
            np.cumsum(interval_fuel)
        ]

        return {
            "model": self.model_name,
            "observations": len(predictions),
            "duration_s": duration,
            "distance_km": distance_km,
            "estimated_fuel_l": total_fuel,
            "estimated_co2_kg":
                total_fuel * self.co2_factor,

            "fuel_l_per_100km":
                total_fuel / distance_km * 100
                if distance_km > 0 else None,

            "fuel_efficiency_kmpl":
                distance_km / total_fuel
                if total_fuel > 0 else None,

            "out_of_domain_observations": int(
                (~predictions["within_training_range"])
                .sum()
            ),

            "predictions": predictions
        }

    def get_model_info(self):

        return {
            "model_name": self.model_name,
            "features": self.features,
            "feature_domain": self.domain,
            "test_metrics":
                self.metadata.get("test_metrics", {}),
            "trajectory_metrics":
                self.metadata.get("trajectory_metrics", {}),
            "co2_factor": self.co2_factor,
            "sampling_interval_seconds":
                self.sampling_interval
        }


if __name__ == "__main__":

    predictor = FuelPredictor()

    result = predictor.predict_single(
        speed_kmh=45,
        acceleration_ms2=0.5,
        jerk_ms3=0.1,
        distance_to_intersection_m=150
    )

    print(json.dumps(result, indent=4))

    print("\nINFERENCE ENGINE OPERATIONAL")

