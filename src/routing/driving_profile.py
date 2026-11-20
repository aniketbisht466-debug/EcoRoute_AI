
"""
EcoRoute AI
Advanced Driving Profile Simulator
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from routing.route_optimizer import EcoRouteOptimizer


REPORT_DIR = ROOT / "reports" / "routing"
FIGURE_DIR = ROOT / "assets" / "figures" / "routing"

for directory in [REPORT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


OBJECTIVES = {
    "shortest": "length",
    "fastest": "travel_time",
    "eco": "eco_cost"
}


class DrivingProfileSimulator:

    def __init__(
        self,
        optimizer,
        acceleration=0.7,
        deceleration=1.2,
        signal_wait=15.0
    ):

        self.optimizer = optimizer
        self.graph = optimizer.graph
        self.predictor = optimizer.predictor

        self.acceleration = float(acceleration)
        self.deceleration = float(deceleration)
        self.signal_wait = float(signal_wait)

        if (
            not np.isfinite([
                self.acceleration,
                self.deceleration,
                self.signal_wait
            ]).all()
            or self.acceleration <= 0
            or self.deceleration <= 0
            or self.signal_wait < 0
        ):
            raise ValueError("Invalid simulation parameters.")

        self.dt = self.predictor.sampling_interval

    def is_signal(self, node):

        highway = self.graph.nodes[node].get(
            "highway", ""
        )

        return "traffic_signals" in str(highway)

    def is_junction(self, node):

        street_count = self.graph.nodes[node].get(
            "street_count", 0
        )

        try:
            return (
                int(street_count) >= 3
                or self.is_signal(node)
            )

        except (ValueError, TypeError):
            return self.is_signal(node)

    # ======================================
    # BUILD SPATIAL ROAD PROFILE
    # ======================================

    def build_spatial_profile(self, nodes, weight):

        edges = self.optimizer.get_route_edges(
            nodes, weight
        )

        positions = [0.0]
        speed_limits = [0.0]

        junctions = []
        signal_positions = []

        distance = 0.0

        for u, v, key, edge in edges:

            length = float(edge["length"])

            speed = float(edge["speed_kph"]) / 3.6

            if not np.isfinite([length, speed]).all():
                raise ValueError("Invalid road attributes.")

            if length <= 0:
                continue

            if speed <= 0:
                raise ValueError("Invalid road speed.")

            # Apply the lower speed at road transitions.

            speed_limits[-1] = min(
                speed_limits[-1], speed
            )

            segments = max(
                2, int(np.ceil(length / 10))
            )

            points = np.linspace(
                distance,
                distance + length,
                segments + 1
            )[1:]

            positions.extend(points.tolist())

            speed_limits.extend(
                [speed] * segments
            )

            distance += length

            if self.is_junction(v):
                junctions.append(distance)

            if self.is_signal(v):

                speed_limits[-1] = 0.0

                signal_positions.append(distance)

        if distance <= 0:
            raise ValueError("Route has no usable length.")

        speed_limits[0] = 0.0
        speed_limits[-1] = 0.0

        # Destination acts as a terminal-distance proxy.

        junctions.append(distance)

        return (
            np.asarray(positions),
            np.asarray(speed_limits),
            np.unique(junctions),
            np.unique(signal_positions)
        )

    # ======================================
    # ACCELERATION AND BRAKING ENVELOPE
    # ======================================

    def apply_vehicle_dynamics(
        self, positions, limits
    ):

        velocity = limits.copy()

        distances = np.diff(positions)

        for i in range(1, len(velocity)):

            reachable_speed = np.sqrt(
                velocity[i - 1] ** 2
                + 2 * self.acceleration
                * distances[i - 1]
            )

            velocity[i] = min(
                velocity[i],
                reachable_speed
            )

        for i in range(len(velocity) - 2, -1, -1):

            braking_speed = np.sqrt(
                velocity[i + 1] ** 2
                + 2 * self.deceleration
                * distances[i]
            )

            velocity[i] = min(
                velocity[i],
                braking_speed
            )

        return velocity

    # ======================================
    # CONVERT DISTANCE TO TIME
    # ======================================

    def build_timeline(
        self,
        positions,
        velocity,
        signal_positions
    ):

        times = [0.0]
        distances = [float(positions[0])]
        speeds = [float(velocity[0])]

        current_time = 0.0

        signals = set(signal_positions.tolist())

        for i in range(1, len(positions)):

            ds = positions[i] - positions[i - 1]

            speed_sum = (
                velocity[i - 1] + velocity[i]
            )

            if speed_sum <= 0:
                raise ValueError(
                    "Vehicle cannot traverse segment."
                )

            duration = 2 * ds / speed_sum

            current_time += duration

            times.append(current_time)
            distances.append(float(positions[i]))
            speeds.append(float(velocity[i]))

            # Fixed simulated signal delay.
            # No live signal timing is available.

            if (
                positions[i] in signals
                and i < len(positions) - 1
                and self.signal_wait > 0
            ):

                current_time += self.signal_wait

                times.append(current_time)
                distances.append(float(positions[i]))
                speeds.append(0.0)

        return (
            np.asarray(times),
            np.asarray(distances),
            np.asarray(speeds)
        )

    # ======================================
    # SECOND-BY-SECOND TELEMETRY
    # ======================================

    def generate_telemetry(
        self, times, positions, speeds, junctions
    ):

        duration = float(times[-1])

        count = int(np.floor(duration / self.dt))

        if count > 100000:
            raise ValueError(
                "Simulation exceeds observation limit."
            )

        sample_times = (
            np.arange(count + 1) * self.dt
        )

        index = np.searchsorted(
            times,
            sample_times,
            side="right"
        ) - 1

        index = np.clip(
            index, 0, len(times) - 2
        )

        segment_duration = (
            times[index + 1] - times[index]
        )

        segment_acceleration = (
            speeds[index + 1] - speeds[index]
        ) / segment_duration

        local_time = sample_times - times[index]

        sample_speed = (
            speeds[index]
            + segment_acceleration * local_time
        )

        sample_speed = np.maximum(
            sample_speed, 0.0
        )

        sample_distance = (
            positions[index]
            + speeds[index] * local_time
            + 0.5 * segment_acceleration
            * local_time ** 2
        )

        sample_distance = np.clip(
            sample_distance,
            0,
            positions[-1]
        )

        # Assume stationary conditions before departure.

        acceleration = np.diff(
            sample_speed,
            prepend=0.0
        ) / self.dt

        jerk = np.diff(
            acceleration,
            prepend=0.0
        ) / self.dt

        junction_index = np.searchsorted(
            junctions,
            sample_distance,
            side="left"
        )

        junction_index = np.clip(
            junction_index,
            0,
            len(junctions) - 1
        )

        intersection_distance = np.maximum(
            junctions[junction_index]
            - sample_distance,
            0
        )

        return pd.DataFrame({

            "elapsed_time_s": sample_times,

            "cumulative_distance_m":
                sample_distance,

            "speed_kmh":
                sample_speed * 3.6,

            "acceleration_ms2":
                acceleration,

            "jerk_ms3":
                jerk,

            "distance_to_intersection_m":
                intersection_distance
        })

    # ======================================
    # MACHINE LEARNING FUEL ESTIMATION
    # ======================================

    def simulate(self, nodes, weight):

        positions, limits, junctions, signals = (
            self.build_spatial_profile(
                nodes, weight
            )
        )

        velocity = self.apply_vehicle_dynamics(
            positions,
            limits
        )

        times, distances, speeds = (
            self.build_timeline(
                positions,
                velocity,
                signals
            )
        )

        telemetry = self.generate_telemetry(
            times,
            distances,
            speeds,
            junctions
        )

        predictions = self.predictor.predict(
            telemetry
        )

        rates = predictions[
            "fuel_rate_lps"
        ].to_numpy()

        supported = predictions[
            "within_training_range"
        ].to_numpy(dtype=bool)

        telemetry["fuel_rate_lps"] = np.where(
            supported,
            rates,
            np.nan
        )

        telemetry["fuel_rate_lph"] = (
            telemetry["fuel_rate_lps"] * 3600
        )

        telemetry["within_training_range"] = (
            supported
        )

        outside = int((~supported).sum())

        duration = float(times[-1])

        tail = max(
            0.0,
            duration
            - telemetry["elapsed_time_s"].iloc[-1]
        )

        total_fuel = None

        # Withhold complete journey estimates if any
        # simulated observation is outside training ranges.

        if outside == 0 and len(rates) >= 2:

            total_fuel = float(
                np.trapezoid(
                    rates,
                    dx=self.dt
                )
                + rates[-1] * tail
            )

        distance_km = float(positions[-1] / 1000)

        co2 = (
            total_fuel * self.predictor.co2_factor
            if total_fuel is not None
            else None
        )

        summary = {

            "distance_km": distance_km,

            "simulated_duration_minutes":
                duration / 60,

            "average_speed_kmh":
                distance_km / (duration / 3600)
                if duration > 0 else None,

            "maximum_speed_kmh":
                float(velocity.max() * 3.6),

            "mapped_signal_stops":
                int(np.sum(signals < positions[-1])),

            "observations":
                len(telemetry),

            "out_of_domain_observations":
                outside,

            "domain_coverage_percent":
                float(supported.mean() * 100),

            "estimated_fuel_l":
                total_fuel,

            "estimated_co2_kg":
                co2,

            "fuel_l_per_100km": (
                total_fuel / distance_km * 100
                if total_fuel is not None
                and distance_km > 0
                else None
            ),

            "tail_duration_s":
                float(tail),

            "estimation_mode":
                "dynamic_driving_simulation"
        }

        return telemetry, summary


# ==========================================
# VISUALIZATION
# ==========================================

def plot_profile(data, route_name):

    plt.style.use("dark_background")

    fig, axes = plt.subplots(
        4, 1,
        figsize=(15, 12),
        sharex=True
    )

    fig.patch.set_facecolor("#080F19")

    fields = [

        ("speed_kmh",
         "Speed (km/h)", "#38BDF8"),

        ("acceleration_ms2",
         "Acceleration (m/s²)", "#A78BFA"),

        ("jerk_ms3",
         "Jerk (m/s³)", "#F59E0B"),

        ("fuel_rate_lph",
         "Fuel rate (L/h)", "#10B981")
    ]

    for ax, (field, label, color) in zip(
        axes, fields
    ):

        ax.set_facecolor("#111C2E")

        ax.plot(
            data["elapsed_time_s"],
            data[field],
            color=color,
            linewidth=1.5
        )

        ax.set_ylabel(label)

        ax.grid(alpha=0.12)

    outside = data[
        ~data["within_training_range"]
    ]

    if not outside.empty:

        axes[-1].scatter(
            outside["elapsed_time_s"],
            np.zeros(len(outside)),
            marker="x",
            color="#F43F5E",
            s=10,
            label="Unsupported sample (marker only)"
        )

        axes[-1].legend()

    axes[-1].set_xlabel("Elapsed time (seconds)")

    fig.suptitle(
        f"EcoRoute AI | {route_name.title()} "
        "Dynamic Driving Simulation",
        fontsize=20
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        f"{route_name}_driving_profile.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# EXECUTION
# ==========================================

def main():

    optimizer = EcoRouteOptimizer()

    result = optimizer.optimize(
        origin=(28.5708, 77.3261),
        destination=(28.5355, 77.3910)
    )

    simulator = DrivingProfileSimulator(
        optimizer
    )

    records = []

    for name, route in result["routes"].items():

        print("\nSimulating:", name)

        telemetry, summary = simulator.simulate(
            route["nodes"],
            OBJECTIVES[name]
        )

        telemetry.to_csv(
            REPORT_DIR /
            f"{name}_driving_profile.csv",
            index=False
        )

        plot_profile(telemetry, name)

        records.append({
            "route_type": name,
            **summary
        })

    comparison = pd.DataFrame(records)

    comparison.to_csv(
        REPORT_DIR /
        "dynamic_route_comparison.csv",
        index=False
    )

    (REPORT_DIR /
     "dynamic_route_comparison.json").write_text(

        json.dumps(
            records,
            indent=4,
            allow_nan=False
        ),

        encoding="utf-8"
    )

    print("\nDYNAMIC ROUTE COMPARISON")

    print(
        comparison.round(4).to_string(
            index=False
        )
    )

    print("\nDRIVING PROFILE SIMULATION COMPLETED!")


if __name__ == "__main__":
    main()
