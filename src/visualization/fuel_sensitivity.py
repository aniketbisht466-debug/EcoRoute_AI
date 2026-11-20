
"""
EcoRoute AI
Fuel Sensitivity Analysis
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Locate the models directory
ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src" / "models"))

from fuel_engine import simulate_trip


# Baseline vehicle and road conditions
baseline = {
    "vehicle_mass_kg": 6000,
    "payload_kg": 4000,
    "speed_kmh": 60,
    "gradient_percent": 2,
    "distance_km": 10,
    "idle_minutes": 15,
    "number_of_stops": 5
}


experiments = [
    (
        "payload_kg",
        np.linspace(0, 8000, 50),
        "Payload Weight (kg)"
    ),
    (
        "speed_kmh",
        np.linspace(20, 90, 50),
        "Moving Speed (km/h)"
    ),
    (
        "gradient_percent",
        np.linspace(-5, 10, 50),
        "Road Gradient (%)"
    ),
    (
        "idle_minutes",
        np.linspace(0, 60, 50),
        "Idle Duration (minutes)"
    )
]


# Create output directory
output_dir = ROOT / "reports" / "figures"

output_dir.mkdir(parents=True, exist_ok=True)


# Generate scientific plots
for parameter, values, xlabel in experiments:

    fuel_results = []

    for value in values:

        inputs = baseline.copy()

        inputs[parameter] = float(value)

        result = simulate_trip(**inputs)

        fuel_results.append(result["total_fuel_l"])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        values,
        fuel_results,
        color="#168567",
        linewidth=2.5
    )

    ax.fill_between(
        values,
        fuel_results,
        alpha=0.10,
        color="#168567"
    )

    ax.set_title(
        f"Effect of {xlabel} on Fuel Consumption",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(xlabel)

    ax.set_ylabel("Total Fuel Consumption (L)")

    ax.grid(alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    file_path = output_dir / f"fuel_vs_{parameter}.png"

    fig.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {file_path.name}")


print("\nAll sensitivity plots generated successfully!")
