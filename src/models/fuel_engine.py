
"""
EcoRoute AI
Chapter 02: Physics-Based Fuel Simulation Engine
Version: 0.1
"""

from vehicle_physics import (
    rolling_resistance,
    aerodynamic_drag,
    gradient_resistance
)


# Preliminary simulation assumptions
DIESEL_ENERGY_DENSITY = 35_800_000  # Joules per litre
CO2_EMISSION_FACTOR = 2.68         # kg CO2 per litre
IDLE_FUEL_RATE = 2.0               # Litres per hour
DEFAULT_EFFICIENCY = 0.35


def calculate_traction_fuel(
    mass_kg,
    speed_kmh,
    gradient_percent,
    distance_km,
    efficiency
):
    rolling = rolling_resistance(
        mass_kg, gradient_percent
    )

    drag = aerodynamic_drag(speed_kmh)

    slope = gradient_resistance(
        mass_kg, gradient_percent
    )

    total_force = max(0, rolling + drag + slope)

    distance_m = distance_km * 1000

    mechanical_energy = total_force * distance_m

    fuel_energy = mechanical_energy / efficiency

    fuel_liters = fuel_energy / DIESEL_ENERGY_DENSITY

    return fuel_liters


def calculate_idle_fuel(idle_minutes):

    idle_hours = idle_minutes / 60

    return IDLE_FUEL_RATE * idle_hours


def calculate_stop_fuel(
    mass_kg,
    speed_kmh,
    number_of_stops,
    efficiency
):
    speed_ms = speed_kmh / 3.6

    kinetic_energy = 0.5 * mass_kg * speed_ms ** 2

    total_energy = kinetic_energy * number_of_stops

    fuel_energy = total_energy / efficiency

    return fuel_energy / DIESEL_ENERGY_DENSITY


def simulate_trip(
    vehicle_mass_kg,
    payload_kg,
    speed_kmh,
    gradient_percent,
    distance_km,
    idle_minutes,
    number_of_stops,
    efficiency=DEFAULT_EFFICIENCY,
    fuel_price=90
):

    # Input validation
    if vehicle_mass_kg <= 0:
        raise ValueError("Vehicle mass must be positive.")

    if payload_kg < 0:
        raise ValueError("Payload cannot be negative.")

    if speed_kmh <= 0 or distance_km <= 0:
        raise ValueError("Speed and distance must be positive.")

    if idle_minutes < 0 or number_of_stops < 0:
        raise ValueError("Idle time and stops cannot be negative.")

    if not 0 < efficiency <= 1:
        raise ValueError("Efficiency must be between 0 and 1.")

    if fuel_price < 0:
        raise ValueError("Fuel price cannot be negative.")

    # Calculate total vehicle mass
    total_mass = vehicle_mass_kg + payload_kg

    # Calculate fuel components
    traction_fuel = calculate_traction_fuel(
        total_mass,
        speed_kmh,
        gradient_percent,
        distance_km,
        efficiency
    )

    idle_fuel = calculate_idle_fuel(idle_minutes)

    stop_fuel = calculate_stop_fuel(
        total_mass,
        speed_kmh,
        number_of_stops,
        efficiency
    )

    total_fuel = traction_fuel + idle_fuel + stop_fuel

    # Environmental and economic calculations
    co2_emissions = total_fuel * CO2_EMISSION_FACTOR

    fuel_cost = total_fuel * fuel_price

    fuel_efficiency = (
        distance_km / total_fuel
        if total_fuel > 0 else None
    )

    travel_time = (
        distance_km / speed_kmh * 60
        + idle_minutes
    )

    return {
        "total_mass_kg": total_mass,
        "distance_km": distance_km,
        "traction_fuel_l": traction_fuel,
        "idle_fuel_l": idle_fuel,
        "stop_fuel_l": stop_fuel,
        "total_fuel_l": total_fuel,
        "co2_emissions_kg": co2_emissions,
        "fuel_efficiency_kmpl": fuel_efficiency,
        "fuel_cost_inr": fuel_cost,
        "estimated_time_min": travel_time
    }


if __name__ == "__main__":

    result = simulate_trip(
        vehicle_mass_kg=6000,
        payload_kg=4000,
        speed_kmh=60,
        gradient_percent=2,
        distance_km=10,
        idle_minutes=15,
        number_of_stops=5
    )

    print("=" * 55)
    print("ECOROUTE AI - FUEL SIMULATION ENGINE")
    print("=" * 55)

    for key, value in result.items():
        if value is None:
            print(f"{key:25s}: N/A")
        else:
            print(f"{key:25s}: {value:.3f}")

    print("=" * 55)
