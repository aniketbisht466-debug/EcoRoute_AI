
"""
EcoRoute AI
Chapter 01: Vehicle Physics
Purpose: Calculate forces acting on a moving truck.
"""

import math


# Physical constants and initial assumptions
GRAVITY = 9.81
AIR_DENSITY = 1.225
ROLLING_COEFFICIENT = 0.008
DRAG_COEFFICIENT = 0.7
FRONTAL_AREA = 8.0


def rolling_resistance(mass_kg, gradient_percent):

    theta = math.atan(gradient_percent / 100)

    force = (
        ROLLING_COEFFICIENT
        * mass_kg
        * GRAVITY
        * math.cos(theta)
    )

    return force


def aerodynamic_drag(speed_kmh):

    speed_ms = speed_kmh / 3.6

    force = (
        0.5
        * AIR_DENSITY
        * DRAG_COEFFICIENT
        * FRONTAL_AREA
        * speed_ms ** 2
    )

    return force


def gradient_resistance(mass_kg, gradient_percent):

    theta = math.atan(gradient_percent / 100)

    force = mass_kg * GRAVITY * math.sin(theta)

    return force


def acceleration_force(mass_kg, acceleration_ms2):

    return mass_kg * acceleration_ms2


if __name__ == "__main__":

    vehicle_mass = 6000
    payload_mass = 4000

    total_mass = vehicle_mass + payload_mass

    speed = 60
    gradient = 2
    acceleration = 0

    rolling = rolling_resistance(total_mass, gradient)

    drag = aerodynamic_drag(speed)

    slope = gradient_resistance(total_mass, gradient)

    acceleration_f = acceleration_force(
        total_mass, acceleration
    )

    total_force = rolling + drag + slope + acceleration_f

    print("=" * 50)
    print("ECOROUTE AI - VEHICLE PHYSICS ENGINE")
    print("=" * 50)

    print(f"Vehicle mass: {vehicle_mass} kg")
    print(f"Payload mass: {payload_mass} kg")
    print(f"Total mass: {total_mass} kg")
    print(f"Speed: {speed} km/h")
    print(f"Road gradient: {gradient}%")

    print("-" * 50)

    print(f"Rolling resistance: {rolling:.2f} N")
    print(f"Aerodynamic drag: {drag:.2f} N")
    print(f"Gradient resistance: {slope:.2f} N")
    print(f"Acceleration force: {acceleration_f:.2f} N")

    print("-" * 50)

    print(f"Total tractive force: {total_force:.2f} N")
    print("=" * 50)
