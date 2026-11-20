
"""
EcoRoute AI
Fuel-Aware Route Optimization Engine
Version 2.0
"""

from pathlib import Path
import sys
import json
import logging
import time

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import folium


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from routing.network_manager import RoadNetworkManager
from inference.fuel_predictor import FuelPredictor


REPORT_DIR = ROOT / "reports" / "routing"
FIGURE_DIR = ROOT / "assets" / "figures" / "routing"

for directory in [REPORT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("EcoRoute Optimizer")


ROUTE_COLORS = {
    "shortest": "#38BDF8",
    "fastest": "#F59E0B",
    "eco": "#10B981"
}

ROUTE_LABELS = {
    "shortest": "Shortest Route",
    "fastest": "Fastest Route",
    "eco": "Eco Route"
}


# ==========================================
# ROUTE OPTIMIZATION ENGINE
# ==========================================

class EcoRouteOptimizer:

    def __init__(self, city=None):

        self.manager = RoadNetworkManager()

        self.graph = self.manager.load_network(city)

        self.predictor = FuelPredictor()

        self.config = self.manager.config

        self.co2_factor = self.predictor.co2_factor

        self.costs_ready = False


    # ======================================
    # PREPARE ML FUEL COSTS
    # ======================================

    def prepare_fuel_costs(self):

        logger.info("Preparing ML-powered fuel costs")

        records = []
        edge_keys = []
        travel_times = []

        intersection_distance = self.predictor.domain[
            "distance_to_intersection_m"
        ]["median"]

        for u, v, key, edge in self.graph.edges(
            keys=True,
            data=True
        ):

            length = float(edge["length"])
            speed = float(edge["speed_kph"])
            duration = float(edge["travel_time"])

            if not np.isfinite(
                [length, speed, duration]
            ).all():

                raise ValueError(
                    f"Invalid edge: {u}, {v}, {key}"
                )

            if length < 0 or speed <= 0 or duration < 0:

                raise ValueError(
                    f"Invalid road attributes: {u}, {v}, {key}"
                )

            records.append({
                "speed_kmh": speed,
                "acceleration_ms2": 0.0,
                "jerk_ms3": 0.0,
                "distance_to_intersection_m":
                    intersection_distance
            })

            edge_keys.append((u, v, key))

            travel_times.append(duration)

        if not records:

            raise RuntimeError(
                "Road network contains no usable edges."
            )

        telemetry = pd.DataFrame(records)

        predictions = self.predictor.predict(
            telemetry
        )

        fuel_rates = predictions[
            "fuel_rate_lps"
        ].to_numpy()

        in_domain = predictions[
            "within_training_range"
        ].to_numpy()

        fuel_costs = (
            fuel_rates
            * np.asarray(travel_times)
        )

        if not np.isfinite(fuel_costs).all():

            raise ValueError(
                "Invalid fuel predictions."
            )

        outside_count = 0

        for i, (u, v, key) in enumerate(edge_keys):

            edge = self.graph[u][v][key]

            fuel = max(float(fuel_costs[i]), 0.0)

            edge["estimated_fuel_l"] = fuel

            edge["estimated_co2_kg"] = (
                fuel * self.co2_factor
            )

            edge["within_model_domain"] = bool(
                in_domain[i]
            )

            edge["eco_cost"] = max(fuel, 1e-9)

            if not in_domain[i]:
                outside_count += 1

        self.costs_ready = True

        logger.info(
            "Fuel costs assigned to %s edges",
            len(edge_keys)
        )

        logger.info(
            "Out-of-domain edges: %s",
            outside_count
        )

        return {
            "total_edges": len(edge_keys),
            "out_of_domain_edges": outside_count
        }


    # ======================================
    # LOCATION VALIDATION
    # ======================================

    def resolve_location(self, coordinates):

        latitude, longitude = map(
            float,
            coordinates
        )

        if not np.isfinite(
            [latitude, longitude]
        ).all():

            raise ValueError(
                "Coordinates must be finite."
            )

        if not (
            -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):

            raise ValueError(
                "Invalid geographical coordinates."
            )

        node, distance = ox.distance.nearest_nodes(
            self.graph,
            X=longitude,
            Y=latitude,
            return_dist=True
        )

        if distance > 1000:

            raise ValueError(
                f"Location is {distance:.0f} metres "
                "from the downloaded road network."
            )

        return node, float(distance)


    # ======================================
    # CALCULATE ROUTE
    # ======================================

    def find_route(
        self,
        origin,
        destination,
        weight
    ):

        graph = self.graph

        # Exclude unsupported ML edges from eco-routing.

        if weight == "eco_cost":

            graph = nx.subgraph_view(
                self.graph,
                filter_edge=lambda u, v, key: (
                    self.graph[u][v][key]
                    ["within_model_domain"]
                )
            )

        try:

            return nx.shortest_path(
                graph,
                source=origin,
                target=destination,
                weight=weight,
                method="dijkstra"
            )

        except (
            nx.NetworkXNoPath,
            nx.NodeNotFound
        ):

            return None


    # ======================================
    # EXTRACT ROAD SEGMENTS
    # ======================================

    def get_route_edges(self, route, weight):

        selected_edges = []

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            options = self.graph.get_edge_data(
                u, v
            )

            if not options:

                raise RuntimeError(
                    f"Missing road segment: {u}, {v}"
                )

            if weight == "eco_cost":

                options = {
                    key: edge
                    for key, edge in options.items()
                    if edge["within_model_domain"]
                }

            if not options:

                raise RuntimeError(
                    "No eligible road segment."
                )

            key, edge = min(
                options.items(),
                key=lambda item: item[1][weight]
            )

            selected_edges.append(
                (u, v, key, edge)
            )

        return selected_edges


    # ======================================
    # ROUTE METRICS
    # ======================================

    def calculate_metrics(self, route, weight):

        edges = self.get_route_edges(
            route,
            weight
        )

        distance_m = sum(
            float(edge["length"])
            for _, _, _, edge in edges
        )

        travel_time = sum(
            float(edge["travel_time"])
            for _, _, _, edge in edges
        )

        estimated_fuel = sum(
            float(edge["estimated_fuel_l"])
            for _, _, _, edge in edges
        )

        out_of_domain = sum(
            not edge["within_model_domain"]
            for _, _, _, edge in edges
        )

        distance_km = distance_m / 1000

        valid_fuel = (
            float(estimated_fuel)
            if out_of_domain == 0
            else None
        )

        return {

            "distance_km":
                float(distance_km),

            "duration_minutes":
                float(travel_time / 60),

            "estimated_fuel_l":
                valid_fuel,

            "estimated_co2_kg": (
                valid_fuel * self.co2_factor
                if valid_fuel is not None
                else None
            ),

            "fuel_l_per_100km": (
                valid_fuel / distance_km * 100
                if valid_fuel is not None
                and distance_km > 0
                else None
            ),

            "fuel_efficiency_kmpl": (
                distance_km / valid_fuel
                if valid_fuel is not None
                and valid_fuel > 0
                else None
            ),

            "average_speed_kmh": (
                distance_km / (travel_time / 3600)
                if travel_time > 0
                else None
            ),

            "road_segments":
                len(edges),

            "out_of_domain_edges":
                int(out_of_domain)
        }


    # ======================================
    # ROAD GEOMETRY
    # ======================================

    def route_geometry(self, route, weight):

        coordinates = []

        for u, v, key, edge in self.get_route_edges(
            route,
            weight
        ):

            geometry = edge.get("geometry")

            if (
                geometry is not None
                and not geometry.is_empty
                and geometry.geom_type == "LineString"
            ):

                points = list(geometry.coords)

                start = self.graph.nodes[u]

                first_distance = (
                    (points[0][0] - start["x"]) ** 2
                    + (points[0][1] - start["y"]) ** 2
                )

                last_distance = (
                    (points[-1][0] - start["x"]) ** 2
                    + (points[-1][1] - start["y"]) ** 2
                )

                if last_distance < first_distance:

                    points.reverse()

                coordinates.extend([
                    [float(point[1]), float(point[0])]
                    for point in points
                ])

            else:

                coordinates.extend([
                    [
                        float(self.graph.nodes[u]["y"]),
                        float(self.graph.nodes[u]["x"])
                    ],
                    [
                        float(self.graph.nodes[v]["y"]),
                        float(self.graph.nodes[v]["x"])
                    ]
                ])

        return coordinates


    # ======================================
    # MAIN OPTIMIZATION
    # ======================================

    def optimize(self, origin, destination):

        if not self.costs_ready:

            self.prepare_fuel_costs()

        origin_node, origin_snap = (
            self.resolve_location(origin)
        )

        destination_node, destination_snap = (
            self.resolve_location(destination)
        )

        if origin_node == destination_node:

            raise ValueError(
                "Origin and destination resolve "
                "to the same road node."
            )

        objectives = {
            "shortest": "length",
            "fastest": "travel_time",
            "eco": "eco_cost"
        }

        routes = {}

        for name, weight in objectives.items():

            logger.info(
                "Calculating %s route",
                name
            )

            route = self.find_route(
                origin_node,
                destination_node,
                weight
            )

            if route is None:

                logger.warning(
                    "%s route unavailable",
                    name
                )

                continue

            metrics = self.calculate_metrics(
                route,
                weight
            )

            routes[name] = {
                "name": name,

                "nodes": [
                    int(node)
                    for node in route
                ],

                "metrics": metrics,

                "geometry": self.route_geometry(
                    route,
                    weight
                )
            }

        if not routes:

            raise RuntimeError(
                "No route available."
            )

        reference = routes.get(
            "fastest",
            {}
        ).get("metrics", {})

        reference_fuel = reference.get(
            "estimated_fuel_l"
        )

        reference_distance = reference.get(
            "distance_km"
        )

        reference_duration = reference.get(
            "duration_minutes"
        )

        for route in routes.values():

            metrics = route["metrics"]

            fuel = metrics["estimated_fuel_l"]

            metrics["fuel_savings_percent"] = (

                (reference_fuel - fuel)
                / reference_fuel * 100

                if reference_fuel is not None
                and reference_fuel > 0
                and fuel is not None

                else None
            )

            metrics["co2_savings_kg"] = (

                (reference_fuel - fuel)
                * self.co2_factor

                if reference_fuel is not None
                and fuel is not None

                else None
            )

            metrics["detour_percent"] = (

                (
                    metrics["distance_km"]
                    - reference_distance
                ) / reference_distance * 100

                if reference_distance

                else None
            )

            metrics["additional_time_minutes"] = (

                metrics["duration_minutes"]
                - reference_duration

                if reference_duration is not None

                else None
            )

            detour = metrics["detour_percent"]

            metrics["within_detour_limit"] = (

                detour <= self.config[
                    "maximum_detour_percent"
                ]

                if detour is not None

                else None
            )

        eligible = [

            route["name"]

            for route in routes.values()

            if route["metrics"]["estimated_fuel_l"]
            is not None

            and route["metrics"]["within_detour_limit"]
            is True
        ]

        recommended = min(

            eligible,

            key=lambda name: (
                routes[name]["metrics"]["estimated_fuel_l"],
                routes[name]["metrics"]["duration_minutes"]
            ),

            default=None
        )

        return {

            "project": "EcoRoute AI",

            "city": self.manager.city,

            "model": self.predictor.model_name,

            "origin": list(map(float, origin)),

            "destination":
                list(map(float, destination)),

            "snapped_origin": [
                float(self.graph.nodes[origin_node]["y"]),
                float(self.graph.nodes[origin_node]["x"])
            ],

            "snapped_destination": [
                float(self.graph.nodes[destination_node]["y"]),
                float(self.graph.nodes[destination_node]["x"])
            ],

            "origin_snap_distance_m":
                origin_snap,

            "destination_snap_distance_m":
                destination_snap,

            "reference_route": "fastest",

            "recommended_route":
                recommended,

            "estimation_mode":
                "steady_speed_ml_simulation",

            "routes": routes
        }


# ==========================================
# FORMATTING HELPERS
# ==========================================

def format_metric(value, unit="", decimals=2):

    if value is None:

        return "Unavailable"

    return f"{value:.{decimals}f} {unit}".strip()


# ==========================================
# INTERACTIVE DARK MAP
# ==========================================

def create_route_map(result, graph=None):

    origin = result["origin"]

    destination = result["destination"]

    if graph is None:

        manager = RoadNetworkManager()

        graph = manager.load_network(
            result["city"]
        )

    # No external tile provider.
    # Prevents CARTO key errors and OSM 403 tiles.

    route_map = folium.Map(

        location=[
            (origin[0] + destination[0]) / 2,
            (origin[1] + destination[1]) / 2
        ],

        zoom_start=13,

        tiles=None,

        control_scale=True,

        prefer_canvas=True,

        max_zoom=19
    )


    # ======================================
    # DARK UI STYLING
    # ======================================

    route_map.get_root().header.add_child(
        folium.Element("""
        <style>

            .leaflet-container {
                background: #0B1220 !important;
                font-family: Arial, sans-serif;
            }

            .leaflet-control-layers {
                background: #111827 !important;
                color: #E2E8F0 !important;
                border: 1px solid #334155 !important;
                border-radius: 10px !important;
                padding: 12px !important;
            }

            .leaflet-control-zoom a {
                background: #111827 !important;
                color: #F1F5F9 !important;
                border-color: #334155 !important;
            }

            .leaflet-popup-content-wrapper {
                background: #111827 !important;
                color: #F1F5F9 !important;
                border-radius: 12px !important;
            }

            .leaflet-popup-tip {
                background: #111827 !important;
            }

            .leaflet-control-scale-line {
                background: #111827 !important;
                border-color: #64748B !important;
                color: #F1F5F9 !important;
            }

        </style>
        """)
    )


    # ======================================
    # CACHED ROAD NETWORK BACKGROUND
    # ======================================

    logger.info(
        "Rendering cached road network"
    )

    _, edges = ox.graph_to_gdfs(graph)

    road_geometry = edges[
        ["geometry"]
    ].to_crs(epsg=4326)

    network_layer = folium.FeatureGroup(
        name="Road Network",
        show=True
    )

    folium.GeoJson(

        road_geometry.to_json(),

        style_function=lambda feature: {

            "color": "#475569",

            "weight": 1.4,

            "opacity": 0.65
        },

        interactive=False

    ).add_to(network_layer)

    network_layer.add_to(route_map)


    # ======================================
    # OPTIMIZED ROUTE LAYERS
    # ======================================

    all_coordinates = []

    recommended = result[
        "recommended_route"
    ]

    # Draw recommended route last.

    route_names = sorted(

        result["routes"],

        key=lambda name: name == recommended
    )

    for name in route_names:

        route = result["routes"][name]

        metrics = route["metrics"]

        geometry = route["geometry"]

        color = ROUTE_COLORS[name]

        distance = format_metric(
            metrics["distance_km"],
            "km"
        )

        duration = format_metric(
            metrics["duration_minutes"],
            "min",
            1
        )

        fuel = format_metric(
            metrics["estimated_fuel_l"],
            "L",
            3
        )

        co2 = format_metric(
            metrics["estimated_co2_kg"],
            "kg",
            3
        )

        savings = format_metric(
            metrics["fuel_savings_percent"],
            "%",
            2
        )

        recommended_label = (

            "<p style='color:#10B981'>"
            "RECOMMENDED ROUTE</p>"

            if name == recommended

            else ""
        )

        popup = f"""
        <div style="
            font-family:Arial;
            min-width:220px;
            color:#F1F5F9;
            line-height:1.7;
        ">

            <h3 style="color:{color}">
                {ROUTE_LABELS[name]}
            </h3>

            {recommended_label}

            <b>Distance:</b> {distance}<br>
            <b>Estimated time:</b> {duration}<br>
            <b>Estimated fuel:</b> {fuel}<br>
            <b>Estimated CO₂:</b> {co2}<br>
            <b>Fuel savings:</b> {savings}<br>

            <hr style="border-color:#334155">

            <small style="color:#94A3B8">
                Simulation estimates only.
            </small>

        </div>
        """

        layer = folium.FeatureGroup(

            name=ROUTE_LABELS[name],

            show=True
        )

        # Outer route glow.

        folium.PolyLine(

            geometry,

            color=color,

            weight=14,

            opacity=0.12,

            interactive=False

        ).add_to(layer)

        # Main route line.

        folium.PolyLine(

            geometry,

            color=color,

            weight=6 if name == recommended else 4,

            opacity=0.95,

            tooltip=ROUTE_LABELS[name],

            dash_array=(
                "8, 8"
                if name == "shortest"
                else None
            ),

            popup=folium.Popup(
                popup,
                max_width=320
            )

        ).add_to(layer)

        layer.add_to(route_map)

        all_coordinates.extend(geometry)


    # ======================================
    # ORIGIN AND DESTINATION MARKERS
    # ======================================

    folium.Marker(

        location=origin,

        tooltip="Journey Origin",

        popup="Selected origin",

        icon=folium.Icon(
            color="green",
            icon="play"
        )

    ).add_to(route_map)

    folium.Marker(

        location=destination,

        tooltip="Journey Destination",

        popup="Selected destination",

        icon=folium.Icon(
            color="red",
            icon="flag"
        )

    ).add_to(route_map)


    # Visualize location snapping gaps.

    for selected, snapped in [

        (
            origin,
            result["snapped_origin"]
        ),

        (
            destination,
            result["snapped_destination"]
        )
    ]:

        folium.PolyLine(

            [selected, snapped],

            color="#94A3B8",

            weight=2,

            opacity=0.8,

            dash_array="4, 6",

            tooltip="Location snapping gap"

        ).add_to(route_map)


    # ======================================
    # ROUTE SUMMARY PANEL
    # ======================================

    recommended_title = (

        ROUTE_LABELS[recommended]

        if recommended is not None

        else "No eligible eco recommendation"
    )

    route_cards = ""

    for name, route in result["routes"].items():

        metrics = route["metrics"]

        color = ROUTE_COLORS[name]

        distance = format_metric(
            metrics["distance_km"],
            "km"
        )

        duration = format_metric(
            metrics["duration_minutes"],
            "min",
            1
        )

        fuel = format_metric(
            metrics["estimated_fuel_l"],
            "L",
            3
        )

        co2 = format_metric(
            metrics["estimated_co2_kg"],
            "kg",
            3
        )

        route_cards += f"""

        <div style="
            border-left:3px solid {color};
            background:#1E293B;
            margin-top:10px;
            padding:10px;
            border-radius:8px;
        ">

            <div style="
                color:{color};
                font-weight:bold;
                margin-bottom:6px;
            ">
                {ROUTE_LABELS[name]}
            </div>

            Distance: {distance}<br>
            Time: {duration}<br>
            Fuel: {fuel}<br>
            CO₂: {co2}

        </div>
        """

    panel = f"""

    <div style="
        position:fixed;
        top:20px;
        left:60px;
        width:270px;
        max-width:calc(100vw - 140px);
        max-height:78vh;
        overflow-y:auto;
        background:#111827;
        color:#F1F5F9;
        padding:18px;
        border:1px solid #334155;
        border-radius:14px;
        z-index:9999;
        font-family:Arial,sans-serif;
        font-size:12px;
        line-height:1.6;
        box-shadow:0 8px 30px rgba(0,0,0,0.35);
    ">

        <div style="
            color:#10B981;
            font-size:24px;
            font-weight:bold;
        ">
            EcoRoute AI
        </div>

        <div style="
            color:#94A3B8;
            font-size:10px;
            letter-spacing:1.5px;
        ">
            INTELLIGENT ROUTE OPTIMIZATION
        </div>

        <hr style="
            border:0;
            border-top:1px solid #334155;
        ">

        <div style="
            color:#94A3B8;
            font-size:11px;
        ">
            RECOMMENDED
        </div>

        <div style="
            color:#34D399;
            font-size:16px;
            font-weight:bold;
        ">
            {recommended_title}
        </div>

        {route_cards}

        <div style="
            color:#94A3B8;
            margin-top:12px;
            font-size:10px;
        ">
            ML estimates using simulated steady-speed
            driving conditions.
        </div>

    </div>
    """

    route_map.get_root().html.add_child(
        folium.Element(panel)
    )


    # ======================================
    # LEGEND AND ATTRIBUTION
    # ======================================

    route_map.get_root().html.add_child(
        folium.Element("""

        <div style="
            position:fixed;
            bottom:45px;
            right:20px;
            background:#111827;
            color:#F1F5F9;
            padding:14px;
            border:1px solid #334155;
            border-radius:10px;
            z-index:9999;
            font-family:Arial;
            font-size:12px;
            line-height:2;
        ">

            <b>Route Legend</b><br>

            <span style="color:#38BDF8">
                &#9632;
            </span> Shortest Route<br>

            <span style="color:#F59E0B">
                &#9632;
            </span> Fastest Route<br>

            <span style="color:#10B981">
                &#9632;
            </span> Eco Route<br>

            <span style="color:#64748B">
                &#9632;
            </span> Road Network

        </div>

        <div style="
            position:fixed;
            bottom:5px;
            right:5px;
            background:#111827;
            color:#94A3B8;
            padding:5px 10px;
            border-radius:5px;
            font-size:10px;
            z-index:9999;
        ">

            &copy;
            <a href="https://www.openstreetmap.org/copyright"
               style="color:#10B981">

                OpenStreetMap contributors

            </a>

        </div>

        """)
    )


    # ======================================
    # MAP CONTROLS
    # ======================================

    folium.LayerControl(

        collapsed=True,

        position="topright"

    ).add_to(route_map)

    all_coordinates.extend([
        origin,
        destination
    ])

    if all_coordinates:

        route_map.fit_bounds(

            all_coordinates,

            padding=(50, 50)
        )

    output = (
        FIGURE_DIR / "optimized_routes.html"
    )

    route_map.save(str(output))

    logger.info(
        "Interactive route map saved"
    )

    return output


# ==========================================
# EXPORT REPORTS
# ==========================================

def save_results(result):

    records = []

    for name, route in result["routes"].items():

        records.append({

            "route_type": name,

            "recommended": (
                name == result["recommended_route"]
            ),

            **route["metrics"]
        })

    comparison = pd.DataFrame(records)

    comparison.to_csv(

        REPORT_DIR / "route_comparison.csv",

        index=False
    )

    (REPORT_DIR / "optimized_routes.json").write_text(

        json.dumps(
            result,
            indent=4,
            allow_nan=False
        ),

        encoding="utf-8"
    )

    return comparison


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    start = time.perf_counter()

    print("\n" + "=" * 70)

    print("ECOROUTE AI")

    print("FUEL-AWARE ROUTE OPTIMIZATION ENGINE")

    print("=" * 70)

    optimizer = EcoRouteOptimizer()

    origin = (
        28.5708,
        77.3261
    )

    destination = (
        28.5355,
        77.3910
    )

    result = optimizer.optimize(
        origin,
        destination
    )

    comparison = save_results(result)

    map_path = create_route_map(
        result,
        graph=optimizer.graph
    )

    print("\nROUTE COMPARISON")

    print("-" * 70)

    display_columns = [

        "route_type",

        "distance_km",

        "duration_minutes",

        "estimated_fuel_l",

        "estimated_co2_kg",

        "fuel_savings_percent",

        "recommended"
    ]

    print(
        comparison[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nSELECTED MODEL:")

    print(result["model"])

    print("\nRECOMMENDED ROUTE:")

    print(
        result["recommended_route"]
        or "No eligible recommendation"
    )

    print("\nINTERACTIVE MAP:")

    print(map_path)

    print(
        "\nExecution time:",
        round(
            time.perf_counter() - start,
            2
        ),
        "seconds"
    )

    print("\n" + "=" * 70)

    print("ROUTE OPTIMIZATION COMPLETED!")

    print("=" * 70)


if __name__ == "__main__":

    main()
