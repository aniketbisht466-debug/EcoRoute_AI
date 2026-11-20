
"""
EcoRoute AI
Geospatial Road Network Manager
"""

from pathlib import Path
import json
import hashlib
import logging
import time

import osmnx as ox
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "routing_config.json"

MAP_DIR = ROOT / "data" / "maps"

CACHE_DIR = ROOT / "data" / "cache"

REPORT_DIR = ROOT / "reports" / "routing"

for directory in [MAP_DIR, CACHE_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("EcoRoute Routing")


class RoadNetworkManager:

    def __init__(self):

        self.config = json.loads(
            CONFIG_PATH.read_text(encoding="utf-8")
        )

        ox.settings.use_cache = self.config[
            "cache_enabled"
        ]

        ox.settings.cache_folder = str(CACHE_DIR)

        ox.settings.requests_timeout = self.config[
            "request_timeout_seconds"
        ]

        ox.settings.log_console = False

        self.graph = None
        self.city = None

    def get_cache_path(self, city):

        key = (
            city.lower().strip()
            + self.config["network_type"]
        )

        identifier = hashlib.sha256(
            key.encode()
        ).hexdigest()[:12]

        return MAP_DIR / f"road_network_{identifier}.graphml"

    def load_network(self, city=None):

        self.city = city or self.config["default_city"]

        cache_path = self.get_cache_path(self.city)

        if cache_path.exists():

            logger.info("Loading cached road network")

            self.graph = ox.load_graphml(cache_path)

        else:

            logger.info(
                "Downloading road network: %s",
                self.city
            )

            self.graph = ox.graph_from_place(
                self.city,
                network_type=self.config["network_type"],
                simplify=True,
                retain_all=False
            )

            ox.save_graphml(
                self.graph,
                filepath=cache_path
            )

        self.prepare_network()

        logger.info(
            "Network ready | Nodes: %s | Edges: %s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges()
        )

        return self.graph

    def prepare_network(self):

        if self.graph is None:
            raise RuntimeError("Road network not loaded.")

        self.graph = ox.routing.add_edge_speeds(
            self.graph,
            hwy_speeds=self.config[
                "highway_speeds_kmh"
            ],
            fallback=self.config["default_speed_kmh"]
        )

        self.graph = ox.routing.add_edge_travel_times(
            self.graph
        )

        for u, v, key, edge in self.graph.edges(
            keys=True,
            data=True
        ):

            length = float(edge["length"])

            speed = float(edge["speed_kph"])

            travel_time = float(edge["travel_time"])

            if not np.isfinite(
                [length, speed, travel_time]
            ).all():

                raise ValueError(
                    f"Invalid edge: {u}, {v}, {key}"
                )

            if length < 0 or speed <= 0:

                raise ValueError(
                    f"Invalid road attributes: {u}, {v}"
                )

    def nearest_node(self, latitude, longitude):

        if self.graph is None:

            raise RuntimeError(
                "Load a road network first."
            )

        return ox.distance.nearest_nodes(
            self.graph,
            X=float(longitude),
            Y=float(latitude)
        )

    def network_statistics(self):

        graph = self.graph

        if graph is None:

            raise RuntimeError(
                "Road network not loaded."
            )

        lengths = [
            float(edge["length"])
            for _, _, edge in graph.edges(data=True)
        ]

        speeds = [
            float(edge["speed_kph"])
            for _, _, edge in graph.edges(data=True)
        ]

        statistics = {
            "city": self.city,
            "nodes": graph.number_of_nodes(),
            "directed_edges": graph.number_of_edges(),
            "directed_edge_length_km":
                round(sum(lengths) / 1000, 3),
            "mean_edge_speed_kmh":
                round(float(np.mean(speeds)), 2),
            "strongly_connected":
                nx.is_strongly_connected(graph),
            "network_type":
                self.config["network_type"]
        }

        (REPORT_DIR / "network_statistics.json").write_text(
            json.dumps(statistics, indent=4),
            encoding="utf-8"
        )

        return statistics


def main():

    start = time.perf_counter()

    manager = RoadNetworkManager()

    manager.load_network()

    statistics = manager.network_statistics()

    print("\nROAD NETWORK STATISTICS")

    print(json.dumps(statistics, indent=4))

    print(
        "\nExecution time:",
        round(time.perf_counter() - start, 2),
        "seconds"
    )

    print("\nROAD NETWORK ENGINE OPERATIONAL")


if __name__ == "__main__":
    main()
