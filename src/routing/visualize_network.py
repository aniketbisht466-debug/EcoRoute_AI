
from pathlib import Path

import folium
import osmnx as ox

from network_manager import RoadNetworkManager


ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "assets"
    / "figures"
    / "routing"
    / "road_network.html"
)


def main():

    manager = RoadNetworkManager()

    graph = manager.load_network()

    nodes, edges = ox.graph_to_gdfs(graph)

    center = [
        float(nodes["y"].median()),
        float(nodes["x"].median())
    ]

    road_map = folium.Map(
        location=center,
        zoom_start=12,
        tiles=None,
        control_scale=True
    )

    roads = edges[["geometry"]].to_crs(
        epsg=4326
    )

    folium.GeoJson(
        roads.to_json(),
        name="Road Network",
        style_function=lambda feature: {
            "color": "#059669",
            "weight": 1.5,
            "opacity": 0.75
        }
    ).add_to(road_map)

    folium.LayerControl().add_to(road_map)

    road_map.fit_bounds([
        [float(nodes["y"].min()),
         float(nodes["x"].min())],
        [float(nodes["y"].max()),
         float(nodes["x"].max())]
    ])

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    road_map.save(str(OUTPUT))

    print("Interactive map saved:")
    print(OUTPUT)

    print("\nNETWORK VISUALIZATION COMPLETED")


if __name__ == "__main__":
    main()

