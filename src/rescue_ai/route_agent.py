
import networkx as nx

from src.rescue_ai.synthetic_map import create_synthetic_map


class RouteAgent:
    """Find the fastest route through the synthetic road network."""

    def __init__(self):
        self.road_map = create_synthetic_map()

    def calculate_route(self, origin: str, destination: str) -> dict:
        """
        Calculate the fastest route.

        Args:
            origin: Starting intersection ID, such as A.
            destination: Destination intersection ID, such as E.
        """

        if origin not in self.road_map:
            raise ValueError(f"Unknown origin intersection: {origin}")

        if destination not in self.road_map:
            raise ValueError(f"Unknown destination intersection: {destination}")

        try:
            path = nx.shortest_path(
                self.road_map,
                source=origin,
                target=destination,
                weight="travel_time_s",
                method="dijkstra",
            )
        except nx.NetworkXNoPath as exc:
            raise ValueError("No road route connects these intersections.") from exc

        total_distance = 0
        total_time = 0

        for start, end in zip(path, path[1:]):
            road = self.road_map[start][end]
            total_distance += road["distance_m"]
            total_time += road["travel_time_s"]

        return {
            "origin": origin,
            "destination": destination,
            "route": path,
            "distance_m": total_distance,
            "travel_time_seconds": total_time,
            "eta_minutes": round(total_time / 60, 2),
            "algorithm": "Dijkstra",
            "map_type": "synthetic",
        }


if __name__ == "__main__":
    agent = RouteAgent()

    result = agent.calculate_route("A", "E")

    print("RESCUE-AI Route Agent")
    print("-" * 30)
    print("Origin:", result["origin"])
    print("Destination:", result["destination"])
    print("Fastest route:", " -> ".join(result["route"]))
    print("Distance:", result["distance_m"], "meters")
    print("Estimated travel time:", result["eta_minutes"], "minutes")
