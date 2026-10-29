
import networkx as nx


def create_synthetic_map() -> nx.Graph:
    """Create a fictional road network for RESCUE-AI."""

    road_map = nx.Graph()
         
    # Fictional road intersections
    intersections = {
        "A": "North Junction",
        "B": "Central Junction",
        "C": "East Junction",
        "D": "South Junction",
        "E": "Hospital Junction",
        "F": "West Junction",
    }

    for node, name in intersections.items():
        road_map.add_node(node, name=name)
       
    # start, end, distance (meters), estimated travel time (seconds)
    roads = [
        ("A", "B", 1000, 120),
        ("A", "F", 700, 100),
        ("F", "B", 500, 60),
        ("B", "C", 800, 100),
        ("B", "D", 900, 120),
        ("C", "E", 600, 80),
        ("D", "E", 500, 70),
        ("F", "D", 1000, 130),
    ]
    
    for start, end, distance, travel_time in roads:
        road_map.add_edge(
            start,
            end,
            distance_m=distance,
            travel_time_s=travel_time,
        )

    return road_map


if __name__ == "__main__":
    graph = create_synthetic_map()

    print("RESCUE-AI Synthetic Road Map")
    print("Intersections:", graph.number_of_nodes())
    print("Roads:", graph.number_of_edges())
    print("Nodes:", list(graph.nodes))
