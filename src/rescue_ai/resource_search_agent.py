
from src.rescue_ai.route_agent import RouteAgent


# Fictional ambulance records for testing.
# Locations must match nodes in synthetic_map.py.
AMBULANCES = [
    {
        "ambulance_id": "AMB-101",
        "location": "A",
        "status": "available",
        "capabilities": ["ALS", "oxygen", "monitor"],
    },
    {
        "ambulance_id": "AMB-102",
        "location": "B",
        "status": "available",
        "capabilities": ["BLS", "oxygen"],
    },
    {
        "ambulance_id": "AMB-103",
        "location": "F",
        "status": "busy",
        "capabilities": ["ALS", "oxygen", "monitor"],
    },
    {
        "ambulance_id": "AMB-104",
        "location": "D",
        "status": "available",
        "capabilities": ["ALS", "oxygen", "monitor"],
    },
]


HOSPITALS = [
    {
        "hospital_id": "HOSP-101",
        "name": "Central Trauma Hospital",
        "location": "E",
        "status": "open",
        "available_beds": 5,
        "capabilities": ["EMERGENCY", "TRAUMA"],
    },
    {
        "hospital_id": "HOSP-102",
        "name": "Community Hospital",
        "location": "C",
        "status": "open",
        "available_beds": 3,
        "capabilities": ["EMERGENCY"],
    },
    {
        "hospital_id": "HOSP-103",
        "name": "South Medical Center",
        "location": "D",
        "status": "open",
        "available_beds": 0,
        "capabilities": ["EMERGENCY", "TRAUMA"],
    },
]



class ResourceSearchAgent:
    """Find and rank eligible ambulances using route travel times."""
    
    def __init__(self):
        self.route_agent = RouteAgent()
     
    def search_ambulances(
        self,
        incident_location: str,
        required_capabilities: list[str],
    ) -> list[dict]:
        """Return eligible ambulances ranked by estimated travel time."""
                    
        if incident_location not in self.route_agent.road_map:
            raise ValueError(
                f"Unknown incident location: {incident_location}"
            )
    
        candidates = []

        for ambulance in AMBULANCES:
            # 1. Ignore busy or unavailable ambulances.
            if ambulance["status"] != "available":
                continue

            # 2. Check whether all required capabilities are available.
            if not set(required_capabilities).issubset(
                set(ambulance["capabilities"])
            ):
                continue

            # 3. Calculate the route from the ambulance to the incident.
            try:
                route = self.route_agent.calculate_route(
                    ambulance["location"],
                    incident_location,
                )
            except ValueError:
                continue

            # 4. Build the candidate response.
            candidates.append({
                "ambulance_id": ambulance["ambulance_id"],
                "location": ambulance["location"],
                "capabilities": ambulance["capabilities"],
                "distance_m": route["distance_m"],
                "eta_minutes": route["eta_minutes"],
                "route": route["route"],
            })

        # 5. Rank the fastest candidates first.
        candidates.sort(key=lambda item: item["eta_minutes"])

        return candidates

    
    def search_hospitals(
        self,
        incident_location: str,
        required_capabilities: list[str],
    ) -> list[dict]:
        """Find and rank suitable hospitals by estimated travel time."""

        if incident_location not in self.route_agent.road_map:
            raise ValueError(
                f"Unknown incident location: {incident_location}"
            )

        candidates = []

        for hospital in HOSPITALS:
            # Exclude closed hospitals and hospitals with no reported beds.
            if hospital["status"] != "open":
                continue

            if hospital["available_beds"] <= 0:
                continue

            # Ensure the hospital has all requested capabilities.
            if not set(required_capabilities).issubset(
                set(hospital["capabilities"])
            ):
                continue

            try:
                route = self.route_agent.calculate_route(
                    incident_location,
                    hospital["location"],
                )
            except ValueError:
                continue

            candidates.append({
                "hospital_id": hospital["hospital_id"],
                "name": hospital["name"],
                "location": hospital["location"],
                "available_beds": hospital["available_beds"],
                "capabilities": hospital["capabilities"],
                "distance_m": route["distance_m"],
                "eta_minutes": route["eta_minutes"],
                "route": route["route"],
            })

        candidates.sort(key=lambda item: item["eta_minutes"])
        return candidates




if __name__ == "__main__":
    agent = ResourceSearchAgent()

    results = agent.search_ambulances(
        incident_location="E",
        required_capabilities=["ALS"],
    )

    print("RESCUE-AI Resource Search Agent")
    print("-" * 35)

    if not results:
        print("No eligible ambulances found.")
    else:
        for rank, ambulance in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print("Ambulance:", ambulance["ambulance_id"])
            print("Location:", ambulance["location"])
            print("ETA:", ambulance["eta_minutes"], "minutes")
            print("Distance:", ambulance["distance_m"], "meters")
            print("Route:", " -> ".join(ambulance["route"]))

    
    
    print("\nRESCUE-AI Hospital Search")
    print("-" * 35)

    hospitals = agent.search_hospitals(
        incident_location="A",
        required_capabilities=["TRAUMA"],
    )

    for rank, hospital in enumerate(hospitals, start=1):
        print(f"\nRank: {rank}")
        print("Hospital:", hospital["name"])
        print("Location:", hospital["location"])
        print("Available beds:", hospital["available_beds"])
        print("ETA:", hospital["eta_minutes"], "minutes")
        print("Route:", " -> ".join(hospital["route"]))
        
