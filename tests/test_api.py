
from fastapi.testclient import TestClient

from src.rescue_ai.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_route_endpoint():
    response = client.post(
        "/agents/route",
        json={
            "origin": "A",
            "destination": "E",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["origin"] == "A"
    assert data["destination"] == "E"
    assert data["route"][0] == "A"
    assert data["route"][-1] == "E"
    assert data["distance_m"] > 0
    assert data["eta_minutes"] > 0


def test_route_rejects_unknown_intersection():
    response = client.post(
        "/agents/route",
        json={
            "origin": "X",
            "destination": "E",
        },
    )

    assert response.status_code == 422


def test_ambulance_search_endpoint():
    response = client.post(
        "/agents/ambulances/search",
        json={
            "incident_location": "E",
            "required_capabilities": ["ALS"],
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert "ambulances" in data

    ambulance_ids = [
        ambulance["ambulance_id"]
        for ambulance in data["ambulances"]
    ]

    assert "AMB-101" in ambulance_ids
    assert "AMB-102" not in ambulance_ids
    assert "AMB-103" not in ambulance_ids


def test_hospital_search_endpoint():
    response = client.post(
        "/agents/hospitals/search",
        json={
            "incident_location": "A",
            "required_capabilities": ["TRAUMA"],
        },
    )

    assert response.status_code == 200

    data = response.json()
    hospital_ids = [
        hospital["hospital_id"]
        for hospital in data["hospitals"]
    ]

    assert "HOSP-101" in hospital_ids
    assert "HOSP-102" not in hospital_ids
    assert "HOSP-103" not in hospital_ids


def test_combined_emergency_recommendation():
    response = client.post(
        "/agents/emergency/recommend",
        json={
            "incident_location": "E",
            "required_capabilities": ["ALS"],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["map_type"] == "synthetic"
    assert data["dispatch_authorized"] is False
    assert "ambulance_candidates" in data
    assert "hospital_candidates" in data
