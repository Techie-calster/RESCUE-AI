
import pytest

from src.rescue_ai.route_agent import RouteAgent
from src.rescue_ai.resource_search_agent import ResourceSearchAgent


@pytest.fixture
def route_agent():
    return RouteAgent()


@pytest.fixture
def resource_agent():
    return ResourceSearchAgent()


def test_route_finds_fastest_path(route_agent):
    result = route_agent.calculate_route("A", "E")

    assert result["route"][0] == "A"
    assert result["route"][-1] == "E"
    assert result["distance_m"] > 0
    assert result["travel_time_seconds"] > 0


def test_route_rejects_unknown_intersection(route_agent):
    with pytest.raises(ValueError):
        route_agent.calculate_route("X", "E")


def test_ambulance_search_excludes_busy_units(resource_agent):
    results = resource_agent.search_ambulances(
        incident_location="E",
        required_capabilities=["ALS"],
    )

    ambulance_ids = [item["ambulance_id"] for item in results]

    assert "AMB-103" not in ambulance_ids


def test_ambulance_search_checks_capabilities(resource_agent):
    results = resource_agent.search_ambulances(
        incident_location="E",
        required_capabilities=["ALS"],
    )

    ambulance_ids = [item["ambulance_id"] for item in results]

    assert "AMB-101" in ambulance_ids
    assert "AMB-104" in ambulance_ids
    assert "AMB-102" not in ambulance_ids


def test_ambulances_are_ranked_by_eta(resource_agent):
    results = resource_agent.search_ambulances(
        incident_location="E",
        required_capabilities=["ALS"],
    )

    etas = [item["eta_minutes"] for item in results]

    assert etas == sorted(etas)


def test_hospital_search_filters_capacity_and_capabilities(
    resource_agent,
):
    results = resource_agent.search_hospitals(
        incident_location="A",
        required_capabilities=["TRAUMA"],
    )

    hospital_ids = [item["hospital_id"] for item in results]

    assert "HOSP-101" in hospital_ids
    assert "HOSP-102" not in hospital_ids
    assert "HOSP-103" not in hospital_ids


def test_hospitals_are_ranked_by_eta(resource_agent):
    results = resource_agent.search_hospitals(
        incident_location="A",
        required_capabilities=["TRAUMA"],
    )

    etas = [item["eta_minutes"] for item in results]

    assert etas == sorted(etas)
