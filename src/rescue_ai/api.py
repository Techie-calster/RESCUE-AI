
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rescue_ai.route_agent import RouteAgent
from src.rescue_ai.resource_search_agent import ResourceSearchAgent


app = FastAPI(
    title="RESCUE-AI Agent API",
    description=(
        "Prototype APIs for synthetic road routing "
        "and emergency resource search."
    ),
    version="0.1.0",
)

route_agent = RouteAgent()
resource_agent = ResourceSearchAgent()


class RouteRequest(BaseModel):
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)


class ResourceSearchRequest(BaseModel):
    incident_location: str = Field(min_length=1)
    required_capabilities: list[str] = Field(default_factory=list)


class HospitalSearchRequest(BaseModel):
    incident_location: str = Field(min_length=1)
    required_capabilities: list[str] = Field(default_factory=list)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "RESCUE-AI Agent API",
        "map_type": "synthetic",
    }


@app.post("/agents/route")
def calculate_route(request: RouteRequest):
    try:
        return route_agent.calculate_route(
            request.origin,
            request.destination,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/agents/ambulances/search")
def search_ambulances(request: ResourceSearchRequest):
    try:
        return {
            "agent": "Resource Search Agent",
            "incident_location": request.incident_location,
            "ambulances": resource_agent.search_ambulances(
                request.incident_location,
                request.required_capabilities,
            ),
            "note": "Prototype recommendations only; no dispatch is performed.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/agents/hospitals/search")
def search_hospitals(request: HospitalSearchRequest):
    try:
        return {
            "agent": "Resource Search Agent",
            "incident_location": request.incident_location,
            "hospitals": resource_agent.search_hospitals(
                request.incident_location,
                request.required_capabilities,
            ),
            "note": "Prototype recommendations only; no dispatch is performed.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/agents/emergency/recommend")
def recommend_emergency_resources(
    request: ResourceSearchRequest,
):
    try:
        ambulances = resource_agent.search_ambulances(
            request.incident_location,
            request.required_capabilities,
        )

        hospitals = resource_agent.search_hospitals(
            request.incident_location,
            ["TRAUMA"],
        )

        return {
            "incident_location": request.incident_location,
            "map_type": "synthetic",
            "ambulance_candidates": ambulances,
            "hospital_candidates": hospitals,
            "coordinator_status": "awaiting_coordinator_decision",
            "dispatch_authorized": False,
            "note": (
                "Synthetic research prototype. These are candidate "
                "recommendations, not real emergency dispatch instructions."
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
