# 🚨 RESCUE-AI – Emergency Response Coordination System

**An intelligent multi-agent system for emergency resource discovery, route optimization, and emergency response coordination.**

RESCUE-AI is a research prototype designed to demonstrate how intelligent agents can support emergency response coordination by identifying suitable emergency resources and calculating efficient routes.

The system currently implements a synthetic road network, ambulance and hospital resource search, shortest-time route calculation, and REST API integration using Python and FastAPI.

---

## 📌 Project Overview

During emergencies, identifying suitable resources and determining efficient routes are important parts of response coordination.

RESCUE-AI explores a multi-agent approach to this problem through specialized agents that work together to:

- Identify available emergency ambulances.
- Filter resources according to required medical capabilities.
- Search for suitable hospitals based on capabilities and reported bed availability.
- Calculate routes between locations.
- Estimate travel distance and travel time.
- Rank suitable resources based on estimated travel time.
- Expose agent functionality through REST APIs.

**Current implementation:** A research prototype using synthetic road-network and resource data. It is not connected to live emergency services.

---

## 🎯 Project Objectives

1. Develop specialized agents for emergency resource search and route calculation.
2. Represent a road network using graph-based structures.
3. Apply Dijkstra's shortest-path algorithm for travel-time optimization.
4. Identify suitable ambulances based on availability and capabilities.
5. Identify hospitals based on status, capabilities, and reported bed availability.
6. Provide API endpoints for communication between system components.
7. Build a foundation for integration with the RESCUE-AI Coordinator Agent.

---

## 🏗️ System Architecture

The current implementation consists of the following components:

```text
                 RESCUE-AI
                     |
             Emergency Incident
                     |
                     v
          Resource Search Agent
                     |
              +------+------+
              |             |
              v             v
        Ambulance Search  Hospital Search
              |             |
              +------+------+
                     |
                     v
                Route Agent
                     |
                     v
             Synthetic Road Map
                     |
                     v
             Dijkstra Algorithm
                     |
                     v
          Distance and ETA Calculation
                     |
                     v
           Ranked Recommendations
                     |
                     v
             Coordinator Integration
```

### Agent Responsibilities

#### 1. Route Agent

The Route Agent calculates the fastest route between two intersections using Dijkstra's shortest-path algorithm.

**Responsibilities:**
- Read the synthetic road network.
- Identify available paths between intersections.
- Calculate the minimum travel-time path.
- Calculate total route distance.
- Estimate travel time.
- Return structured route information.

**Example:**

```json
{
  "origin": "A",
  "destination": "E",
  "route": ["A", "F", "D", "E"],
  "distance_m": 2200,
  "travel_time_seconds": 300,
  "eta_minutes": 5.0,
  "algorithm": "Dijkstra",
  "map_type": "synthetic"
}
```

#### 2. Resource Search Agent

The Resource Search Agent identifies and ranks suitable emergency resources.

**Ambulance Search:**
- Checks ambulance availability.
- Filters by required medical capabilities.
- Calculates estimated travel time.
- Ranks eligible ambulances by ETA.

**Hospital Search:**
- Checks hospital operational status.
- Checks reported available beds.
- Filters by required medical capabilities.
- Calculates estimated travel time.
- Ranks eligible hospitals by ETA.

The agent returns candidate recommendations for further coordination.

---

## 🗺️ Synthetic Map Approach

The current prototype uses a synthetic road network built with NetworkX.

Instead of downloading real-world map data, the project creates fictional intersections and roads for controlled testing.

### Road Network

The synthetic map contains:

- 6 intersections.
- 8 roads.
- Distance information for each road.
- Estimated travel time for each road.

### Sample Intersections

| ID | Intersection |
|---|---|
| A | North Junction |
| B | Central Junction |
| C | East Junction |
| D | South Junction |
| E | Hospital Junction |
| F | West Junction |

Each intersection represents a graph node, and each road represents an edge.

Each edge stores:

- `distance_m` – road distance in meters.
- `travel_time_s` – estimated travel time in seconds.

### Routing Algorithm

The Route Agent uses Dijkstra's algorithm with `travel_time_s` as the edge weight.

This means the selected route minimizes estimated travel time rather than simply minimizing physical distance.

**Note:** All roads, locations, distances, and travel times are fictional demonstration data.

---

## 🚑 Sample Resource Data

The prototype includes fictional ambulance records.

| Ambulance ID | Location | Status | Capability |
|---|---|---|---|
| AMB-101 | A | Available | ALS |
| AMB-102 | B | Available | BLS |
| AMB-103 | F | Busy | ALS |
| AMB-104 | D | Available | ALS |

### Sample Hospital Data

| Hospital ID | Hospital Name | Location | Available Beds | Capability |
|---|---|---|---:|---|
| HOSP-101 | Central Trauma Hospital | E | 5 | Emergency, Trauma |
| HOSP-102 | Community Hospital | C | 3 | Emergency |
| HOSP-103 | South Medical Center | D | 0 | Emergency, Trauma |

These records are stored in memory and are used only for prototype testing.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| NetworkX | Synthetic graph and road-network representation |
| FastAPI | REST API development |
| Uvicorn | API server |
| Pydantic | Request validation |
| Pytest | Automated testing |
| HTTPX | API testing through FastAPI TestClient |
| Git & GitHub | Version control and collaboration |

---

## 📂 Project Structure

```text
RESCUE-AI/
│
├── src/
│   └── rescue_ai/
│       ├── __init__.py
│       ├── synthetic_map.py
│       ├── route_agent.py
│       ├── resource_search_agent.py
│       └── api.py
│
├── tests/
│   ├── test_agents.py
│   └── test_api.py
│
├── .venv/
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12 or compatible Python 3 version.
- Git.
- Visual Studio Code (recommended).

### 1. Clone the Repository

```bash
git clone https://github.com/Techie-calster/RESCUE-AI.git
```

### 2. Navigate to the Project

```bash
cd RESCUE-AI
```

### 3. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
```

### 4. Activate the Environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip

python -m pip install networkx fastapi uvicorn pytest httpx
```

### 6. Run the Synthetic Map

```bash
python -m src.rescue_ai.synthetic_map
```

### 7. Run the Route Agent

```bash
python -m src.rescue_ai.route_agent
```

### 8. Run the Resource Search Agent

```bash
python -m src.rescue_ai.resource_search_agent
```

---

## 🚀 Running the FastAPI Server

Start the API server from the project root:

```bash
python -m uvicorn src.rescue_ai.api:app --reload
```

The API will be available at:

**Base URL:** `http://127.0.0.1:8000`

**Interactive API Documentation:**

http://127.0.0.1:8000/docs

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/agents/route` | Calculate fastest route |
| POST | `/agents/ambulances/search` | Search suitable ambulances |
| POST | `/agents/hospitals/search` | Search suitable hospitals |
| POST | `/agents/emergency/recommend` | Get combined resource recommendations |

### 1. Health Check

```http
GET /health
```

### 2. Route Calculation

```http
POST /agents/route
```

Request:

```json
{
  "origin": "A",
  "destination": "E"
}
```

### 3. Ambulance Search

```http
POST /agents/ambulances/search
```

Request:

```json
{
  "incident_location": "E",
  "required_capabilities": ["ALS"]
}
```

### 4. Hospital Search

```http
POST /agents/hospitals/search
```

Request:

```json
{
  "incident_location": "A",
  "required_capabilities": ["TRAUMA"]
}
```

### 5. Combined Emergency Recommendation

```http
POST /agents/emergency/recommend
```

Request:

```json
{
  "incident_location": "E",
  "required_capabilities": ["ALS"]
}
```

The combined endpoint returns ambulance candidates, hospital candidates, and a coordinator status indicating that further decision-making is required.

No actual dispatch is performed.

---

## 🧪 Testing

The project uses Pytest for automated testing.

Run all tests:

```bash
python -m pytest tests -v
```

Current local testing results:

**13 tests passed successfully.**

The tests cover:

- Fastest route calculation.
- Invalid intersection handling.
- Ambulance availability filtering.
- Ambulance capability filtering.
- Ambulance ETA ranking.
- Hospital capacity filtering.
- Hospital capability filtering.
- Hospital ETA ranking.
- API health check.
- Route API.
- Ambulance search API.
- Hospital search API.
- Combined emergency recommendation API.

---

## 🔮 Future Scope

The following are planned improvements and are not yet implemented in the current prototype:

- Integration with real-world OpenStreetMap road-network data.
- Dynamic route recalculation when roads are blocked.
- Live ambulance GPS tracking.
- Integration with real ambulance availability databases.
- Integration with hospital information systems.
- Real-time hospital bed and emergency department capacity updates.
- Improved resource allocation across multiple incidents.
- Coordinator Agent integration.
- Interactive emergency response map dashboard.
- Deployment and production infrastructure.

---

## ⚠️ Limitations and Safety Disclaimer

RESCUE-AI is currently a research and educational prototype.

- Road-network data is synthetic.
- Ambulance and hospital records are fictional.
- Travel times are predefined estimates.
- Hospital capacity is represented by sample data.
- No real emergency services are connected.
- No live GPS tracking or actual dispatch functionality is implemented.
- Recommendations have not been validated for clinical or operational use.

The system must not be used to make real emergency dispatch or medical decisions.

---

## 👥 Team and Collaboration

**Project:** RESCUE-AI – Emergency Response Coordination System

**GitHub Repository:**  
https://github.com/Techie-calster/RESCUE-AI

The project is developed collaboratively using Git and GitHub.

---

## 📄 License

Add the license selected by the project team before distributing the project publicly.

---

**RESCUE-AI – Towards Smarter Emergency Response Coordination.**