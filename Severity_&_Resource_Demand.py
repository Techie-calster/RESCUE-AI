"""
Rescue-AI: Severity and Resource Demand Agent (Model 2)
Rule-based prototype. No external dependencies.

IMPORTANT:
- This is not a clinically validated triage or dispatch system.
- Wound confidence is interpreted using project-specific prototype thresholds.
- It cannot detect internal hemorrhage, internal organ injury, vital signs, etc.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# -------------------------- Configurable assumptions --------------------------

DEFAULT_CONFIG = {
    "no_wound_max_score": 0.30,       # score <= this => no wound detected
    "severe_wound_threshold": 0.80,   # score > this => severe wound
    "casualty_thresholds": {
        "small_max": 2,
        "moderate_max": 5,
        "high_max": 9,
        # 10+ => MASS_CASUALTY
    },
    "mass_casualty_threshold": 10,
    "patients_per_paramedic_team": 10,
    "severe_patients_per_paramedic_team": 3,
    "minimum_teams_for_severe_incident": 1,
    "minimum_teams_for_high_incident": 1,
    "bls_capacity": 2,
    "als_per_severe_patient": 1,
    "police_required_for_detected_accident": True,
}


def load_input(source: Union[str, Path, dict, list]) -> Any:
    """Load JSON from a file path, or pass through an already parsed object."""
    if isinstance(source, (dict, list)):
        return source
    if not isinstance(source, (str, Path)):
        raise TypeError("Input must be a file path, dictionary, or list.")
    try:
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read input file: {exc}") from exc


def parse_analysis_data(data: Any) -> Dict[str, Any]:
    """Validate top-level structure and return the three model sections."""
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object/dictionary.")

    accident = data.get("accident_detection")
    persons = data.get("person_detection")
    wounds = data.get("wound_detection")

    if not isinstance(accident, dict):
        accident = {}
    if not isinstance(persons, dict):
        persons = {}
    if not isinstance(wounds, dict):
        wounds = {}

    return {
        "raw": data,
        "accident_detection": accident,
        "person_detection": persons,
        "wound_detection": wounds,
    }


def _valid_score(value: Any) -> Optional[float]:
    """Return a numeric score in [0, 1], otherwise None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _normalize_evaluation_status(person_wound: dict) -> str:
    status = person_wound.get("evaluation_status")
    if isinstance(status, str) and status.strip():
        return status.strip().lower()
    # Actual supplied JSON uses "completed".
    if "evaluation_status" not in person_wound:
        return "unknown"
    return "unknown"


def assess_individual_patients(
    person_detection: dict,
    wound_detection: dict,
    config: dict,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """Match person and wound records by person_id; assess each detected person."""
    detected_people = person_detection.get("people")
    wound_people = wound_detection.get("people")

    if not isinstance(detected_people, list):
        detected_people = []
        warnings.append("person_detection.people is missing or is not a list.")
    if not isinstance(wound_people, list):
        wound_people = []
        warnings.append("wound_detection.people is missing or is not a list.")

    wound_by_id = {}
    duplicate_wound_ids = set()
    for record in wound_people:
        if not isinstance(record, dict):
            warnings.append("A wound-detection person record is malformed and was skipped.")
            continue
        pid = record.get("person_id")
        if pid in wound_by_id:
            duplicate_wound_ids.add(pid)
        else:
            wound_by_id[pid] = record

    for pid in duplicate_wound_ids:
        warnings.append(f"Duplicate wound person_id {pid!r}; assessment marked unknown.")

    results = []
    seen_person_ids = set()
    for person in detected_people:
        if not isinstance(person, dict):
            warnings.append("A person-detection record is malformed and was skipped.")
            continue

        pid = person.get("person_id")
        if pid in seen_person_ids:
            warnings.append(f"Duplicate detected person_id {pid!r}; duplicate record skipped.")
            continue
        seen_person_ids.add(pid)

        if pid in duplicate_wound_ids:
            wound_record = None
        else:
            wound_record = wound_by_id.get(pid)

        base = {
            "person_id": pid,
            "person_detection_confidence": person.get("confidence"),
            "wound_evaluation_status": "unknown",
            "wounds_detected": 0,
            "individual_severity_score": None,
            "wound_category": "UNKNOWN",
            "considered_injured": None,
            "severe_wound": None,
        }

        if wound_record is None:
            base["wound_evaluation_status"] = "unknown"
            base["notes"] = "No unique wound-assessment record matched this person_id."
            results.append(base)
            continue

        eval_status = _normalize_evaluation_status(wound_record)
        base["wound_evaluation_status"] = eval_status
        if eval_status != "completed":
            base["notes"] = "Wound evaluation was skipped, failed, or incomplete."
            results.append(base)
            continue

        wounds = wound_record.get("wounds")
        if not isinstance(wounds, list):
            wounds = []
            warnings.append(f"Person {pid!r}: wounds field missing or malformed.")

        valid_scores = []
        invalid_score_found = False
        for wound in wounds:
            if not isinstance(wound, dict):
                invalid_score_found = True
                warnings.append(f"Person {pid!r}: malformed wound entry.")
                continue
            score = _valid_score(wound.get("confidence"))
            if score is None:
                invalid_score_found = True
                warnings.append(
                    f"Person {pid!r}: missing, invalid, or out-of-range wound confidence."
                )
            else:
                valid_scores.append(score)

        # Use wound confidences directly; no second upstream adjustment.
        if valid_scores:
            score = max(valid_scores)
            base["individual_severity_score"] = round(score, 4)
            base["wounds_detected"] = len(wounds)
            if score <= config["no_wound_max_score"]:
                base["wound_category"] = "NO_WOUND_DETECTED"
                base["considered_injured"] = False
                base["severe_wound"] = False
            elif score <= config["severe_wound_threshold"]:
                base["wound_category"] = "NON_SEVERE_WOUND"
                base["considered_injured"] = True
                base["severe_wound"] = False
            else:
                base["wound_category"] = "SEVERE_WOUND"
                base["considered_injured"] = True
                base["severe_wound"] = True

            if invalid_score_found:
                base["notes"] = (
                    "At least one wound score was invalid; category is based on valid scores only."
                )
        else:
            # Completed, no wound boxes and wound_detected false means evaluated/no wound.
            # If upstream explicitly says wound_detected true but no usable scores, status is unknown.
            if wound_record.get("wound_detected") is False and not wounds:
                base["wound_category"] = "NO_WOUND_DETECTED"
                base["considered_injured"] = False
                base["severe_wound"] = False
                base["individual_severity_score"] = 0.0
                base["wounds_detected"] = 0
            else:
                base["wound_evaluation_status"] = "unknown"
                base["notes"] = "No valid wound confidence was available; not treated as score 0."

        results.append(base)

    # Wound records with no corresponding person record should be reported.
    detected_ids = {p.get("person_id") for p in detected_people if isinstance(p, dict)}
    for wound_person in wound_people:
        if isinstance(wound_person, dict) and wound_person.get("person_id") not in detected_ids:
            warnings.append(
                f"Wound record person_id {wound_person.get('person_id')!r} has no matching detected person."
            )

    return results


def calculate_patient_statistics(
    assessments: List[dict], person_detection: dict, wound_detection: dict, warnings: List[str]
) -> Dict[str, int]:
    """Count patients, not wound boxes."""
    total_detected = len(assessments)
    evaluated = sum(p["wound_evaluation_status"] == "completed" for p in assessments)
    injured = sum(p["considered_injured"] is True for p in assessments)
    severe = sum(p["severe_wound"] is True for p in assessments)
    non_severe = sum(
        p["considered_injured"] is True and p["severe_wound"] is False for p in assessments
    )
    unknown = sum(p["considered_injured"] is None for p in assessments)

    upstream_count = person_detection.get("count")
    if isinstance(upstream_count, int) and upstream_count != len(
        person_detection.get("people", []) if isinstance(person_detection.get("people"), list) else []
    ):
        warnings.append(
            f"person_detection.count ({upstream_count}) does not match people-array length."
        )

    if wound_detection.get("people_total") is not None:
        people_total = wound_detection.get("people_total")
        if isinstance(people_total, int) and people_total != total_detected:
            warnings.append(
                f"wound_detection.people_total ({people_total}) differs from detected-person count ({total_detected})."
            )

    return {
        "total_people_detected": total_detected,
        "people_evaluated": evaluated,
        "total_injured": injured,
        "non_severe_injured": non_severe,
        "severe_wound_patients": severe,
        "unknown_injury_status": unknown,
    }


def classify_casualty_load(total_injured: int, config: dict) -> str:
    thresholds = config["casualty_thresholds"]
    if total_injured <= 0:
        return "NONE"
    if total_injured <= thresholds["small_max"]:
        return "SMALL"
    if total_injured <= thresholds["moderate_max"]:
        return "MODERATE"
    if total_injured <= thresholds["high_max"]:
        return "HIGH"
    return "MASS_CASUALTY"


def classify_incident_severity(
    stats: dict,
    casualty_load: str,
    accident_detected: Optional[bool],
    assessment_complete: bool,
    config: dict,
) -> Dict[str, Any]:
    """Rule-based incident severity; output includes explanation reasons."""
    reasons = []
    if accident_detected is False:
        return {"severity": "NOT_DETECTED", "reasons": ["Accident detection is false."]}

    injured = stats["total_injured"]
    severe = stats["severe_wound_patients"]
    non_severe = stats["non_severe_injured"]

    if casualty_load == "MASS_CASUALTY":
        reasons.append("Mass-casualty threshold reached.")
        severity = "CRITICAL"
    elif severe >= 3:
        reasons.append(f"{severe} severe wound patients (3 or more).")
        severity = "CRITICAL"
    elif severe in (1, 2):
        reasons.append(f"{severe} severe wound patient(s) present.")
        severity = "HIGH"
    elif non_severe >= 6:
        reasons.append(f"{non_severe} non-severe injured patients (6 or more).")
        severity = "HIGH"
    elif 3 <= injured <= 5 and severe == 0:
        reasons.append(f"{injured} injured patients and no severe wound patients.")
        severity = "MEDIUM"
    elif 1 <= injured <= 2 and severe == 0:
        reasons.append(f"{injured} injured patient(s) and no severe wound patients.")
        severity = "LOW"
    elif injured == 0:
        reasons.append("Accident detected but no wound-positive patients identified.")
        severity = "LOW"
    else:
        reasons.append("Patient counts do not match a specific configured severity rule.")
        severity = "LOW"

    if not assessment_complete:
        reasons.append("Assessment is incomplete; severity is provisional.")
        return {"severity": severity, "reasons": reasons, "uncertain": True}

    return {"severity": severity, "reasons": reasons, "uncertain": False}


def calculate_paramedic_team_demand(
    total_injured: int,
    severe_patient_count: int,
    incident_severity: str,
    casualty_load: str,
    config: dict,
) -> Dict[str, Any]:
    """Calculate on-scene teams using capacity and severe-patient assumptions."""
    patient_capacity = config["patients_per_paramedic_team"]
    severe_capacity = config["severe_patients_per_paramedic_team"]
    if patient_capacity <= 0 or severe_capacity <= 0:
        raise ValueError("Paramedic team capacities must be positive.")

    casualty_demand = math.ceil(total_injured / patient_capacity) if total_injured else 0
    severe_demand = math.ceil(severe_patient_count / severe_capacity) if severe_patient_count else 0

    support_triggered = (
        severe_patient_count > 0
        or casualty_load == "MASS_CASUALTY"
        or incident_severity in ("HIGH", "CRITICAL")
    )

    teams = max(casualty_demand, severe_demand) if support_triggered else 0
    if incident_severity == "CRITICAL" and support_triggered:
        teams = max(teams, config["minimum_teams_for_severe_incident"])
    elif incident_severity == "HIGH" and support_triggered:
        teams = max(teams, config["minimum_teams_for_high_incident"])

    reasons = []
    if casualty_demand:
        reasons.append(
            f"Casualty capacity: ceil({total_injured}/{patient_capacity}) = {casualty_demand} team(s)."
        )
    if severe_demand:
        reasons.append(
            f"Severe-patient capacity: ceil({severe_patient_count}/{severe_capacity}) = {severe_demand} team(s)."
        )
    if support_triggered and teams == 0:
        teams = 1
        reasons.append("Support trigger applied minimum of one team.")
    if casualty_load == "MASS_CASUALTY":
        reasons.append("Mass-casualty threshold triggers on-scene paramedic support.")
    if not support_triggered:
        reasons.append("No configured on-scene paramedic support trigger.")

    return {
        "teams_required": teams,
        "casualty_load_team_demand": casualty_demand,
        "severe_patient_team_demand": severe_demand,
        "support_triggered": support_triggered,
        "reasons": reasons,
    }


def calculate_resource_demand(
    stats: dict,
    incident_severity: str,
    casualty_load: str,
    accident_detected: bool,
    config: dict,
) -> Dict[str, Any]:
    severe = stats["severe_wound_patients"]
    non_severe = stats["non_severe_injured"]
    injured = stats["total_injured"]

    als = severe * config["als_per_severe_patient"]
    bls = math.ceil(non_severe / config["bls_capacity"]) if non_severe else 0

    paramedic = calculate_paramedic_team_demand(
        injured, severe, incident_severity, casualty_load, config
    )

    police = bool(accident_detected and config["police_required_for_detected_accident"])
    reasons = [
        f"ALS: {severe} severe patient(s) × {config['als_per_severe_patient']} = {als}.",
        f"BLS: ceil({non_severe}/{config['bls_capacity']}) = {bls} (non-severe patients only).",
        f"Paramedic teams: {paramedic['teams_required']} based on configured capacity/support rules.",
        "Police required for every detected accident under project policy."
        if police else "Police not triggered by current project policy/input.",
        "Fire department omitted: no upstream fire-detection input is available.",
    ]

    return {
        "ALS_ambulances": als,
        "BLS_ambulances": bls,
        "paramedic_teams": paramedic["teams_required"],
        "police_required": police,
        "fire_department": "NOT_ASSESSED",
        "paramedic_team_details": paramedic,
        "resource_reasons": reasons,
    }


def generate_output(
    parsed: dict,
    assessments: List[dict],
    stats: dict,
    incident_info: dict,
    casualty_load: str,
    resources: dict,
    warnings: List[str],
) -> Dict[str, Any]:
    accident_detected = parsed["accident_detection"].get("detected")
    status = "completed"
    if accident_detected is not True:
        status = "accident_not_detected" if accident_detected is False else "incomplete"
    elif stats["total_people_detected"] == 0:
        status = "no_people_detected"
    elif stats["unknown_injury_status"] > 0:
        status = "partial"
    elif stats["people_evaluated"] == 0:
        status = "incomplete"

    if accident_detected is not True:
        warnings.append("Accident detection is false or unknown; do not interpret this as proof of safety.")
    if stats["unknown_injury_status"]:
        warnings.append(
            f"{stats['unknown_injury_status']} person(s) have unknown injury status; unknown does not mean uninjured."
        )
    if stats["total_people_detected"] == 0 and accident_detected is True:
        warnings.append("Accident detected but no people were detected; actual casualties may be unobserved.")
    if parsed["wound_detection"].get("people_skipped", 0):
        warnings.append("Upstream wound detection reports skipped person evaluations.")
    warnings.append(
        "Prototype limitation: visual wound scores do not assess internal hemorrhage, internal organ injury, "
        "pulse, breathing, blood pressure, consciousness, or other vital signs."
    )
    warnings.append(
        "This output is a project prototype, not medically validated triage or real-world dispatch guidance."
    )

    return {
        "status": status,
        "accident_detected": accident_detected,
        "incident_severity": incident_info["severity"],
        "incident_severity_uncertain": incident_info.get("uncertain", False),
        "casualty_load": casualty_load,
        "patient_statistics": stats,
        "individual_patient_assessments": assessments,
        "resource_demand": {
            "ALS_ambulances": resources["ALS_ambulances"],
            "BLS_ambulances": resources["BLS_ambulances"],
            "paramedic_teams": resources["paramedic_teams"],
            "police_required": resources["police_required"],
            "fire_department": resources["fire_department"],
            "paramedic_team_details": resources["paramedic_team_details"],
        },
        "severity_reasons": incident_info["reasons"],
        "resource_reasons": resources["resource_reasons"],
        "warnings": warnings,
    }


def analyze_accident(source: Union[str, Path, dict, list], config_overrides: Optional[dict] = None) -> Dict[str, Any]:
    """Main pipeline. Accepts file path, dict, or parsed JSON."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # independent nested copy
    if config_overrides:
        # Simple top-level/nested dictionary override.
        for key, value in config_overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value

    warnings = []
    try:
        data = load_input(source)
        parsed = parse_analysis_data(data)
    except (ValueError, TypeError) as exc:
        return {
            "status": "invalid_input",
            "accident_detected": None,
            "incident_severity": "UNKNOWN",
            "casualty_load": "UNKNOWN",
            "patient_statistics": {
                "total_people_detected": 0, "people_evaluated": 0, "total_injured": 0,
                "non_severe_injured": 0, "severe_wound_patients": 0, "unknown_injury_status": 0,
            },
            "individual_patient_assessments": [],
            "resource_demand": {
                "ALS_ambulances": 0, "BLS_ambulances": 0, "paramedic_teams": 0,
                "police_required": False, "fire_department": "NOT_ASSESSED",
            },
            "severity_reasons": ["Input could not be parsed."],
            "resource_reasons": [],
            "warnings": [str(exc)],
        }

    accident = parsed["accident_detection"]
    person_detection = parsed["person_detection"]
    wound_detection = parsed["wound_detection"]
    accident_detected = accident.get("detected")

    if "detected" not in accident:
        warnings.append("accident_detection.detected is missing.")
    if "detected" not in person_detection:
        warnings.append("person_detection.detected is missing.")
    if "people" not in wound_detection:
        warnings.append("wound_detection.people is missing; injury assessment may be incomplete.")

    assessments = assess_individual_patients(
        person_detection, wound_detection, config, warnings
    )
    stats = calculate_patient_statistics(assessments, person_detection, wound_detection, warnings)

    assessment_complete = (
        person_detection.get("detected") is True
        and isinstance(person_detection.get("people"), list)
        and wound_detection.get("people") is not None
        and stats["unknown_injury_status"] == 0
    )

    casualty_load = classify_casualty_load(stats["total_injured"], config)
    incident_info = classify_incident_severity(
        stats, casualty_load, accident_detected, assessment_complete, config
    )

    # If upstream assessment is incomplete, the severity label is provisional.
    resources = calculate_resource_demand(
        stats, incident_info["severity"], casualty_load,
        accident_detected is True, config
    )

    return generate_output(
        parsed, assessments, stats, incident_info, casualty_load, resources, warnings
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rescue-AI Severity & Resource Demand Agent")
    parser.add_argument("json_path", help="Path to accident analysis JSON")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")
    args = parser.parse_args()

    result = analyze_accident(args.json_path)
    print(json.dumps(result, indent=2 if args.pretty else None))
