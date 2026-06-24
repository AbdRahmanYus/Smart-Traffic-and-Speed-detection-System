"""Pydantic models for API request and response bodies."""

from typing import Literal, Optional

from pydantic import BaseModel

TrafficLightState = Literal["auto", "red", "amber", "green"]


class TrafficLightUpdate(BaseModel):
    state: TrafficLightState


class ScenarioRequest(BaseModel):
    scenario: TrafficLightState


class TrafficLightStatus(BaseModel):
    override: TrafficLightState


class OffenderEntry(BaseModel):
    track_id: int
    vehicle_type: str
    plate_number: str
    speed_kmh: Optional[float]
    speed_method: str
    violation_type: str
    snapshot_filename: Optional[str]
    timestamp_frame: int


class JobStatus(BaseModel):
    job_id: str
    session_label: str
    status: Literal["queued", "processing", "postprocessing", "done", "error"]
    progress_percent: float
    current_frame: int
    total_frames: int
    traffic_light_detected: str
    traffic_light_effective: str
    offender_count: int
    error_message: Optional[str] = None
