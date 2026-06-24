"""In-memory job tracking for video-processing requests.

A simple dict-backed store is enough for a single-user tool; there is no
need for a database or task queue here. Jobs are lost on restart, which
is an acceptable trade-off for this scale.

Each job also accumulates two pieces of light-scenario-independent data
while the (expensive, GPU-bound) pipeline runs:

- `crossing_events`: every time a tracked vehicle's box reaches the
  bottom of the tracking zone, regardless of what the traffic light was
  doing, along with what the light actually was at that moment.
- `track_summaries`: everything else about a track that doesn't depend
  on the traffic light (vehicle type, plate reading, speed, whether it
  was speeding).

Together these let `pipeline.recompute_offenders` re-judge the whole
video under a different traffic light scenario instantly, without
touching the GPU again.
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from schemas import OffenderEntry


@dataclass
class CrossingEvent:
    track_id: int
    frame_number: int
    detected_light: str


@dataclass
class TrackSummary:
    vehicle_type: str
    speed_kmh: Optional[float]
    speed_method: str
    plate_number: str
    snapshot_filename: Optional[str]
    timestamp_frame: int
    speeding: bool


@dataclass
class JobRecord:
    job_id: str
    session_label: str = ""
    video_path: str = ""
    output_dir: str = ""
    status: str = "queued"  # queued | processing | postprocessing | done | error
    current_frame: int = 0
    total_frames: int = 0
    traffic_light_detected: str = "unknown"
    offenders: List[OffenderEntry] = field(default_factory=list)
    crossing_events: List[CrossingEvent] = field(default_factory=list)
    track_summaries: Dict[int, TrackSummary] = field(default_factory=dict)
    annotated_video_path: Optional[str] = None
    depth_preview_path: Optional[str] = None
    error_message: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def progress_percent(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(100.0, round(100.0 * self.current_frame / self.total_frames, 1))


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._next_session_number = 1

    def create(self) -> JobRecord:
        with self._lock:
            session_label = f"Session {self._next_session_number}"
            self._next_session_number += 1
            job = JobRecord(job_id=uuid.uuid4().hex[:12], session_label=session_label)
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)


job_store = JobStore()
