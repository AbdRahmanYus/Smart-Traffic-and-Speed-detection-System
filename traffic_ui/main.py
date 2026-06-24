"""FastAPI app: a small dashboard for the traffic surveillance pipeline.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

See README.md for instructions on running this inside Google Colab so
the heavy GPU pipeline (YOLO, RAFT, Depth Anything V2) has a free T4 to
work with.
"""

import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
import pipeline
from jobs import job_store
from schemas import JobStatus, OffenderEntry, ScenarioRequest, TrafficLightStatus, TrafficLightUpdate
from traffic_light import TrafficLightOverride

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

light_override = TrafficLightOverride()
models: pipeline.ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    logger.info("Loading models, this can take a minute...")
    models = pipeline.load_models()
    logger.info("Models loaded, ready to accept jobs.")
    yield


app = FastAPI(title="Smart Traffic Surveillance", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

os.makedirs(config.JOB_OUTPUT_ROOT, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/jobs")
async def create_job(video: UploadFile = File(...)) -> dict:
    if models is None:
        raise HTTPException(503, "Models are still loading, try again shortly.")

    job = job_store.create()
    output_dir = os.path.join(config.JOB_OUTPUT_ROOT, job.job_id)
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, video.filename)
    with open(video_path, "wb") as out_file:
        shutil.copyfileobj(video.file, out_file)

    job.video_path = video_path
    job.output_dir = output_dir

    thread = threading.Thread(
        target=pipeline.run_pipeline, args=(job, models, light_override), daemon=True
    )
    thread.start()

    return {"job_id": job.job_id, "session_label": job.session_label}


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")

    effective_label, _ = pipeline.resolve_traffic_state(job.traffic_light_detected, light_override)

    return JobStatus(
        job_id=job.job_id,
        session_label=job.session_label,
        status=job.status,
        progress_percent=job.progress_percent,
        current_frame=job.current_frame,
        total_frames=job.total_frames,
        traffic_light_detected=job.traffic_light_detected,
        traffic_light_effective=effective_label,
        offender_count=len(job.offenders),
        error_message=job.error_message,
    )


@app.get("/jobs/{job_id}/offenders", response_model=List[OffenderEntry])
def get_offenders(job_id: str) -> List[OffenderEntry]:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    with job.lock:
        return list(job.offenders)


@app.post("/jobs/{job_id}/recompute", response_model=List[OffenderEntry])
def recompute_job_offenders(job_id: str, body: ScenarioRequest) -> List[OffenderEntry]:
    """Re-judges a finished job's offender log under a different traffic
    light scenario - no GPU work, just replays the cached crossing events.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if job.status != "done":
        raise HTTPException(409, "Job is still processing - wait until it's done to replay a scenario.")

    offenders = pipeline.recompute_offenders(job, body.scenario)
    with job.lock:
        job.offenders = offenders
    return offenders


@app.get("/jobs/{job_id}/snapshots/{filename}")
def get_snapshot(job_id: str, filename: str) -> FileResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    path = os.path.join(job.output_dir, "snapshots", filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Snapshot not found")
    return FileResponse(path)


@app.get("/jobs/{job_id}/depth")
def get_depth_preview(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if not job.depth_preview_path or not os.path.isfile(job.depth_preview_path):
        raise HTTPException(404, "No depth preview yet")
    return FileResponse(job.depth_preview_path)


@app.get("/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if not job.annotated_video_path or not os.path.isfile(job.annotated_video_path):
        raise HTTPException(404, "Video not ready yet")
    return FileResponse(job.annotated_video_path, media_type="video/mp4")


@app.get("/traffic-light", response_model=TrafficLightStatus)
def get_traffic_light() -> TrafficLightStatus:
    return TrafficLightStatus(override=light_override.get())


@app.post("/traffic-light", response_model=TrafficLightStatus)
def set_traffic_light(update: TrafficLightUpdate) -> TrafficLightStatus:
    light_override.set(update.state)
    return TrafficLightStatus(override=update.state)
