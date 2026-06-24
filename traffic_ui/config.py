"""Configuration constants for the traffic surveillance pipeline.

These values are carried over unchanged from the original notebook
(Namikaze_Sama.ipynb) so the FastAPI app produces the same behaviour.
Adjust PROJECT_DIR (or set the PROJECT_DIR environment variable) to point
at wherever your trained weights live, e.g. a Google Drive folder.
"""

import os
from dataclasses import dataclass

PROJECT_DIR = os.environ.get(
    "PROJECT_DIR",
    "/content/drive/MyDrive/Smart traffic and speed detection system",
)

WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
TRAFFIC_LIGHT_WEIGHTS = os.path.join(WEIGHTS_DIR, "traffic_light_best.pt")
VEHICLE_WEIGHTS = os.path.join(WEIGHTS_DIR, "vehicle_best.pt")
LICENSE_PLATE_WEIGHTS = os.path.join(WEIGHTS_DIR, "license_plate_best.pt")

DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# Where uploaded videos and per-job outputs (annotated video, snapshots,
# depth previews) are written.
JOB_OUTPUT_ROOT = os.environ.get(
    "JOB_OUTPUT_ROOT", os.path.join(PROJECT_DIR, "outputs", "jobs")
)

# Vehicle height priors in metres, used to convert relative depth into
# metres-per-pixel.
VEHICLE_HEIGHT_M = {
    "car": 1.5,
    "truck": 2.5,
    "bus": 3.0,
    "motorcycle": 1.2,
    "bike": 1.2,
    "default": 1.5,
}

VEHICLE_ALLOWED_CLASSES = {"3wheeler", "car", "truck", "bus", "bike"}

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
SPEED_LIMIT_KMH = 100.0

KALMAN_PROCESS_NOISE = 0.5
KALMAN_MEASUREMENT_NOISE = 8.0
KALMAN_INITIAL_VARIANCE = 10.0

EMA_ALPHA_BOX = 0.15
EMA_ALPHA_DEPTH = 0.10
EMA_ALPHA_SCALE = 0.15

MEDIAN_BUFFER_FRAMES = 9
SPEED_DISPLAY_DELTA_KMH = 2.0
MIN_FLOW_PIXELS = 1.5

STATIONARY_CHECK_SECONDS = 0.5
STATIONARY_PIXEL_THRESHOLD = 3.0
STATIONARY_RELATIVE_FRACTION = 0.03

MAX_PLATE_CROPS = 5
OCR_CONF_THRESHOLD = 0.01

# The traffic light widget shows red/amber/green. The trained detector's
# class names use "yellow" instead of "amber" - this maps one to the other.
# Update it if your model uses a different label.
AMBER_DETECTOR_LABEL = "yellow"

# A vehicle counts as having crossed the stop line once its box reaches
# within this fraction of the frame height from the bottom of the
# tracking zone (ROI_Y2). A small tolerance is needed because the box
# bottom approaches ROI_Y2 asymptotically rather than landing on it exactly.
ROI_EXIT_MARGIN_FRACTION = 0.01


@dataclass
class FrameSettings:
    depth_every_n_frames: int
    frame_interval: int
    stationary_check_frames: int


def frame_settings_for_fps(fps: float) -> FrameSettings:
    """Picks depth-update and RAFT frame intervals based on video fps,
    same thresholds as the notebook (20 frames up to ~31fps, 40 above)."""
    if fps <= 31:
        depth_every_n_frames = 20
        frame_interval = 20
    else:
        depth_every_n_frames = 40
        frame_interval = 40

    stationary_check_frames = max(int(round(STATIONARY_CHECK_SECONDS * fps)), 3)

    return FrameSettings(
        depth_every_n_frames=depth_every_n_frames,
        frame_interval=frame_interval,
        stationary_check_frames=stationary_check_frames,
    )
