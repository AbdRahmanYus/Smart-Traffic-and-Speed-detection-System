"""Core video-processing pipeline.

This ports the detection, tracking, and speed-estimation logic from the
original research notebook (Namikaze_Sama.ipynb) into a form a FastAPI
background job can drive. The detection and speed-smoothing logic
(Depth Anything V2 scale + RAFT motion + Kalman + median buffer) is
unchanged from the notebook. What's new is the surrounding plumbing:
per-job state instead of notebook globals, model loading as a single
function, and the manual traffic-light override.
"""

import logging
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as TF

import config
from jobs import CrossingEvent, JobRecord, TrackSummary
from schemas import OffenderEntry
from traffic_light import TrafficLightOverride

logger = logging.getLogger(__name__)

TL_COLOURS = {
    "red": (0, 0, 220),
    "green": (0, 200, 0),
    "yellow": (0, 200, 220),
}


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

@dataclass
class ModelBundle:
    traffic_model: object
    vehicle_model: object
    lp_model: object
    depth_model: object
    depth_processor: object
    raft_model: object
    raft_transforms: object
    ocr_reader: object


def load_models() -> ModelBundle:
    """Loads all detectors and the speed-estimation models onto the GPU.
    Call this once, at app startup."""
    from ultralytics import YOLO
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import easyocr

    logger.info("Loading YOLO detectors...")
    traffic_model = YOLO(config.TRAFFIC_LIGHT_WEIGHTS)
    vehicle_model = YOLO(config.VEHICLE_WEIGHTS)
    lp_model = YOLO(config.LICENSE_PLATE_WEIGHTS)

    logger.info("Loading Depth Anything V2...")
    depth_processor = AutoImageProcessor.from_pretrained(config.DEPTH_MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL_NAME)
    depth_model.to("cuda")
    depth_model.eval()

    logger.info("Loading RAFT...")
    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to("cuda")
    raft_model.eval()

    logger.info("Loading EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=True)

    logger.info("All models loaded.")
    return ModelBundle(
        traffic_model=traffic_model,
        vehicle_model=vehicle_model,
        lp_model=lp_model,
        depth_model=depth_model,
        depth_processor=depth_processor,
        raft_model=raft_model,
        raft_transforms=raft_transforms,
        ocr_reader=ocr_reader,
    )


# --------------------------------------------------------------------------
# Smoothing primitives
# --------------------------------------------------------------------------

class EMAFilter:
    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value


class KalmanSpeedFilter:
    """Standard 1-D Kalman filter, tuned for smooth speed readings: lower
    process noise assumes speed changes slowly; higher measurement noise
    trusts each raw reading less."""

    def __init__(self, process_noise: float = 0.5, measurement_noise: float = 8.0,
                 initial_variance: float = 10.0) -> None:
        self.Q = process_noise
        self.R = measurement_noise
        self.x: Optional[float] = None
        self.P = initial_variance

    def update(self, raw_speed: float) -> float:
        if self.x is None:
            self.x = raw_speed
            return max(raw_speed, 0.0)
        x_pred = self.x
        p_pred = self.P + self.Q
        k = p_pred / (p_pred + self.R)
        self.x = x_pred + k * (raw_speed - x_pred)
        self.P = (1.0 - k) * p_pred
        return max(self.x, 0.0)


class MedianSpeedBuffer:
    """Rolling per-track median of Kalman-filtered speeds. The median
    resists single-frame spikes better than a moving average."""

    def __init__(self, n: int = 9) -> None:
        self.n = n
        self._buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=n))

    def update(self, track_id: int, value: float) -> None:
        self._buffers[track_id].append(value)

    def get(self, track_id: int) -> Optional[float]:
        buf = self._buffers[track_id]
        return float(np.median(buf)) if buf else None


class DepthScaleEstimator:
    """Converts Depth Anything V2's relative depth map plus a per-class
    vehicle height prior into metres-per-pixel for one tracked vehicle."""

    def __init__(self, vehicle_height_map: Dict[str, float], depth_model,
                 depth_processor, depth_every_n: int) -> None:
        self.vehicle_height_m = vehicle_height_map
        self.depth_model = depth_model
        self.depth_processor = depth_processor
        self.depth_every_n = depth_every_n
        self.current_depth: Optional[np.ndarray] = None
        self.frame_counter = 0
        self.depth_ema: Dict[int, EMAFilter] = {}
        self.box_ema: Dict[int, EMAFilter] = {}
        self.scale_ema: Dict[int, EMAFilter] = {}

    def update_depth(self, frame: np.ndarray) -> None:
        self.frame_counter += 1
        if self.frame_counter % self.depth_every_n != 0:
            return
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.depth_processor(images=img_rgb, return_tensors="pt").to("cuda")
        with torch.no_grad():
            predicted_depth = self.depth_model(**inputs).predicted_depth
        d = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic", align_corners=False,
        ).squeeze()
        d_np = d.cpu().numpy()
        d_min, d_max = d_np.min(), d_np.max()
        if d_max > d_min:
            self.current_depth = (d_np - d_min) / (d_max - d_min)
        else:
            self.current_depth = np.zeros_like(d_np)

    def get_scale(self, track_id: int, cx: float, cy: float,
                  box_height_px: float, vehicle_class: str) -> float:
        if track_id not in self.box_ema:
            self.box_ema[track_id] = EMAFilter(alpha=config.EMA_ALPHA_BOX)
        smooth_box_h = max(self.box_ema[track_id].update(box_height_px), 1)

        real_h = self.vehicle_height_m.get(vehicle_class.lower(), self.vehicle_height_m["default"])
        raw_depth = 1.0
        if self.current_depth is not None:
            cy_c = min(int(cy), self.current_depth.shape[0] - 1)
            cx_c = min(int(cx), self.current_depth.shape[1] - 1)
            raw_depth = max(float(self.current_depth[cy_c, cx_c]), 0.01)

        if track_id not in self.depth_ema:
            self.depth_ema[track_id] = EMAFilter(alpha=config.EMA_ALPHA_DEPTH)
        smooth_depth = self.depth_ema[track_id].update(raw_depth)

        raw_mpp = (real_h / smooth_box_h) * smooth_depth

        if track_id not in self.scale_ema:
            self.scale_ema[track_id] = EMAFilter(alpha=config.EMA_ALPHA_SCALE)
        return self.scale_ema[track_id].update(raw_mpp)


class MotionEstimator:
    """Runs RAFT optical flow between the current frame and the frame
    `frame_interval` steps earlier, and reports box displacement from the
    resulting flow field."""

    def __init__(self, raft_model, raft_transforms, frame_interval: int = 20) -> None:
        self.raft_model = raft_model
        self.raft_transforms = raft_transforms
        self.frame_interval = frame_interval
        self.frame_counter = 0
        self.current_flow: Optional[np.ndarray] = None
        self.raft_failed_once = False
        self.frames_since_last_update = frame_interval
        self._frame_buffer: deque = deque(maxlen=frame_interval + 1)

    def update(self, frame: np.ndarray) -> None:
        self.frame_counter += 1
        self._frame_buffer.append(frame.copy())
        if self.raft_failed_once or len(self._frame_buffer) < self.frame_interval + 1:
            return
        ref_frame = self._frame_buffer[0]
        try:
            self._run_raft(ref_frame, frame)
            self.frames_since_last_update = self.frame_interval
        except Exception as exc:
            logger.warning("RAFT failed (%s). Speed estimation will stop.", exc)
            self.raft_failed_once = True
            self.current_flow = None

    def _run_raft(self, img1: np.ndarray, img2: np.ndarray) -> None:
        h, w = img1.shape[:2]
        new_h, new_w = (h // 8) * 8, (w // 8) * 8
        t1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float()
        t2 = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float()
        t1 = TF.resize(t1, size=[new_h, new_w], antialias=False)
        t2 = TF.resize(t2, size=[new_h, new_w], antialias=False)
        t1, t2 = self.raft_transforms(t1, t2)
        with torch.no_grad():
            flows = self.raft_model(t1.to("cuda"), t2.to("cuda"))
        flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()
        flow_resized = cv2.resize(flow, (w, h))
        flow_resized[..., 0] *= w / new_w
        flow_resized[..., 1] *= h / new_h
        self.current_flow = flow_resized

    def get_box_displacement(self, x1: float, y1: float, x2: float, y2: float
                              ) -> Tuple[Optional[float], Optional[float]]:
        if self.current_flow is None:
            return None, None
        h, w = self.current_flow.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        x1c = max(0, int(x1 + box_w * 0.2))
        y1c = max(0, int(y1 + box_h * 0.2))
        x2c = min(w, int(x2 - box_w * 0.2))
        y2c = min(h, int(y2 - box_h * 0.2))
        if x2c <= x1c or y2c <= y1c:
            return None, None
        region = self.current_flow[y1c:y2c, x1c:x2c]
        if region.size == 0:
            return None, None
        return float(np.median(region[..., 0])), float(np.median(region[..., 1]))


class SpeedTracker:
    """Combines depth scale and RAFT motion into a smoothed, displayable
    speed per tracked vehicle.

    Stabilisation order: noise gate -> stationary check -> Kalman filter
    -> median buffer -> display delta gate.
    """

    def __init__(self, fps: float, scale_estimator: DepthScaleEstimator,
                 motion_estimator: MotionEstimator, stationary_check_frames: int,
                 vehicle_class_names: Dict[int, str]) -> None:
        self.fps = fps
        self.estimator = scale_estimator
        self.motion = motion_estimator
        self.stationary_check_frames = stationary_check_frames
        self.vehicle_class_names = vehicle_class_names
        self.raw_pixel_traj: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self.speed_filters: Dict[int, KalmanSpeedFilter] = {}
        self.median_buffers = MedianSpeedBuffer(n=config.MEDIAN_BUFFER_FRAMES)
        self.display_speeds: Dict[int, float] = {}
        self.internal_speeds: Dict[int, float] = {}
        self.scale_methods: Dict[int, str] = {}

    def _is_stationary(self, track_id: int, box_height_px: Optional[float]) -> bool:
        traj = self.raw_pixel_traj.get(track_id)
        if traj is None or len(traj) < self.stationary_check_frames:
            return False
        recent = list(traj)[-self.stationary_check_frames:]
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        spread = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
        threshold = config.STATIONARY_PIXEL_THRESHOLD
        if box_height_px is not None:
            threshold = max(threshold, box_height_px * config.STATIONARY_RELATIVE_FRACTION)
        return spread < threshold

    def update(self, detections, frame: np.ndarray, frame_number: int) -> None:
        self.estimator.update_depth(frame)
        self.motion.update(frame)

        if detections.tracker_id is None:
            return

        for i, track_id in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
            box_height_px = float(y2 - y1)

            self.raw_pixel_traj[track_id].append((frame_number, cx, cy))
            stationary = self._is_stationary(track_id, box_height_px)

            speed_raw: Optional[float] = 0.0 if stationary else None

            if not stationary and not self.motion.raft_failed_once:
                elapsed_s = max(self.motion.frames_since_last_update, 1) / self.fps
                cls_name = self.vehicle_class_names.get(int(detections.class_id[i]), "default")
                mpp = self.estimator.get_scale(track_id, cx, cy, box_height_px, cls_name)
                dx_px, dy_px = self.motion.get_box_displacement(x1, y1, x2, y2)

                if dx_px is not None and mpp is not None and elapsed_s > 0:
                    flow_mag = float(np.hypot(dx_px, dy_px))
                    if flow_mag >= config.MIN_FLOW_PIXELS:
                        dist_m = flow_mag * mpp
                        speed_raw = min(max((dist_m / elapsed_s) * 3.6, 0.0), 250.0)
                        self.scale_methods[track_id] = "depth_v2+raft"

            if speed_raw is None:
                continue

            if track_id not in self.speed_filters:
                self.speed_filters[track_id] = KalmanSpeedFilter(
                    process_noise=config.KALMAN_PROCESS_NOISE,
                    measurement_noise=config.KALMAN_MEASUREMENT_NOISE,
                    initial_variance=config.KALMAN_INITIAL_VARIANCE,
                )
            kalman_speed = self.speed_filters[track_id].update(speed_raw)
            self.internal_speeds[track_id] = round(kalman_speed, 1)

            self.median_buffers.update(track_id, kalman_speed)
            median_speed = self.median_buffers.get(track_id)

            prev_display = self.display_speeds.get(track_id)
            if prev_display is None or abs(median_speed - prev_display) >= config.SPEED_DISPLAY_DELTA_KMH:
                self.display_speeds[track_id] = round(median_speed, 1)

    def get_speed(self, track_id: int) -> Optional[float]:
        return self.display_speeds.get(track_id)

    def get_internal_speed(self, track_id: int) -> Optional[float]:
        return self.internal_speeds.get(track_id)

    def get_method(self, track_id: int) -> str:
        return self.scale_methods.get(track_id, "unknown")


# --------------------------------------------------------------------------
# Plate buffering and OCR (deferred, majority vote)
# --------------------------------------------------------------------------

class PlateBuffer:
    """Per-job replacement for the notebook's global plate_crop_buffer
    dict. Keeps the top `max_crops` highest-confidence plate crops per
    track id, plus the single best vehicle snapshot for the offender log.
    """

    def __init__(self, max_crops: int = 5) -> None:
        self.max_crops = max_crops
        self._entries: Dict[int, dict] = defaultdict(lambda: {
            "crops": [], "best_snapshot": None, "best_frame": 0, "best_conf": 0.0,
        })

    def update(self, track_id: int, plate_crop: Optional[np.ndarray], plate_conf: float,
               vehicle_snapshot: np.ndarray, frame_number: int) -> None:
        if plate_crop is None or plate_crop.size == 0:
            return
        entry = self._entries[track_id]
        entry["crops"].append((plate_conf, plate_crop.copy(), frame_number))
        entry["crops"].sort(key=lambda c: c[0], reverse=True)
        entry["crops"] = entry["crops"][: self.max_crops]
        if plate_conf > entry["best_conf"]:
            entry["best_conf"] = plate_conf
            entry["best_snapshot"] = vehicle_snapshot.copy()
            entry["best_frame"] = frame_number

    def get(self, track_id: int) -> Optional[dict]:
        return self._entries.get(track_id)

    def save_snapshot(self, track_id: int, label_str: str, output_dir: str) -> Optional[str]:
        entry = self._entries.get(track_id)
        if entry is None or entry["best_snapshot"] is None:
            return None
        snap_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        safe_label = label_str.replace(" ", "_").replace(",", "_")
        filename = f"track{track_id}_{safe_label}_f{entry['best_frame']}.jpg"
        cv2.imwrite(os.path.join(snap_dir, filename), entry["best_snapshot"])
        return filename


def preprocess_plate_crop(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Upscales, contrast-boosts, and binarises a plate crop before OCR."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = max(100 / h, 1.0)
    if scale > 1.0:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def ocr_plate_majority_vote(entry: Optional[dict], ocr_reader, conf_threshold: float) -> str:
    """Runs EasyOCR on every buffered crop for one track and returns the
    most common reading. Falls back to "UNREADABLE" if nothing is legible."""
    if entry is None or not entry["crops"]:
        return "UNREADABLE"

    readings = []
    for _plate_conf, crop, _frame in entry["crops"]:
        processed = preprocess_plate_crop(crop)
        if processed is None:
            continue
        try:
            results = ocr_reader.readtext(processed, detail=1, paragraph=False)
        except Exception:
            logger.exception("EasyOCR read failed")
            continue
        raw_text = "".join(
            text.strip() + " " for _bbox, text, ocr_conf in results if ocr_conf > conf_threshold
        )
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip())
        if cleaned:
            readings.append(cleaned)

    if not readings:
        return "UNREADABLE"
    return Counter(readings).most_common(1)[0][0]


# --------------------------------------------------------------------------
# Traffic light resolution (detector + manual override)
# --------------------------------------------------------------------------

def resolve_traffic_state(detected_label: str, override: TrafficLightOverride) -> Tuple[str, bool]:
    """Combines the detector's output with any manual override.

    Returns (effective_label, is_red). "auto" defers to the detector;
    any other override value wins regardless of what the detector saw.
    """
    manual = override.get()
    if manual == "auto":
        label = detected_label
    elif manual == "amber":
        label = config.AMBER_DETECTOR_LABEL
    else:
        label = manual
    return label, "red" in label.lower()


# --------------------------------------------------------------------------
# Depth preview image
# --------------------------------------------------------------------------

def save_depth_preview(depth_norm: np.ndarray, output_dir: str) -> str:
    """Colourises the latest depth map and writes it to disk so the UI
    can poll for a preview image."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap("inferno")
    coloured = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)
    bgr = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, "depth_preview.jpg")
    cv2.imwrite(path, bgr)
    return path


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------

def _draw_label(annotated: np.ndarray, text: str, x1: int, y1: int,
                 colour: Tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_plate_box(annotated: np.ndarray, vx1: int, vy1: int,
                     lx1: int, ly1: int, lx2: int, ly2: int, conf: float) -> None:
    ax1, ay1, ax2, ay2 = vx1 + lx1, vy1 + ly1, vx1 + lx2, vy1 + ly2
    cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 255, 255), 2)
    cv2.putText(annotated, f"LP {conf:.2f}", (ax1, ay1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run_pipeline(job: JobRecord, models: ModelBundle, light_override: TrafficLightOverride) -> None:
    """Processes job.video_path end to end: detection, tracking, speed
    estimation, and offender logging. Mutates `job` in place so the API
    layer can report progress while this runs in a background thread.
    """
    try:
        job.status = "processing"

        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {job.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_settings = config.frame_settings_for_fps(fps)

        # ROI: middle band of the frame, same convention as the notebook -
        # excludes the sky/horizon and the vehicle bonnet at the bottom.
        quarter_height = height / 4
        roi_x1, roi_x2 = 0, width
        roi_y1 = int(quarter_height * 1.65)
        roi_y2 = int(quarter_height * 3.95)

        # A vehicle becomes a red-light candidate only once its box bottom
        # reaches this line - not on every frame it happens to be visible
        # while the light is red.
        exit_threshold = roi_y2 - (height * config.ROI_EXIT_MARGIN_FRACTION)
        track_prev_bottom: Dict[int, float] = {}

        scale_estimator = DepthScaleEstimator(
            vehicle_height_map=config.VEHICLE_HEIGHT_M,
            depth_model=models.depth_model,
            depth_processor=models.depth_processor,
            depth_every_n=frame_settings.depth_every_n_frames,
        )
        motion_estimator = MotionEstimator(
            raft_model=models.raft_model,
            raft_transforms=models.raft_transforms,
            frame_interval=frame_settings.frame_interval,
        )
        speed_tracker = SpeedTracker(
            fps=fps,
            scale_estimator=scale_estimator,
            motion_estimator=motion_estimator,
            stationary_check_frames=frame_settings.stationary_check_frames,
            vehicle_class_names=models.vehicle_model.names,
        )

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, smart_position=True)

        plate_buffer = PlateBuffer(max_crops=config.MAX_PLATE_CROPS)
        vehicle_type_map: Dict[int, str] = {}
        offender_flags: Dict[int, set] = {}

        output_video_path = os.path.join(job.output_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_number = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            job.current_frame = frame_number
            annotated = frame.copy()
            roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # 1. Traffic light detection (full frame, so lights outside
            #    the vehicle ROI are still caught).
            detected_label = "unknown"
            tl_results = models.traffic_model.predict(
                frame, conf=config.CONF_THRESHOLD, device=0, verbose=False
            )
            for box in tl_results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = models.traffic_model.names[cls_id]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_label = cls_name
                colour = TL_COLOURS.get(cls_name.lower().split("_")[0], (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                _draw_label(annotated, f"TL: {cls_name} {conf_val:.2f}", x1, y1, colour)

            job.traffic_light_detected = detected_label
            effective_label, red_light_this_frame = resolve_traffic_state(detected_label, light_override)

            # 2. Vehicle detection + ByteTrack
            vehicle_results = models.vehicle_model.track(
                roi_crop, conf=config.CONF_THRESHOLD, iou=config.IOU_THRESHOLD,
                tracker="bytetrack.yaml", device=0, persist=True, verbose=False,
            )

            if vehicle_results[0].boxes.id is not None:
                detections = sv.Detections.from_ultralytics(vehicle_results[0])
                detections.xyxy[:, 0] += roi_x1
                detections.xyxy[:, 1] += roi_y1
                detections.xyxy[:, 2] += roi_x1
                detections.xyxy[:, 3] += roi_y1

                labels = []
                keep_mask = []

                for i, track_id in enumerate(detections.tracker_id):
                    cls_name = models.vehicle_model.names[detections.class_id[i]]
                    if cls_name.lower() not in config.VEHICLE_ALLOWED_CLASSES:
                        keep_mask.append(False)
                        continue
                    keep_mask.append(True)

                    vehicle_type_map[track_id] = cls_name

                    x1, y1, x2, y2 = map(int, detections.xyxy[i])
                    vehicle_snapshot = frame[y1:y2, x1:x2].copy()

                    # License plate detection inside the vehicle crop
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size > 0:
                        lp_results = models.lp_model.predict(
                            vehicle_crop, conf=config.CONF_THRESHOLD, device=0, verbose=False
                        )
                        for lp_box in lp_results[0].boxes:
                            lx1, ly1, lx2, ly2 = map(int, lp_box.xyxy[0])
                            lp_conf = float(lp_box.conf[0])
                            plate_crop = vehicle_crop[ly1:ly2, lx1:lx2]
                            plate_buffer.update(track_id, plate_crop, lp_conf, vehicle_snapshot, frame_number)
                            _draw_plate_box(annotated, x1, y1, lx1, ly1, lx2, ly2, lp_conf)

                    # Crossing check: has this vehicle's box just reached
                    # the bottom of the tracking zone? Compared against its
                    # own previous frame, so this fires once, not on every
                    # frame the vehicle happens to be near the line.
                    prev_bottom = track_prev_bottom.get(track_id)
                    crossed_exit_line = prev_bottom is not None and prev_bottom < exit_threshold <= y2
                    track_prev_bottom[track_id] = y2

                    # Violation checks
                    speed_internal = speed_tracker.get_internal_speed(track_id)
                    if crossed_exit_line:
                        job.crossing_events.append(
                            CrossingEvent(track_id=track_id, frame_number=frame_number, detected_light=detected_label)
                        )
                        if red_light_this_frame:
                            offender_flags.setdefault(track_id, set()).add("red_light")
                    if speed_internal is not None and speed_internal > config.SPEED_LIMIT_KMH:
                        offender_flags.setdefault(track_id, set()).add("speeding")

                    speed_display = speed_tracker.get_speed(track_id)
                    label = f"{track_id}|{speed_display} km/h" if speed_display is not None else f"{track_id}|--"
                    labels.append(label)

                keep_mask = np.array(keep_mask, dtype=bool)
                filtered_detections = (
                    detections[keep_mask] if keep_mask.any()
                    else detections[np.zeros(len(detections), dtype=bool)]
                )

                speed_tracker.update(filtered_detections, frame, frame_number)

                if len(filtered_detections) > 0:
                    annotated = box_annotator.annotate(scene=annotated, detections=filtered_detections)
                    annotated = label_annotator.annotate(
                        scene=annotated, detections=filtered_detections, labels=labels
                    )

            # HUD banner shows the EFFECTIVE state: manual override if one
            # is set, otherwise whatever the detector saw this frame.
            banner_colour = TL_COLOURS.get(effective_label.lower().split("_")[0], (80, 80, 80))
            cv2.rectangle(annotated, (0, 0), (300, 28), banner_colour, -1)
            source_tag = "MANUAL" if light_override.get() != "auto" else "AUTO"
            cv2.putText(
                annotated, f"Traffic light: {effective_label.upper()} ({source_tag})",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )

            writer.write(annotated)

            if scale_estimator.current_depth is not None and frame_number % frame_settings.depth_every_n_frames == 0:
                job.depth_preview_path = save_depth_preview(scale_estimator.current_depth, job.output_dir)

        cap.release()
        writer.release()
        job.annotated_video_path = output_video_path

        # Postprocessing OCR - same two-phase structure as the notebook:
        # plate reading only happens once a vehicle's full set of crops
        # is known, using majority vote across the buffered crops.
        #
        # We build a summary for every track that's relevant to a possible
        # offence: anything that crossed the stop line (a red-light
        # candidate under some scenario) or was ever flagged for speeding
        # (light-independent). That summary, plus the raw crossing events
        # above, is everything `recompute_offenders` needs to re-judge this
        # video under a different light scenario without touching the GPU.
        job.status = "postprocessing"
        relevant_track_ids = {event.track_id for event in job.crossing_events} | set(offender_flags.keys())

        for track_id in relevant_track_ids:
            vtype = vehicle_type_map.get(track_id, "unknown")
            speed_val = speed_tracker.get_speed(track_id)
            method = speed_tracker.get_method(track_id)
            violations = offender_flags.get(track_id, set())

            entry = plate_buffer.get(track_id)
            plate_text = ocr_plate_majority_vote(entry, models.ocr_reader, config.OCR_CONF_THRESHOLD)
            snapshot_filename = plate_buffer.save_snapshot(track_id, f"{vtype}_{track_id}", job.output_dir)
            snap_frame = entry["best_frame"] if entry else 0

            job.track_summaries[track_id] = TrackSummary(
                vehicle_type=vtype,
                speed_kmh=speed_val,
                speed_method=method,
                plate_number=plate_text,
                snapshot_filename=snapshot_filename,
                timestamp_frame=snap_frame,
                speeding="speeding" in violations,
            )

        # The initial offender log reflects exactly what happened live,
        # including any manual overrides toggled during this run.
        offenders = []
        for track_id, violations in offender_flags.items():
            summary = job.track_summaries[track_id]
            offenders.append(OffenderEntry(
                track_id=track_id,
                vehicle_type=summary.vehicle_type,
                plate_number=summary.plate_number,
                speed_kmh=summary.speed_kmh,
                speed_method=summary.speed_method,
                violation_type=",".join(sorted(violations)),
                snapshot_filename=summary.snapshot_filename,
                timestamp_frame=summary.timestamp_frame,
            ))

        with job.lock:
            job.offenders = offenders
        job.status = "done"

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job.job_id)
        job.status = "error"
        job.error_message = str(exc)


def recompute_offenders(job: JobRecord, scenario: str) -> List[OffenderEntry]:
    """Re-judges an already-processed video under a different traffic
    light scenario, using only the crossing events and track summaries
    `run_pipeline` already cached. No GPU work happens here, which is why
    switching scenarios on a finished job is instant.

    `scenario` is "auto" (use whatever the detector actually saw at each
    crossing) or a forced colour ("red"/"amber"/"green") applied to the
    whole video.
    """
    violations_by_track: Dict[int, set] = defaultdict(set)
    for event in job.crossing_events:
        effective_label = event.detected_light if scenario == "auto" else scenario
        if "red" in effective_label.lower():
            violations_by_track[event.track_id].add("red_light")

    offenders = []
    for track_id, summary in job.track_summaries.items():
        violations = violations_by_track.get(track_id, set())
        if summary.speeding:
            violations.add("speeding")
        if not violations:
            continue
        offenders.append(OffenderEntry(
            track_id=track_id,
            vehicle_type=summary.vehicle_type,
            plate_number=summary.plate_number,
            speed_kmh=summary.speed_kmh,
            speed_method=summary.speed_method,
            violation_type=",".join(sorted(violations)),
            snapshot_filename=summary.snapshot_filename,
            timestamp_frame=summary.timestamp_frame,
        ))
    return offenders
