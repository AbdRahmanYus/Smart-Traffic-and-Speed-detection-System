#  Smart Traffic & Speed Detection System

A computer-vision pipeline for automatic vehicle speed estimation, red-light surveillance, and licence-plate-based offender logging. Built and trained entirely on a free Google Colab T4 GPU.

Three custom-trained YOLOv11 models work alongside RAFT optical flow, Depth Anything V2, ByteTrack, and EasyOCR to do, on every video frame, what would otherwise take three separate roadside officers: watch for red-light runners, measure speed, and read number plates.

## Key Features

- **Calibration-free speed estimation** — RAFT optical flow + Depth Anything V2 convert raw pixel motion into real km/h, on any camera angle, with no manual setup. A Kalman filter and rolling median keep the displayed reading stable.
- **Automatic red-light detection** — a dedicated, trained YOLOv11 classifier reads the traffic light itself (red / green / yellow). No HSV colour heuristics, no manually drawn stop line.
- **Crossing-based violations** — a vehicle is only flagged once its box reaches the bottom of the tracking zone, the same way a real red-light camera judges a vehicle at the stop line, not for however long it happens to be visible.
- **Reliable licence-plate OCR** — up to 5 of the sharpest buffered plate crops per vehicle are read after the fact and resolved by majority vote, instead of trusting a single, possibly blurry, live frame.
- **Offender logging** — one CSV row per offending vehicle: track ID, vehicle type, plate number, speed, violation type, and a snapshot.
- **Optional FastAPI dashboard** (`traffic_ui/`) — upload a video, watch live progress, manually override the traffic light, and instantly replay different light scenarios on an already-processed video without rerunning the GPU pipeline.

## How It Works

```
Input frame
  -> Traffic-light model (full frame)
  -> Vehicle model + ByteTrack (ROI crop, middle 50% of frame height)
  -> Licence-plate model (per vehicle crop)
  -> RAFT flow + Depth Anything V2 scale -> speed (km/h)
  -> Violation flag accumulation (stop-line crossing + speeding)

Post-loop
  -> EasyOCR majority-vote plate reading
  -> offender_log.csv + snapshots
```

## Repository Structure

```
.
├── Smart_surveillance_and_speed_detection_AbdRahmanYus.ipynb     # Main Colab notebook: dataset prep, training, full inference pipeline
├── traffic_ui/             # Optional FastAPI dashboard for running the pipeline outside the notebook
│   ├── main.py             # API routes
│   ├── pipeline.py         # Detection, tracking, speed estimation, OCR (ported from the notebook)
│   ├── config.py           # All pipeline constants and thresholds
│   ├── jobs.py             # In-memory job tracking
│   ├── traffic_light.py    # Manual traffic-light override state
│   ├── schemas.py          # API request/response models
│   ├── templates/, static/ # Dashboard frontend
│   ├── requirements.txt
│   └── README.md           # Full setup, including the Colab launch steps
└── README.md
```

## Getting Started

### Option 1 — Run the notebook in Colab

1. Open `Smart_surveillance_and_speed_detection_AbdRahmanYus.ipynb` in Google Colab.
2. `Runtime > Change runtime type > T4 GPU`.
3. Mount Google Drive and point the notebook at your dataset and weights paths.
4. Run the cells top to bottom: dependency install, dataset download, model training, then the main inference loop on your own video.

### Option 2 — Run the dashboard

The dashboard wraps the same pipeline in a web UI so you don't have to operate the notebook directly. It still needs a GPU, so it's meant to run inside Colab — see [`traffic_ui/README.md`](traffic_ui/README.md) for the full walkthrough, including how to get a public URL straight out of Colab with no extra accounts.

Quick start, assuming a GPU is already available:

```bash
cd traffic_ui
pip install -r requirements.txt
export PROJECT_DIR="/path/to/your/weights/folder"
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Models & Libraries

| Component | Library / Model |
|---|---|
| Object detection (traffic light, vehicle, plate) | Ultralytics YOLOv11 (3 custom-trained models) |
| Vehicle tracking | ByteTrack (via `supervision`) |
| Optical flow | RAFT-large (`torchvision`) |
| Depth estimation | Depth Anything V2 - Small (HuggingFace `transformers`) |
| Licence plate OCR | EasyOCR (CRAFT + CRNN) |
| Dashboard | FastAPI, Jinja2, vanilla JS |

## Model Performance

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Traffic Light Detection | 95.65% | 94.93% | 97.91% | 72.11% |
| Vehicle Type Detection | 91.87% | 90.90% | 94.83% | 63.13% |
| Licence Plate Detection | 94.31% | 84.37% | 92.01% | 63.40% |

Object detectors don't report a single classification-style "accuracy" — precision, recall, and mAP together are the detection equivalent. mAP50-95 is noticeably lower than mAP50 for every model because it only rewards near-pixel-perfect bounding boxes, rather than the more forgiving overlap threshold mAP50 uses.

## Known Limitations

- Demo speed limit (1 km/h) — must be set to the road's real legal limit before any real deployment.
- Roboflow API key is hard-coded in the dataset-download cell — rotate it before sharing the notebook publicly.
- Depth Anything V2 produces *relative* depth, which can drift slightly frame to frame — only an approximate metric bridge, not absolute distance.
- RAFT runs on the full frame even though speed estimation only uses the middle ROI, wasting some GPU compute.
- The licence-plate dataset may not fully generalise to Nigerian plate formats.
- `timestamp_real` is always `None` — only the frame number is logged, not wall-clock time.
- A vehicle that's lost and re-acquired by ByteTrack gets a new track ID and can be logged twice — there's no position-based deduplication yet.
- The notebook itself is Colab-only; the FastAPI dashboard is the first step toward a standalone deployment.

## Acknowledgements

Built after studying two reference projects' approaches and documented limitations:

- [jayy-agu/SpeedDetection](https://github.com/jayy-agu/SpeedDetection) — homography-based speed estimation
- [Nabeel-99/smart-surveillance-system](https://github.com/Nabeel-99/smart-surveillance-system) — HSV-based red-light surveillance
- joshfatoye0011-bit

## Author

**AbdulRahman Yusuf


## License

`License`
