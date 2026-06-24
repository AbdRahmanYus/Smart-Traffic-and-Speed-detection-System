# Smart Traffic Surveillance - FastAPI UI

A small dashboard for the speed-detection and red-light-violation pipeline
from `Namikaze_Sama.ipynb`. Upload a video, watch it process, and check the
offender log, the depth view, and a manual traffic light control.

## What this is

- `config.py` - all the thresholds and constants from the notebook's config cell, unchanged.
- `pipeline.py` - the detection/tracking/speed-estimation logic, ported from the notebook's classes (EMAFilter, KalmanSpeedFilter, MedianSpeedBuffer, DepthScaleEstimator, MotionEstimator, SpeedTracker) plus the main loop and postprocessing OCR step. Behaviour is unchanged; only the bookkeeping is new (per-job state instead of notebook globals).
- `main.py` - the FastAPI routes.
- `jobs.py` - a plain in-memory job store. No database, no task queue - this is a single-user tool.
- `traffic_light.py` - the manual override flag shared across the app.
- `templates/`, `static/` - the dashboard page, plain HTML/CSS/JS.

## How red-light violations are decided

A vehicle is only flagged as a red-light violator at the moment its box
reaches the bottom of the tracking zone (`ROI_Y2`) - not on every frame
it happens to be visible while the light is red. This matches how a real
red-light camera works: it judges the vehicle at the stop line, not for
its entire time in view.

## Replaying a different light scenario without reprocessing

The GPU work (detection, tracking, speed, depth, plates) only needs to
run once per video. Separately, the app caches, for every vehicle that
reached the stop line, what the light actually was at that moment. Once
a video has finished processing, switching the traffic light buttons
re-judges the whole video under that scenario - "what if the light had
been red the entire time?" - instantly, with no GPU involved. Each
upload is labelled Session 1, Session 2, and so on, so you can keep
track of which run you're replaying.

While a video is still processing, the same buttons instead act as a
live override - useful for demoing the response to a forced red light
in real time as it's being detected.

## How the traffic light widget works

The widget has four states: **Auto**, **Red**, **Amber**, **Green**.

- **Auto** - the existing traffic light detector decides, frame by frame, exactly as in the notebook.
- **Red / Amber / Green** - your choice overrides the detector immediately. Every frame's red-light-violation check now reacts to your override instead of the detector, until you switch back to Auto.

This is one shared switch for the whole app, not per-video - fine for a single
person operating the dashboard, demoing violation scenarios on demand without
needing footage where the light happens to change at the right moment.

## Running locally

Only do this if you have a CUDA GPU - the pipeline runs three YOLO models,
EasyOCR, RAFT-large, and Depth Anything V2 per frame, and is not designed to
run on CPU in any reasonable time.

```bash
pip install -r requirements.txt
export PROJECT_DIR="/path/to/your/weights/folder"   # must contain weights/*.pt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Running inside Google Colab (recommended)

This is the practical option: Colab gives you a free T4 GPU, and it's where
your weights already live on Drive. Run the FastAPI server as a background
process inside the notebook, then use Colab's own port-forwarding to get a
browser URL - no ngrok account or auth token needed.

1. Mount Drive and upload this `traffic_ui` folder into your Colab session (or `git clone` it if you push it to a repo).

2. Install dependencies:

```python
!pip install -r traffic_ui/requirements.txt -q
```

3. Point the app at your existing weights folder on Drive (same structure the notebook already saves to):

```python
import os
os.environ["PROJECT_DIR"] = "/content/drive/MyDrive/Smart traffic and speed detection system"
```

4. Start the server in the background and open a tunnel through Colab's own proxy:

```python
import subprocess, threading

def run_server():
    subprocess.run(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd="traffic_ui",
    )

threading.Thread(target=run_server, daemon=True).start()

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8000)"))
```

5. Open the printed URL in a new tab. That's your dashboard, running on the Colab GPU.

If you'd rather have a stable link you can share outside your own Colab
session (e.g. to show someone else without giving them your notebook), use
`pyngrok` instead - it needs a free ngrok account and auth token, which is
the only reason it isn't the default here:

```python
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(8000)
print(public_url)
```

## Things worth knowing before you rely on this

- **Speed, not real-time.** Three YOLO passes plus EasyOCR plus RAFT-large plus Depth Anything V2, all per frame, will run well below real-time even on a T4. Test with short clips first.
- **Jobs are in-memory.** Restarting the server loses job history. Annotated videos and snapshots on disk survive, but the dashboard won't know about old job ids any more.
- **One traffic light override for the whole app.** If you process two videos at once, they share the same manual state.
- **Total frame count can be wrong** for some video containers (cv2 sometimes misreports it), which would just leave the progress bar showing 0% while frames are clearly advancing in the status text - not a sign anything's actually broken.
