## Found (Memryx Hackathon)

**Short abstract:** On-device AI for crowd safety. We use AR glasses + fixed webcams to (1) count people at chokepoints for crowd control and (2) recognize lost individuals to help reunite lost people, all processed on the edge device the glasses are connected to (Memryx acceleration), keeping latency low and data local.

## Why this matters (hackathon fit)
- **Edge-first:** Runs on the device with **Memryx MXA** acceleration for real-time detection without shipping video to the cloud.
- **Safety impact:** Live counts for congestion detection and fast **“find my person”** workflows during large events.
- **AR workflow:** Brilliant Labs **Frame** glasses show recognized names on-device; fixed webcams feed people counts into a simple UI dashboard.

---

## Features

- **People counting (YOLOv7-tiny on Memryx):**
  - Real-time person detection + lightweight tracker with unique IDs.
  - Works from a webcam (`/dev/video0` / index `0`) or a video file (e.g. `security.mp4`).

- **Face enrollment + recognition:**
  - Drop photos in `dataset/<person_name>/...`.
  - `encode_face.py` builds `encodings.pickle`.
  - Glasses app (`recognize_faces.py`) compares live frames to known faces and displays the recognized name on the **Frame**.

- **Flask UI**
  - Buttons to **Start/Stop People Count** and **Start/Stop Recognize Faces**.
  - Upload a photo + name, then **Encode**.
  - A **Safety Log** stream shows system messages and script output.

---

Got it—here’s a clean, drop-in **“How it works”** section you can paste into your README. It assumes you already have Memryx hardware + software set up.

---

## How it works (assumes Memryx HW/SW is ready)

### Assumptions

* You have a Memryx accelerator connected and the Memryx SDK (Python package) installed.
* YOLOv7-tiny DFP and post-processing ONNX are available at:

  * `models/YOLO_v7_tiny_416_416_3_onnx.dfp`
  * `models/YOLO_v7_tiny_416_416_3_onnx_post.onnx`
* You’re using Brilliant Labs **Frame** AR glasses for on-device face recognition (the AI runs on the computer the glasses are connected to).

### Python libraries

Install these (plus any platform prerequisites for `face-recognition`/`dlib`):

```bash
pip install flask opencv-python numpy face-recognition memryx frame-sdk
```

> If `face-recognition` fails to build, install a prebuilt `dlib` wheel for your OS/Python, then reinstall `face-recognition`.

---

### Component overview

#### 1) People counting (Memryx + YOLOv7-tiny)

* **Capture:** OpenCV grabs frames from a source (`/dev/video0` or `0` by default; can be a file like `security.mp4`).
* **Accelerated inference:** Frames go to `memryx.MultiStreamAsyncAccl` using the YOLOv7-tiny `.dfp`.
* **Post-processing:** The `.onnx` post model converts raw outputs to boxes/scores/classes; we filter for *person*.
* **Tracking & count:** A lightweight IoU/ByteTrack-lite style tracker assigns IDs to reduce double-counting; we compute a live **People:** count.
* **Display & logs:** The annotated stream (boxes, IDs, FPS, count) renders in an OpenCV window; progress is also logged.

Run directly (examples):

```bash
# Webcam
python people_count.py \
  --dfp models/YOLO_v7_tiny_416_416_3_onnx.dfp \
  --postmodel models/YOLO_v7_tiny_416_416_3_onnx_post.onnx \
  --video_paths /dev/video0

# Video file
python people_count.py \
  --dfp models/YOLO_v7_tiny_416_416_3_onnx.dfp \
  --postmodel models/YOLO_v7_tiny_416_416_3_onnx_post.onnx \
  --video_paths security.mp4
```

> Make sure `yolov7.py` (the Memryx example helper) is in the same folder so its preprocess/postprocess utilities are importable.

#### 2) Face enrollment (build the database)

* **Add photos:** Place images under `dataset/<person_name>/...` or use the Flask UI’s upload form.
* **Encode:** `encode_face.py` scans `dataset/`, detects faces, computes embeddings with `face_recognition`, and writes `encodings.pickle`.
* **Result:** A compact on-device face database (no cloud).

Create/update encodings:

```bash
python encode_face.py
```

#### 3) Face recognition (on AR glasses)

* **Load DB:** `recognize_faces.py` loads `encodings.pickle`.
* **Stream frames:** Captures images from Brilliant Labs **Frame** via `frame-sdk`.
* **Recognize:** Runs face detection + embeddings with `face_recognition`, compares to known encodings, picks the best match.
* **Display on device:** Sends the recognized name to the **Frame** display so the steward sees it in AR; also logs events locally.

Run:

```bash
python recognize_faces.py
```

> All recognition happens on the computer the glasses are tethered to—**AI stays on device** for low latency and privacy.

#### 4) Flask control panel (orchestration)

* `app.py` exposes a minimal dashboard (`templates/index.html`) with buttons to:

  * **Start/Stop People Count** (spawns/kills `people_count.py` as a subprocess)
  * **Start/Stop Recognize Faces** (spawns/kills `recognize_faces.py`)
  * **Upload Photo** → saved to `dataset/<name>/`
  * **Encode** → runs `encode_face.py`
* A rolling **Safety Log** shows stdout from each subprocess for quick debugging and auditability.

Run the panel:

```bash
python app.py
# open http://localhost:8000
```

---

### Data/Control flow at a glance

```
Webcam/Video ──► Memryx YOLOv7-tiny ──► Postprocess ──► Track ──► Count/Overlay ──► Display + Log
                         ▲
                         │  (DFP on device, no cloud)
                         │
AR Frame ──► Frames ──► face_recognition ──► Compare to encodings.pickle ──► Name on glasses + Log

Flask UI ──► Start/Stop scripts + Upload photos + Trigger encode ──► Safety Log (stdout)
```

**Why this fits the hackathon:** everything runs on the device (Memryx-accelerated), enabling real-time crowd metrics and rapid “find my person” assistance with strong privacy.

