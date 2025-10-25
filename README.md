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

## Repository structure
.
├── app.py # Flask control panel 
├── people_count.py # Memryx YOLOv7-tiny + tracker
├── recognize_faces.py # Frame glasses -> on-device face recognition
├── encode_face.py # Build encodings.pickle from dataset/
├── templates/
│ └── index.html # Web UI
├── dataset/ # <person_name>/<photos> 
├── models/ # *.dfp, *.onnx 
├── security.mp4 # Optional test video 
└── yolov7.py # Memryx YOLOv7 helper




