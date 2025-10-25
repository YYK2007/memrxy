import os, sys, signal, subprocess, threading
from datetime import datetime
from collections import deque
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ---------- paths ----------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PEOPLE_SCRIPT = os.path.join(BASE_DIR, "people_count.py")
FACE_SCRIPT   = os.path.join(BASE_DIR, "recognize_faces.py")
ENCODE_SCRIPT = os.path.join(BASE_DIR, "encode_face.py")

DFP_PATH      = os.path.join(BASE_DIR, "models/YOLO_v7_tiny_416_416_3_onnx.dfp")
POST_PATH     = os.path.join(BASE_DIR, "models/YOLO_v7_tiny_416_416_3_onnx_post.onnx")
VIDEO_SRC     = "/dev/video0"   # change if needed

DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
ALLOWED_EXTS  = {"jpg","jpeg","png"}
os.makedirs(DATASET_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
log_lines = deque(maxlen=400)

def log(msg: str):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    app.logger.info(line)
    log_lines.appendleft(line)

# ---------- simple process manager ----------
class ProcRunner:
    def __init__(self, name: str, cmd: list[str]):
        self.name = name
        self.cmd  = cmd
        self.proc: subprocess.Popen | None = None

    def start(self):
        if self.running():
            return False, "already running"
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,      # make a process group we can kill
                env=os.environ.copy(),
            )
            threading.Thread(target=self._pump, daemon=True).start()
            log(f"{self.name} started: {' '.join(self.cmd)}")
            return True, ""
        except Exception as e:
            log(f"{self.name} failed to start: {e}")
            return False, str(e)

    def _pump(self):
        assert self.proc is not None
        if self.proc.stdout:
            for line in self.proc.stdout:
                log(f"[{self.name}] {line.rstrip()}")
        code = self.proc.wait()
        log(f"{self.name} exited with code {code}")

    def stop(self):
        if not self.running():
            return False, "not running"
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
            log(f"{self.name} stopped")
            return True, ""
        except Exception as e:
            log(f"{self.name} failed to stop: {e}")
            return False, str(e)

    def running(self):
        return self.proc is not None and self.proc.poll() is None

# ---------- command lines ----------
people_cmd = [
    sys.executable, PEOPLE_SCRIPT,
    "--dfp", DFP_PATH,
    "--postmodel", POST_PATH,
    "--video_paths", VIDEO_SRC
]
PEOPLE = ProcRunner("people_count", people_cmd)
FACE   = ProcRunner("recognize_faces", [sys.executable, FACE_SCRIPT])

# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html",
                           people_running=PEOPLE.running(),
                           face_running=FACE.running())

@app.post("/start_people")
def start_people():
    ok, err = PEOPLE.start()
    return (jsonify(ok=ok) if ok else (jsonify(ok=False, error=err), 500))

@app.post("/stop_people")
def stop_people():
    ok, err = PEOPLE.stop()
    return (jsonify(ok=ok) if ok else (jsonify(ok=False, error=err), 400))

@app.post("/start_face")
def start_face():
    ok, err = FACE.start()
    return (jsonify(ok=ok) if ok else (jsonify(ok=False, error=err), 500))

@app.post("/stop_face")
def stop_face():
    ok, err = FACE.stop()
    return (jsonify(ok=ok) if ok else (jsonify(ok=False, error=err), 400))

@app.get("/safety_log")
def safety_log():
    return jsonify(list(log_lines)[:200])

@app.post("/upload_pic")
def upload_pic():
    name = (request.form.get("name") or "").strip()
    if not name: return jsonify(ok=False, error="Missing 'name'"), 400
    if "file" not in request.files: return jsonify(ok=False, error="Missing file"), 400
    f = request.files["file"]
    if not f.filename: return jsonify(ok=False, error="Empty filename"), 400
    ext = f.filename.rsplit(".",1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXTS:
        return jsonify(ok=False, error=f"Unsupported file .{ext} (use jpg/jpeg/png)"), 400
    folder = os.path.join(DATASET_DIR, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, secure_filename(f.filename))
    f.save(path)
    log(f"Uploaded photo for {name}: {os.path.basename(path)}")
    return jsonify(ok=True, path=path)

@app.post("/encode")
def encode_pic():
    # run your existing encoder; its stdout goes to the safety log
    try:
        proc = subprocess.Popen([sys.executable, ENCODE_SCRIPT],
                                cwd=BASE_DIR,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                preexec_fn=os.setsid)
        def pump(p):
            for line in p.stdout: log(f"[encode] {line.rstrip()}")
            code = p.wait(); log(f"[encode] exited with {code}")
        threading.Thread(target=pump, args=(proc,), daemon=True).start()
        return jsonify(ok=True)
    except Exception as e:
        log(f"encode failed: {e}")
        return jsonify(ok=False, error=str(e)), 500

if __name__ == "__main__":
    # Run from your desktop session if your scripts open GUI windows
    app.run(host="0.0.0.0", port=8000, debug=False)
