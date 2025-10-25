''' 
python people_count.py \
  --dfp models/YOLO_v7_tiny_416_416_3_onnx.dfp \
  --postmodel models/YOLO_v7_tiny_416_416_3_onnx_post.onnx \
  --video_paths /dev/video0

python people_count.py \
  --dfp models/YOLO_v7_tiny_416_416_3_onnx.dfp \
  --postmodel models/YOLO_v7_tiny_416_416_3_onnx_post.onnx \
  --video_paths security.mp4
'''
import argparse
import time
from threading import Thread
from queue import Queue, Full
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np

# Memryx + YOLOv7-tiny helper
try:
    from memryx import MultiStreamAsyncAccl
except Exception as e:
    raise RuntimeError("Memryx 'memryx' package not available. Install memryx first.") from e

try:
    # This comes from the Memryx example (yolov7.py). Make sure it's importable (same folder/PYTHONPATH).
    from yolov7 import YoloV7Tiny as YoloModel
except Exception as e:
    raise RuntimeError(
        "Cannot import 'YoloV7Tiny' from yolov7. "
        "Place yolov7.py (from the Memryx example) next to this script or add it to PYTHONPATH."
    ) from e


# --------------------------- BYTETrack-lite (IoU-based) ---------------------------
@dataclass
class BTTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    hits: int = 0
    missed: int = 0

class BYTETrackerLite:
    """Tiny IoU-based tracker (ByteTrack-style)."""
    def __init__(self, match_thresh: float = 0.45, min_hits: int = 3, max_time_lost: int = 20):
        self.match_thresh = match_thresh
        self.min_hits = min_hits
        self.max_time_lost = max_time_lost
        self.next_id = 1
        self.tracks: Dict[int, BTTrack] = {}

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter == 0: return 0.0
        area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
        area_b = max(0, bx2-bx1) * max(0, by2-by1)
        return inter / max(1e-9, area_a + area_b - inter)

    def update(self, detections: List[Tuple[int,int,int,int]], scores: List[float] | None = None) -> Dict[int, Tuple[int,int,int,int]]:
        if scores is None:
            scores = [1.0] * len(detections)

        # age
        for t in self.tracks.values():
            t.missed += 1

        # greedy IoU assign
        unmatched = list(range(len(detections)))
        for tid, t in list(self.tracks.items()):
            best_iou, best_idx = 0.0, -1
            for di in unmatched:
                iou = self.iou(t.bbox, detections[di])
                if iou > best_iou:
                    best_iou, best_idx = iou, di
            if best_idx >= 0 and best_iou >= self.match_thresh:
                t.bbox = detections[best_idx]
                t.score = scores[best_idx]
                t.hits += 1
                t.missed = 0
                unmatched.remove(best_idx)

        # new tracks
        for di in unmatched:
            det = detections[di]; sc = scores[di]
            self.tracks[self.next_id] = BTTrack(self.next_id, det, sc, hits=1, missed=0)
            self.next_id += 1

        # prune
        for tid in [tid for tid, t in self.tracks.items() if t.missed > self.max_time_lost]:
            del self.tracks[tid]

        # only confirmed
        return {tid: t.bbox for tid, t in self.tracks.items() if t.hits >= self.min_hits}


# --------------------------- App (Memryx + YOLOv7-tiny helper) -------------------
class PeopleCountMemryxV7:
    def __init__(self, args):
        self.args = args
        self.num_streams = len(args.video_paths)
        self.show = args.show
        self.done = False

        # queues & state
        self.streams: List[cv2.VideoCapture] = []
        self.cap_queue = {i: Queue(maxsize=4) for i in range(self.num_streams)}
        self.det_queue = {i: Queue(maxsize=4) for i in range(self.num_streams)}
        self.fps = {i: 0.0 for i in range(self.num_streams)}
        self.last_time = {i: time.time() for i in range(self.num_streams)}
        self.trackers = {i: BYTETrackerLite(match_thresh=0.45, min_hits=3, max_time_lost=20) for i in range(self.num_streams)}

        # open captures + create YOLO helpers (use original stream dims)
        self.models = {}
        for vp in args.video_paths:
            try:
                idx = int(vp)
                cap = cv2.VideoCapture(idx)
            except ValueError:
                cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {vp}")
            self.streams.append(cap)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # IMPORTANT: use the Memryx helper's preprocess/postprocess
            self.models[len(self.streams)-1] = YoloModel(stream_img_size=(h, w, 3))

        self.display_thread = Thread(target=self._display_loop, daemon=True)

    def run(self):
        accl = MultiStreamAsyncAccl(dfp=self.args.dfp)
        accl.set_postprocessing_model(self.args.postmodel, model_idx=0)

        self.display_thread.start()
        accl.connect_streams(self._capture_preprocess, self._postprocess, self.num_streams)
        accl.wait()
        self.done = True
        self.display_thread.join()

    # feed frames -> model.preprocess() -> MXA
    def _capture_preprocess(self, stream_idx: int):
        while True:
            ok, frame = self.streams[stream_idx].read()
            if not ok or self.done:
                return None
            # keep original for draw
            try:
                self.cap_queue[stream_idx].put(frame, timeout=0.01)
            except Full:
                # drop if display is behind
                pass
            # use Memryx example's preprocess (handles 416x416, layout, norm, letterbox if any)
            pre = self.models[stream_idx].preprocess(frame)
            return pre

    # MXA outputs -> model.postprocess() -> person-only -> tracker
    def _postprocess(self, stream_idx: int, *mxa_output):
        # Memryx example returns a list of dicts: {'bbox':[l,t,r,b], 'score':..., 'class':'...', 'class_idx':int}
        dets = self.models[stream_idx].postprocess(mxa_output)

        # Filter to persons (COCO class 0) and apply a confidence floor
        boxes, scores = [], []
        conf_floor = self.args.min_conf
        for d in dets:
            if (d.get('class_idx', -1) == 0 or d.get('class') == 'person') and float(d.get('score', 0.0)) >= conf_floor:
                l, t, r, b = d['bbox']
                boxes.append((int(l), int(t), int(r), int(b)))
                scores.append(float(d['score']))

        # push to display
        try:
            self.det_queue[stream_idx].put((boxes, scores), timeout=0.01)
        except Full:
            pass

        # fps
        now = time.time()
        dt = now - self.last_time[stream_idx]
        if dt > 0:
            self.fps[stream_idx] = 1.0 / dt
        self.last_time[stream_idx] = now

    def _display_loop(self):
        while not self.done:
            for i in range(self.num_streams):
                if self.cap_queue[i].empty() or self.det_queue[i].empty():
                    continue
                frame = self.cap_queue[i].get()
                boxes, scores = self.det_queue[i].get()
                self.cap_queue[i].task_done(); self.det_queue[i].task_done()

                tracks = self.trackers[i].update(boxes, scores)
                people_count = len(tracks)

                # draw tracks
                for tid, (x1,y1,x2,y2) in tracks.items():
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"ID {tid}", (x1, max(0, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(frame, f"People: {people_count}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                cv2.putText(frame, f"FPS: {self.fps[i]:.1f}", (20,75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                if self.show:
                    cv2.imshow(f"Stream {i} - YOLOv7-tiny (Memryx)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.done = True
                break

        for cap in self.streams:
            cap.release()
        cv2.destroyAllWindows()


# --------------------------- CLI ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="People counting (Memryx + YOLOv7-tiny helper postprocess)")
    p.add_argument('--video_paths', nargs='+', default=['/dev/video0'],
                   help='Webcam index like 0, or /dev/videoX, or file path(s)')
    p.add_argument('-d','--dfp', required=True,
                   help='Path to DFP, e.g. models/YOLO_v7_tiny_416_416_3_onnx.dfp')
    p.add_argument('-m','--postmodel', required=True,
                   help='Path to post ONNX, e.g. models/YOLO_v7_tiny_416_416_3_onnx_post.onnx')
    p.add_argument('--min_conf', type=float, default=0.35,
                   help='Confidence floor applied after Memryx postprocess (default: 0.35)')
    p.add_argument('--no_display', dest='show', action='store_false', default=True)
    return p.parse_args()

def main():
    args = parse_args()
    app = PeopleCountMemryxV7(args)
    app.run()

if __name__ == '__main__':
    main()
