import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from core.gallery import load_gallery, search
from core.models import get_face_app
from core.quality import assess_quality
from core.tracker import FaceTracker, crop_with_margin

# --- Settings (all can be given on the command line) ------------------------
# Usage:
#   python 12_live_recognize.py [source] [gallery_dir] [samples/sec] [out_dir] [max_seconds] [cpu|gpu]
# Example (needs the RTSP simulation from PROJECT_LOG.md running):
#   python 12_live_recognize.py rtsp://localhost:8554/railway quality_out/gallery_video 5 quality_out/live 60
# source      : an rtsp:// link, or a video file (played at its normal speed)
# max_seconds : stop after this many seconds (0 = run until Ctrl+C)
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "rtsp://localhost:8554/railway"
GALLERY_DIR = sys.argv[2] if len(sys.argv) > 2 else "quality_out/gallery_video"
SAMPLES_PER_SECOND = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
OUT_ROOT = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("quality_out/live")
MAX_SECONDS = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
WANT_GPU = len(sys.argv) > 6 and sys.argv[6].lower() == "gpu"

THRESHOLD = 0.4          # PROVISIONAL, the same value as 04_recognize.py (see 13_calibrate_threshold.py)
MIN_HITS_TO_DECIDE = 3   # a track must be seen this many times before we name it
MAX_MISSED = 8           # samples a person may be missed and still keep the same track
MAX_OBS = 60             # per track, keeps memory flat on long runs
MIN_WIDTH = 40.0         # same face filters and detector settings as 06 / 10
MIN_DET = 0.6
DET_THRESH = 0.5
DET_SIZE = (640, 640)
STATUS_EVERY_S = 10
# ---------------------------------------------------------------------------

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")   # TCP: no lost packets
IS_FILE = "://" not in SOURCE


class LatestFrameReader(threading.Thread):
    """Reads the source all the time and keeps ONLY the newest frame.

    WHY: recognition is slower than the camera. If we read frames one by one at our
    own pace, old frames pile up and we end up looking at the past. Here a separate
    thread keeps reading, and the main loop always takes the newest frame, so we
    stay close to real time. It also reconnects if the stream drops.
    """

    def __init__(self, source: str):
        super().__init__(daemon=True)
        self.source = source
        self.frame = None
        self.index = 0              # number of frames received so far
        self.reconnects = 0
        self.ended = False          # a video file reached its end
        self.stop_flag = threading.Event()
        self._lock = threading.Lock()

    def run(self):
        while not self.stop_flag.is_set():
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.reconnects += 1
                time.sleep(2.0)
                continue
            pace = (1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0)) if IS_FILE else 0.0
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self.frame = frame
                    self.index += 1
                if pace:
                    time.sleep(pace)    # play a file at normal speed, like a camera
            cap.release()
            if IS_FILE:
                self.ended = True
                return
            self.reconnects += 1
            time.sleep(1.0)

    def latest(self, after_index: int):
        """The newest frame if it is newer than after_index, else None."""
        with self._lock:
            if self.frame is None or self.index == after_index:
                return None
            return self.frame, self.index


def memory_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


# ---- Setup -------------------------------------------------------------------
use_gpu = WANT_GPU
if WANT_GPU and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
    print("GPU requested, but this onnxruntime has no CUDA support (CPU build): using the CPU.")
    use_gpu = False

gallery = load_gallery(GALLERY_DIR)
if not gallery:
    print("No people found in", GALLERY_DIR, "- enroll some first (11_enroll_from_tracks.py).")
    sys.exit(1)

app = get_face_app(use_gpu)
app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=DET_THRESH, det_size=DET_SIZE)
providers = app.models["detection"].session.get_providers()

run_dir = OUT_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
(run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
events_file = open(run_dir / "events.jsonl", "a", buffering=1)

print("Source          :", SOURCE)
print("Gallery         :", GALLERY_DIR, "->", sorted(gallery.keys()))
print(f"Target          : {SAMPLES_PER_SECOND:g} samples/sec  (budget {1000 / SAMPLES_PER_SECOND:.0f} ms per frame)")
print("Running on      :", providers[0], "| threshold", THRESHOLD, "(provisional)")
print("Output          :", run_dir)
print("Stop with Ctrl+C.\n")

tracker = FaceTracker(max_missed=MAX_MISSED, keep_all_crops=False, max_obs=MAX_OBS)
reader = LatestFrameReader(SOURCE)
reader.start()

states = {}          # track id -> {"rank": best rank already used, "identity": name or None, "score": ...}
n_processed = 0
slow_frames = 0
proc_ms = []
last_status = time.time()
started = time.time()


def log_event(kind, track, state, best, matches, extra=None):
    runner = matches[1] if len(matches) > 1 else (None, None)
    snap = None
    if best is not None and best.get("crop") is not None and state.get("identity"):
        snap = f"track{track.id:04d}_{state['identity']}.jpg"
        cv2.imwrite(str(run_dir / "snapshots" / snap), best["crop"])
    ev = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_s": round(time.time() - started, 1),
        "event": kind,
        "track": track.id,
        "identity": state.get("identity"),
        "score": None if not matches else round(matches[0][1], 3),
        "runner_up": runner[0],
        "runner_up_score": None if runner[1] is None else round(runner[1], 3),
        "faces_seen": track.n_obs,
        "snapshot": snap,
    }
    if best is not None:
        q = best["quality"]
        ev.update(det=round(best["det_score"], 2), yaw=round(q["yaw"], 1),
                  pitch=round(q["pitch"], 1), sharpness=round(q["sharpness"]))
    if extra:
        ev.update(extra)
    events_file.write(json.dumps(ev) + "\n")
    who = ev["identity"] or "UNKNOWN"
    print(f"[{ev['run_s']:7.1f}s] {kind:11s} track {track.id:<4d} -> {who:12s} "
          f"score {ev['score']}  (runner-up {ev['runner_up']} {ev['runner_up_score']})")


def close_finished(tracks):
    for t in tracks:
        st = states.pop(t.id, None)
        if t.n_obs < MIN_HITS_TO_DECIDE:
            continue            # too short: noise, not worth an event
        best = next((o for o in t.best if o["quality"]["passed"]), None)
        if st is None:
            # Never had a good enough face to decide on.
            log_event("track_closed", t, {"identity": None}, None, [],
                      {"note": "no face passed the quality filter"})
        else:
            matches = search(best["embedding"], gallery, top_k=3) if best else []
            log_event("track_closed", t, st, best, matches)


# ---- Main loop -----------------------------------------------------------------
period = 1.0 / SAMPLES_PER_SECOND
next_time = time.time()
last_index = 0
try:
    while True:
        now = time.time()
        if MAX_SECONDS and now - started > MAX_SECONDS:
            break
        if reader.ended and reader.latest(last_index) is None:
            break
        if now < next_time:
            time.sleep(min(0.01, next_time - now))
            continue
        got = reader.latest(last_index)
        if got is None:
            time.sleep(0.01)
            continue
        frame, last_index = got
        t0 = time.time()
        next_time = max(next_time + period, t0)      # if we are late, do not try to catch up

        detections = []
        for f in app.get(frame):
            x1, y1, x2, y2 = f.bbox
            if float(x2 - x1) < MIN_WIDTH or float(f.det_score) < MIN_DET:
                continue
            detections.append({
                "bbox": [float(v) for v in f.bbox],
                "embedding": f.normed_embedding,
                "quality": assess_quality(frame, f),
                "det_score": float(f.det_score),
                "crop": crop_with_margin(frame, f.bbox),
            })
        tracker.update(detections, last_index, t0 - started)

        # Name the people we can see right now, using each track's best passing face.
        for t in tracker.tracks:
            if t.closed or t.last["frame"] != last_index or t.n_obs < MIN_HITS_TO_DECIDE:
                continue
            best = next((o for o in t.best if o["quality"]["passed"]), None)
            if best is None:
                continue
            st = states.setdefault(t.id, {"rank": -1.0, "identity": None})
            if best["rank"] <= st["rank"] + 1e-9:
                continue                              # no better face than the one already used
            st["rank"] = best["rank"]
            matches = search(best["embedding"], gallery, top_k=3)
            identity = matches[0][0] if matches[0][1] >= THRESHOLD else None
            if identity and identity != st["identity"]:
                st["identity"] = identity
                log_event("identified", t, st, best, matches)

        close_finished(tracker.pop_closed())

        n_processed += 1
        ms = (time.time() - t0) * 1000
        proc_ms.append(ms)
        slow_frames += ms > period * 1000

        if time.time() - last_status >= STATUS_EVERY_S:
            elapsed = time.time() - started
            recent = proc_ms[-50:]
            print(f"[status {elapsed:5.0f}s] input {reader.index / elapsed:4.1f} fps | "
                  f"processed {n_processed / elapsed:4.2f}/s (target {SAMPLES_PER_SECOND:g}) | "
                  f"{np.mean(recent):4.0f} ms/frame (budget {period * 1000:.0f}) | "
                  f"slow frames {slow_frames}/{n_processed} | active tracks "
                  f"{sum(not t.closed for t in tracker.tracks)} | reconnects {reader.reconnects} | "
                  f"memory {memory_mb():.0f} MB", flush=True)
            last_status = time.time()
except KeyboardInterrupt:
    print("\nStopping (Ctrl+C)...")

reader.stop_flag.set()
reader.join(timeout=10)      # let the reader leave the video library cleanly before Python exits
tracker.finish()
close_finished(tracker.pop_closed())
events_file.close()
elapsed = time.time() - started
print(f"\nStopped after {elapsed:.0f} s: processed {n_processed} frames "
      f"({n_processed / max(elapsed, 1e-9):.2f}/s), input frames {reader.index}, "
      f"reconnects {reader.reconnects}, memory {memory_mb():.0f} MB.")
print("Events saved in", run_dir / "events.jsonl")
