import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.models import get_face_app
from core.quality import assess_quality
from core.tracker import FaceTracker, crop_with_margin

# --- Settings (all can be given on the command line) ------------------------
# Usage:
#   python 10_track_faces.py [video_or_rtsp] [samples/sec] [out_dir] [max_missed] [min_hits] [start_s] [end_s]
# Examples:
#   python 10_track_faces.py                                  # railway video, 5 samples/sec
#   python 10_track_faces.py videos/railway_5min.mp4 5 quality_out/tracks 5 3
#   python 10_track_faces.py videos/railway_5min.mp4 5 quality_out/tracks_test 5 3 165 200   # a short window
#   python 10_track_faces.py rtsp://user:pass@ip:554/stream 5 quality_out/tracks_live       # (untested)
# max_missed : a person can be missed this many samples in a row and keep the same track
# min_hits   : tracks seen fewer times than this are treated as noise (listed, not saved as images)
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "videos/railway_5min.mp4"
SAMPLES_PER_SECOND = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("quality_out/tracks")
MAX_MISSED = int(sys.argv[4]) if len(sys.argv) > 4 else 5
MIN_HITS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
START_S = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
END_S = float(sys.argv[7]) if len(sys.argv) > 7 else None

# Same face filters and detector settings as 06_extract_faces.py, so the faces
# (and their embeddings) are identical to the ones you already labeled.
MIN_WIDTH = 40.0
MIN_DET = 0.6
DET_THRESH = 0.5
DET_SIZE = (640, 640)

CELL = 112          # contact-sheet cell size
STRIP_MAX = 10      # faces per track strip
SHEET_PER_PAGE = 60
# ---------------------------------------------------------------------------


def label_cell(img, text, size=CELL):
    img = cv2.resize(img, (size, size))
    cv2.putText(img, text, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (0, 255, 0), 1, cv2.LINE_AA)
    return img


def make_strip(track, height=96):
    """One picture with up to STRIP_MAX evenly spaced faces of a track."""
    picks = np.unique(np.linspace(0, track.hits - 1, min(track.hits, STRIP_MAX)).astype(int))
    cells = []
    for i in picks:
        o = track.obs[i]
        img = o["crop"]
        w = max(1, int(img.shape[1] * height / img.shape[0]))
        img = cv2.resize(img, (w, height))
        cv2.putText(img, f"{o['time']:.1f}s", (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 0), 1, cv2.LINE_AA)
        cells.append(img)
    return np.hstack(cells)


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Could not open video:", VIDEO_PATH)
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 0 or negative for live streams
step = max(1, round(fps / SAMPLES_PER_SECOND))
start_frame = int(START_S * fps)
end_frame = int(END_S * fps) if END_S else (total_frames if total_frames > 0 else None)
if start_frame:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if start_frame else 0
total_samples = max(1, ((end_frame or frame_idx) - frame_idx) // step)

print("Video:", VIDEO_PATH)
print(f"{fps:.1f} fps -> sampling every {step} frames (about {fps / step:.1f} per second)")
print(f"Faces: width >= {MIN_WIDTH:.0f} px, detection >= {MIN_DET}; "
      f"tracks survive {MAX_MISSED} missed samples; noise if seen < {MIN_HITS} times")
if end_frame:
    print(f"Will test about {total_samples} frames...")

app = get_face_app()
app.prepare(ctx_id=-1, det_thresh=DET_THRESH, det_size=DET_SIZE)
tracker = FaceTracker(max_missed=MAX_MISSED)

# Clean only OUR OWN old outputs in the folder (never anything else).
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "strips").mkdir(exist_ok=True)
for pattern in ("track_*_best.jpg", "sheet_*.jpg"):
    for old in OUT_DIR.glob(pattern):
        old.unlink()
for old in (OUT_DIR / "strips").glob("track_*.jpg"):
    old.unlink()

tested = 0
started = time.time()
while cap.grab():
    if frame_idx % step == 0:
        ok, frame = cap.retrieve()
        if not ok:
            break
        tested += 1
        time_s = frame_idx / fps

        detections = []
        for f in app.get(frame):
            x1, y1, x2, y2 = f.bbox
            if float(x2 - x1) < MIN_WIDTH or float(f.det_score) < MIN_DET:
                continue
            crop = crop_with_margin(frame, f.bbox)
            if crop.size == 0:
                continue
            detections.append({
                "bbox": [float(v) for v in f.bbox],
                "embedding": f.normed_embedding,
                "quality": assess_quality(frame, f),
                "det_score": float(f.det_score),
                "crop": crop,
            })
        tracker.update(detections, frame_idx, time_s)

        if tested % 25 == 0:
            print(f"  {tested}/{total_samples} frames done ({time.time() - started:.0f} s elapsed), "
                  f"{len(tracker.tracks)} tracks so far", flush=True)
    frame_idx += 1
    if end_frame and frame_idx >= end_frame:
        break

cap.release()
tracks = tracker.finish()
kept = [t for t in tracks if t.hits >= MIN_HITS]
print(f"\nFinished in {time.time() - started:.0f} seconds. Frames tested: {tested}")
print(f"Tracks: {len(tracks)} total, {len(kept)} kept (seen >= {MIN_HITS} times), "
      f"{len(tracks) - len(kept)} noise")

# ---- Save: best crop, strip, and a row in the summary for every kept track --
rows = []
for t in kept:
    best = t.best[0]
    best_name = f"track_{t.id:04d}_best.jpg"
    cv2.imwrite(str(OUT_DIR / best_name), best["crop"])
    cv2.imwrite(str(OUT_DIR / "strips" / f"track_{t.id:04d}.jpg"), make_strip(t))
    passed = sum(o["quality"]["passed"] for o in t.obs)
    q = best["quality"]
    rows.append({
        "track_id": t.id,
        "first_s": round(t.obs[0]["time"], 1),
        "last_s": round(t.obs[-1]["time"], 1),
        "observations": t.hits,
        "passed_quality": passed,
        "best_time_s": round(best["time"], 1),
        "best_width_px": round(best["width"]),
        "best_yaw": round(q["yaw"], 1),
        "best_pitch": round(q["pitch"], 1),
        "best_sharpness": round(q["sharpness"]),
        "best_passed": q["passed"],
        "best_file": best_name,
    })

with open(OUT_DIR / "tracks.csv", "w", newline="") as fh:
    if rows:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# Everything per observation, so later steps (enrollment, calibration) can reuse it.
all_obs = [(t, o) for t in tracks for o in t.obs]
if all_obs:
    np.savez(
        OUT_DIR / "tracks.npz",
        obs_track=np.array([t.id for t, _ in all_obs]),
        obs_frame=np.array([o["frame"] for _, o in all_obs]),
        obs_time=np.array([o["time"] for _, o in all_obs]),
        obs_bbox=np.array([o["bbox"] for _, o in all_obs]),
        obs_det=np.array([o["det_score"] for _, o in all_obs]),
        obs_width=np.array([o["width"] for _, o in all_obs]),
        obs_yaw=np.array([o["quality"]["yaw"] for _, o in all_obs], dtype=float),
        obs_pitch=np.array([o["quality"]["pitch"] for _, o in all_obs], dtype=float),
        obs_sharp=np.array([o["quality"]["sharpness"] for _, o in all_obs]),
        obs_passed=np.array([o["quality"]["passed"] for _, o in all_obs]),
        obs_rank=np.array([o["rank"] for _, o in all_obs]),
        obs_embedding=np.stack([o["embedding"] for _, o in all_obs]),
        kept_track_ids=np.array([t.id for t in kept]),
    )

# Contact sheets of every kept track's best face (pages of SHEET_PER_PAGE).
for page, start in enumerate(range(0, len(kept), SHEET_PER_PAGE), start=1):
    chunk = kept[start:start + SHEET_PER_PAGE]
    cells = [label_cell(t.best[0]["crop"].copy(),
                        f"T{t.id} n{t.hits} {t.obs[0]['time']:.0f}-{t.obs[-1]['time']:.0f}s")
             for t in chunk]
    while len(cells) % 10:
        cells.append(np.zeros((CELL, CELL, 3), np.uint8))
    sheet = np.vstack([np.hstack(cells[r:r + 10]) for r in range(0, len(cells), 10)])
    cv2.imwrite(str(OUT_DIR / f"sheet_{page}.jpg"), sheet)

print("\nTrack  first-last (s)  seen  passed  best: width yaw pitch sharp")
for r in rows[:40]:
    print(f"  T{r['track_id']:<4d} {r['first_s']:6.1f}-{r['last_s']:<6.1f}  {r['observations']:4d}  "
          f"{r['passed_quality']:5d}   {r['best_width_px']:4d}px {r['best_yaw']:+5.0f} "
          f"{r['best_pitch']:+5.0f} {r['best_sharpness']:5d}")
if len(rows) > 40:
    print(f"  ... {len(rows) - 40} more in tracks.csv")
print("\nSaved in", OUT_DIR, ": tracks.csv, tracks.npz, track_*_best.jpg, strips/, sheet_*.jpg")
