import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.models import get_face_app
from core.quality import assess_quality

# --- Settings (all can be given on the command line) ------------------------
# Usage:
#   python 06_extract_faces.py [video] [min_width] [min_det] [samples/sec] [out_dir] [filter]
# Example:
#   python 06_extract_faces.py videos/railway_5min.mp4 40 0.6 1
# To keep existing crops safe, write somewhere new; add "filter" to DROP faces
# that fail the quality check (otherwise they are kept, with their scores saved):
#   python 06_extract_faces.py videos/railway_5min.mp4 40 0.6 1 quality_out
#   python 06_extract_faces.py videos/railway_5min.mp4 40 0.6 1 quality_out filter
# NOTE: the output folder's old *.jpg files are deleted at the start of a run.
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "videos/railway_5min.mp4"
MIN_WIDTH = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0   # px: skip smaller faces
MIN_DET = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6      # skip low-confidence faces
SAMPLES_PER_SECOND = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
DET_SIZE = (640, 640)
OUT_DIR = Path(sys.argv[5] if len(sys.argv) > 5 else "videos/faces")  # default is inside videos/, so Git ignores it
QUALITY_FILTER = len(sys.argv) > 6 and sys.argv[6] == "filter"        # drop faces that fail core/quality.py
# ---------------------------------------------------------------------------


def make_crop(frame, bbox, margin=0.2):
    """Face crop with a small margin, clipped to the frame borders."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
    return frame[cy1:cy2, cx1:cx2].copy()


def sharpness(crop):
    """Rough sharpness score: higher = more detail, lower = blurrier."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Could not open video:", VIDEO_PATH)
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, round(fps / SAMPLES_PER_SECOND))
total_samples = max(1, total_frames // step)

print("Video:", VIDEO_PATH)
print(f"Keeping faces with width >= {MIN_WIDTH:.0f} px and confidence >= {MIN_DET}")
print(f"Will test about {total_samples} frames (every {step}th frame)...")

app = get_face_app()
app.prepare(ctx_id=-1, det_thresh=0.5, det_size=DET_SIZE)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for old in OUT_DIR.glob("*.jpg"):      # start clean so runs never mix
    old.unlink()

embeddings, frame_ids, times, widths, det_scores, sharps, files = [], [], [], [], [], [], []
kps_all, quality_rows, rejected = [], [], 0
tested = 0
frame_idx = 0
started = time.time()

while cap.grab():
    if frame_idx % step == 0:
        ok, frame = cap.retrieve()
        if not ok:
            break
        tested += 1
        time_s = frame_idx / fps

        for k, f in enumerate(app.get(frame)):
            x1, y1, x2, y2 = f.bbox
            width = float(x2 - x1)
            if width < MIN_WIDTH or float(f.det_score) < MIN_DET:
                continue
            crop = make_crop(frame, f.bbox)
            if crop.size == 0:
                continue

            # Quality check on the FULL frame (landmarks live in frame coordinates).
            q = assess_quality(frame, f)
            if QUALITY_FILTER and not q["passed"]:
                rejected += 1
                continue

            name = f"t{time_s:06.1f}_f{frame_idx:05d}_{k}_w{int(width)}.jpg"
            cv2.imwrite(str(OUT_DIR / name), crop)

            embeddings.append(f.normed_embedding)
            frame_ids.append(frame_idx)
            times.append(time_s)
            widths.append(width)
            det_scores.append(float(f.det_score))
            sharps.append(sharpness(crop))
            files.append(name)
            kps_all.append(f.kps)
            quality_rows.append(q)

        if tested % 20 == 0:
            print(f"  {tested}/{total_samples} frames done "
                  f"({time.time() - started:.0f} s elapsed), "
                  f"{len(files)} faces kept so far", flush=True)
    frame_idx += 1

cap.release()
print(f"\nFinished in {time.time() - started:.0f} seconds.")
print("Frames tested:", tested)
print(f"Faces kept: {len(files)}")
if QUALITY_FILTER:
    print(f"Rejected by the quality check: {rejected}")

if files:
    w = np.array(widths)
    s = np.array(sharps)
    print(f"Kept face width: smallest {w.min():.0f}, "
          f"median {np.median(w):.0f}, largest {w.max():.0f}")
    print(f"Sharpness (rough): lowest {s.min():.0f}, "
          f"median {np.median(s):.0f}, highest {s.max():.0f}")
    np.savez(
        OUT_DIR / "faces_data.npz",
        embeddings=np.stack(embeddings),
        frames=np.array(frame_ids),
        times=np.array(times),
        widths=w,
        det_scores=np.array(det_scores),
        sharpness=s,
        files=np.array(files),
        # Added for the quality check (the keys above are unchanged):
        kps=np.stack(kps_all),
        q_yaw=np.array([q["yaw"] for q in quality_rows], dtype=float),
        q_pitch=np.array([q["pitch"] for q in quality_rows], dtype=float),
        q_roll=np.array([q["roll"] for q in quality_rows], dtype=float),
        q_sharp112=np.array([q["sharpness"] for q in quality_rows]),
        q_emb_norm=np.array([q["emb_norm"] for q in quality_rows]),
        q_mouth_eye=np.array([q["mouth_eye_ratio"] for q in quality_rows]),
    )
    print("Embeddings and details saved in:", OUT_DIR / "faces_data.npz")

print("Crops saved in:", OUT_DIR)
