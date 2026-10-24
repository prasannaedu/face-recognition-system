import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.models import get_face_app

# --- Settings (all can be given on the command line) ------------------------
# Usage:
#   python 05_video_detection_stats.py [video] [det_thresh] [samples/sec] [det_sizes]
# Example:
#   python 05_video_detection_stats.py videos/railway_5min.mp4 0.5 0.5 640
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "videos/cam1_trim.mp4"
DET_THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # "is it a face?" cut-off
SAMPLES_PER_SECOND = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
DET_SIZE_ARG = sys.argv[4] if len(sys.argv) > 4 else "640,1280"
DET_SIZES = [(int(s), int(s)) for s in DET_SIZE_ARG.split(",")]
MAX_SAVED_IMAGES = 6                    # example frames saved per detector size
OUT_DIR = Path("videos/annotated")      # inside videos/, so Git ignores it
# ---------------------------------------------------------------------------


def draw_faces(frame, faces):
    """Return a copy of the frame with a green box + confidence per face."""
    out = frame.copy()
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{f.det_score:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def analyse(det_size, video_stem):
    """Run detection on sampled frames using one detector size."""
    app = get_face_app()
    # det_thresh = minimum confidence for something to count as a face.
    # det_size   = the size the frame is shrunk to before the detector looks.
    app.prepare(ctx_id=-1, det_thresh=DET_THRESH, det_size=det_size)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / SAMPLES_PER_SECOND))
    total_samples = max(1, total_frames // step)
    save_every = max(1, total_samples // MAX_SAVED_IMAGES)
    print(f"Will test about {total_samples} frames (every {step}th frame)...")

    faces_per_frame, widths, scores = [], [], []
    saved = 0
    frame_idx = 0
    started = time.time()

    while cap.grab():                      # move to the next frame (fast)
        if frame_idx % step == 0:          # only decode the frames we keep
            ok, frame = cap.retrieve()
            if not ok:
                break

            faces = app.get(frame)
            faces_per_frame.append(len(faces))
            for f in faces:
                x1, y1, x2, y2 = f.bbox
                widths.append(float(x2 - x1))
                scores.append(float(f.det_score))

            sample_no = len(faces_per_frame)

            # Progress line every 20 tested frames.
            if sample_no % 20 == 0:
                elapsed = time.time() - started
                print(f"  {sample_no}/{total_samples} frames done "
                      f"({elapsed:.0f} s elapsed)", flush=True)

            # Save a few evenly spaced example frames so you can SEE results.
            if saved < MAX_SAVED_IMAGES and (sample_no - 1) % save_every == 0:
                name = (f"{video_stem}_det{det_size[0]}_t{DET_THRESH}"
                        f"_frame{frame_idx:05d}.jpg")
                cv2.imwrite(str(OUT_DIR / name), draw_faces(frame, faces))
                saved += 1
        frame_idx += 1

    cap.release()
    print(f"Finished in {time.time() - started:.0f} seconds.")
    return faces_per_frame, widths, scores


def summarise(det_size, faces_per_frame, widths, scores):
    print(f"\n=== Detector size {det_size[0]}, det_thresh {DET_THRESH} ===")
    print("Frames tested:", len(faces_per_frame))
    print("Frames with at least one face:",
          sum(1 for n in faces_per_frame if n > 0))
    print("Faces found in total:", len(widths))
    if not widths:
        return
    w = np.array(widths)
    print(f"Faces per frame: average {np.mean(faces_per_frame):.1f}, "
          f"max {max(faces_per_frame)}")
    print(f"Face width in pixels: smallest {w.min():.0f}, "
          f"median {np.median(w):.0f}, largest {w.max():.0f}")
    print("Faces by width:")
    print(f"  under 40 px    : {(w < 40).sum()}")
    print(f"  40 to 80 px    : {((w >= 40) & (w < 80)).sum()}")
    print(f"  80 to 160 px   : {((w >= 80) & (w < 160)).sum()}")
    print(f"  160 px or more : {(w >= 160).sum()}")
    print(f"Detection confidence: average {np.mean(scores):.2f}, "
          f"lowest {min(scores):.2f}")


# ---- Main ------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Could not open video:", VIDEO_PATH)
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print("Video:", VIDEO_PATH)
print(f"Resolution: {width}x{height}, {fps:.1f} fps, {frames} frames, "
      f"about {frames / fps:.1f} seconds")
print("Detection threshold (det_thresh):", DET_THRESH)
print("Samples per second:", SAMPLES_PER_SECOND)

OUT_DIR.mkdir(parents=True, exist_ok=True)
stem = Path(VIDEO_PATH).stem

for det_size in DET_SIZES:
    faces_per_frame, widths, scores = analyse(det_size, stem)
    summarise(det_size, faces_per_frame, widths, scores)

print("\nAnnotated example frames saved in:", OUT_DIR)
