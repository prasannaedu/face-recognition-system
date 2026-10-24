import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.gallery import GALLERY_DIR, enroll
from core.tracker import crop_with_margin

MODEL_NAME = "InsightFace buffalo_l (SCRFD detector + ArcFace w600k_r50, 512-D)"
MIN_SPACING_S = 0.6   # picked faces of one track must be at least this far apart in time

parser = argparse.ArgumentParser(
    description="Enroll people into a gallery from the tracks made by 10_track_faces.py. "
                "You decide who is who: person_id=track_id[,track_id...] "
                "(two tracks of the same person can be listed together).",
    epilog="Example: python 11_enroll_from_tracks.py rail_a=56 rail_b=58 rail_c=59,60",
)
parser.add_argument("people", nargs="+", help="person_id=track_id[,track_id...]")
parser.add_argument("--tracks", default="quality_out/tracks", help="folder written by 10_track_faces.py")
parser.add_argument("--gallery", default="quality_out/gallery_video",
                    help="gallery folder to enroll into (never the main gallery/)")
parser.add_argument("--video", default="videos/railway_5min.mp4",
                    help="the video the tracks came from (used to cut the face pictures)")
parser.add_argument("-k", type=int, default=3, help="best faces to enroll per person (default 3)")
args = parser.parse_args()

gallery_dir = Path(args.gallery)
if gallery_dir.resolve() == GALLERY_DIR.resolve():
    print("Refusing to write into the main gallery/ folder. Use another --gallery folder.")
    sys.exit(1)

data_path = Path(args.tracks) / "tracks.npz"
if not data_path.exists():
    print("Missing", data_path, "- run 10_track_faces.py first.")
    sys.exit(1)
d = np.load(data_path)
kept = set(int(x) for x in d["kept_track_ids"])

# ---- 1. Decide which faces to enroll for each person ------------------------
plan = {}   # person_id -> list of observation indices
for item in args.people:
    if "=" not in item:
        print("Bad argument (expected person_id=track_id[,track_id]):", item)
        sys.exit(1)
    person_id, ids = item.split("=", 1)
    track_ids = [int(t) for t in ids.split(",")]
    for t in track_ids:
        if t not in kept:
            print(f"{person_id}: track {t} is not a kept track (unknown, or seen too few times).")
            sys.exit(1)
    idx = [i for i in np.where(np.isin(d["obs_track"], track_ids) & d["obs_passed"])[0]]
    idx.sort(key=lambda i: -d["obs_rank"][i])        # best first
    picked = []
    for i in idx:
        if len(picked) >= args.k:
            break
        # Skip faces taken almost at the same moment as one already picked
        # (near-duplicates would add nothing).
        if any(d["obs_track"][i] == d["obs_track"][j]
               and abs(d["obs_time"][i] - d["obs_time"][j]) < MIN_SPACING_S for j in picked):
            continue
        picked.append(i)
    if not picked:
        print(f"{person_id}: tracks {track_ids} have no face that passes the quality check; not enrolled.")
        continue
    plan[person_id] = (track_ids, picked)

if not plan:
    print("Nothing to enroll.")
    sys.exit(1)

# Skip people who already exist in the target gallery (same rule as 03).
for person_id in list(plan):
    if (gallery_dir / person_id).exists():
        print(f"{person_id}: already in {gallery_dir}, skipping")
        del plan[person_id]

# ---- 2. Cut the face pictures from the video (one pass, in frame order) ----
needed = {}   # frame -> list of (person_id, observation index)
for person_id, (_, picked) in plan.items():
    for i in picked:
        needed.setdefault(int(d["obs_frame"][i]), []).append((person_id, i))

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print("Could not open video:", args.video)
    sys.exit(1)
crops = {}    # observation index -> face picture
last_frame = max(needed) if needed else -1
frame_idx = 0
while frame_idx <= last_frame and cap.grab():
    if frame_idx in needed:
        ok, frame = cap.retrieve()
        if ok:
            for _, i in needed[frame_idx]:
                crops[i] = crop_with_margin(frame, d["obs_bbox"][i])
    frame_idx += 1
cap.release()

# ---- 3. Enroll --------------------------------------------------------------
info_path = gallery_dir / "gallery_info.json"
info = json.loads(info_path.read_text()) if info_path.exists() else {"persons": {}}
info["model"] = MODEL_NAME
info["note"] = ("Embeddings from different models cannot be compared. If the model is "
                "changed, everyone must be enrolled again.")
info["source_video"] = args.video

for person_id, (track_ids, picked) in plan.items():
    done = []
    for i in picked:
        crop = crops.get(i)
        if crop is None or crop.size == 0:
            print(f"{person_id}: could not cut the face at frame {int(d['obs_frame'][i])}; skipped")
            continue
        index = enroll(person_id, d["obs_embedding"][i], crop, gallery_dir=gallery_dir)
        done.append({"index": index, "track": int(d["obs_track"][i]),
                     "frame": int(d["obs_frame"][i]), "time_s": round(float(d["obs_time"][i]), 1),
                     "rank": round(float(d["obs_rank"][i]), 3)})
    info["persons"][person_id] = {"tracks": track_ids, "faces": done,
                                  "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"{person_id}: enrolled {len(done)} faces from track(s) {track_ids} "
          f"(times {[f['time_s'] for f in done]} s)")

gallery_dir.mkdir(parents=True, exist_ok=True)
info_path.write_text(json.dumps(info, indent=2))
print("\nGallery:", gallery_dir, "| info:", info_path.name)
