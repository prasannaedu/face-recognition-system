import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

from core import quality

# Usage:
#   python 09_quality_report.py [data_dir]
#   python 09_quality_report.py [data_dir] a1.jpg a2.jpg vs b1.jpg b2.jpg ...
# data_dir is the folder written by 06_extract_faces.py (default quality_out).
# Crop names may be given as the full name or just the start (t0176.0_f05280_2).
# With groups (one person per group, separated by "vs") it also prints the
# before/after similarity gap and the quality numbers of every named crop.
args = sys.argv[1:]
DATA_DIR = Path(args[0]) if args and not args[0].endswith(".jpg") and args[0] != "vs" else Path("quality_out")
group_args = args[1:] if args and DATA_DIR == Path(args[0]) else args
DATA = DATA_DIR / "faces_data.npz"

CELL = 112
if not DATA.exists():
    print("Missing", DATA, "- run 06_extract_faces.py with this output folder first.")
    sys.exit(1)

d = np.load(DATA)
if "q_yaw" not in d.files:
    print("This data has no quality columns. Re-run 06_extract_faces.py.")
    sys.exit(1)

files = [str(f) for f in d["files"]]
n = len(files)
yaw, pitch = np.abs(d["q_yaw"]), np.abs(d["q_pitch"])
sharp, det = d["q_sharp112"], d["det_scores"]
norm, mouth_eye = d["q_emb_norm"], d["q_mouth_eye"]
emb = d["embeddings"]


def pct(x):
    return f"{100 * x:.0f}%"


# ---- 1. What the numbers look like ------------------------------------------
print(f"Faces in {DATA}: {n}\n")
print("Distribution (all faces):")
for label, v in [("|yaw| deg", yaw), ("|pitch| deg", pitch), ("sharpness@112", sharp),
                 ("det score", det), ("embedding norm", norm), ("mouth/eye sharp", mouth_eye)]:
    print(f"  {label:16s} min {v.min():8.2f}  p10 {np.percentile(v, 10):8.2f}  "
          f"median {np.median(v):8.2f}  p90 {np.percentile(v, 90):8.2f}  max {v.max():8.2f}")

# ---- 2. How many faces survive each cut-off ----------------------------------
print("\nFaces kept at each cut-off (one limit at a time):")
print("  |yaw| <= " + "   ".join(f"{t}: {pct((yaw <= t).mean())}" for t in (15, 20, 25, 30, 40, 50)))
print("  |pitch| <= " + "   ".join(f"{t}: {pct((pitch <= t).mean())}" for t in (15, 20, 25, 30, 40)))
print("  sharpness >= " + "   ".join(
    f"p{p}={np.percentile(sharp, p):.0f}: {pct((sharp >= np.percentile(sharp, p)).mean())}"
    for p in (5, 10, 20, 30)))

# ---- 3. Verdict with the limits currently set in core/quality.py ------------
yaw_bad = yaw > quality.MAX_YAW_DEG
pitch_bad = pitch > quality.MAX_PITCH_DEG
sharp_bad = sharp < quality.MIN_SHARPNESS
det_bad = det < quality.MIN_DET_SCORE
ok = ~(yaw_bad | pitch_bad | sharp_bad | det_bad)
print(f"\nWith the current limits (yaw {quality.MAX_YAW_DEG:.0f}, pitch {quality.MAX_PITCH_DEG:.0f}, "
      f"sharpness {quality.MIN_SHARPNESS:.0f}, det {quality.MIN_DET_SCORE}):")
print(f"  kept {ok.sum()} / {n} ({pct(ok.mean())})")
print(f"  rejected for yaw {yaw_bad.sum()}, pitch {pitch_bad.sum()}, "
      f"blur {sharp_bad.sum()}, detection {det_bad.sum()} (a face can fail several)")


# ---- 4. Contact sheets so you can judge by eye ------------------------------
def load_cell(name, text):
    img = cv2.imread(str(DATA_DIR / name))
    if img is None:
        img = np.zeros((CELL, CELL, 3), np.uint8)
    img = cv2.resize(img, (CELL, CELL))
    cv2.putText(img, text, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def sheet(indices, path, label_fn, per_row=10, limit=60):
    indices = list(indices)[:limit]
    if not indices:
        return
    cells = [load_cell(files[i], label_fn(i)) for i in indices]
    while len(cells) % per_row:
        cells.append(np.zeros((CELL, CELL, 3), np.uint8))
    rows = [np.hstack(cells[r:r + per_row]) for r in range(0, len(cells), per_row)]
    cv2.imwrite(str(path), np.vstack(rows))
    print("  saved", path)


print("\nContact sheets (green text = yaw/pitch/sharpness):")
label = lambda i: f"y{d['q_yaw'][i]:+.0f} p{d['q_pitch'][i]:+.0f} s{sharp[i]:.0f}"
rng = np.random.default_rng(0)
kept_idx = rng.permutation(np.where(ok)[0])
rej_idx = rng.permutation(np.where(~ok)[0])
sheet(kept_idx, DATA_DIR / "sheet_kept.jpg", label)
sheet(rej_idx, DATA_DIR / "sheet_rejected.jpg", label)
sheet(np.argsort(sharp)[:30], DATA_DIR / "sheet_blurriest.jpg", label)
sheet(np.argsort(norm)[:30], DATA_DIR / "sheet_lowest_emb_norm.jpg", label)
sheet(np.argsort(mouth_eye)[:30], DATA_DIR / "sheet_lowest_mouth_eye.jpg", label)

# ---- 5. Named crops: quality numbers, and similarity before/after ----------
groups, current = [], []
for a in group_args:
    if a == "vs":
        groups.append(current)
        current = []
    else:
        current.append(a)
groups.append(current)

if any(groups):
    items = []  # (label, index)
    for g, group in enumerate(groups):
        for key in group:
            hits = [i for i, f in enumerate(files) if f.startswith(Path(key).name.split("_w")[0])]
            if not hits:
                print("Crop not found:", key)
                sys.exit(1)
            items.append((chr(ord("A") + g), hits[0]))

    print("\nNamed crops (per-crop quality):")
    print("  who  crop                     yaw  pitch  sharp@112  det   emb_norm  mouth/eye  verdict")
    for lab, i in items:
        verdict = "KEEP" if ok[i] else "REJECT"
        print(f"  {lab}    {files[i][:22]:22s} {d['q_yaw'][i]:+5.0f} {d['q_pitch'][i]:+6.0f} "
              f"{sharp[i]:9.0f}  {det[i]:.2f}  {norm[i]:7.1f}  {mouth_eye[i]:8.2f}  {verdict}")

    def gap(subset, title):
        same, diff = [], []
        for (la, ia), (lb, ib) in itertools.combinations(subset, 2):
            (same if la == lb else diff).append(float(np.dot(emb[ia], emb[ib])))
        print(f"  {title}: {len(subset)} crops", end="")
        if same:
            print(f" | same: lowest {min(same):.3f}, mean {np.mean(same):.3f}", end="")
        if diff:
            print(f" | different: highest {max(diff):.3f}, mean {np.mean(diff):.3f}", end="")
        if same and diff:
            print(f" | GAP {min(same) - max(diff):+.3f}", end="")
        print()

    print("\nSimilarity before / after the quality filter:")
    gap(items, "all named crops   ")
    gap([(lab, i) for lab, i in items if ok[i]], "only crops kept   ")
    print("  (Few crops per person: treat this as a sanity check, not a measurement.)")
