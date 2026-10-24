import sys
from pathlib import Path

import cv2
import numpy as np

# Usage:
#   python 13_calibrate_threshold.py [tracks_dir] [out_dir]
# tracks_dir : folder written by 10_track_faces.py (default quality_out/tracks)
# out_dir    : where the report and picture are saved (default quality_out/calibration)
#
# This script only MEASURES and SUGGESTS. It never changes the 0.4 threshold used
# by 04_recognize.py and the API.
TRACKS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("quality_out/tracks")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("quality_out/calibration")

CURRENT_THRESHOLD = 0.4          # the provisional value used elsewhere (for comparison only)
MIN_GAP_S = 1.0                  # same-person pairs must be at least this far apart in time
MIN_OVERLAP_S = 1.0              # tracks must be alive together this long to count as "different people"
FAR_TARGETS = [1e-2, 1e-3, 1e-4]
GALLERY_SIZES = [3, 50, 500, 5000]

data_path = TRACKS_DIR / "tracks.npz"
if not data_path.exists():
    print("Missing", data_path, "- run 10_track_faces.py first.")
    sys.exit(1)
d = np.load(data_path)
kept = set(int(x) for x in d["kept_track_ids"])

# Only faces that PASS the quality filter (the ones that would really be recognized)
# and belong to kept tracks.
ok = d["obs_passed"] & np.isin(d["obs_track"], list(kept))
trk, tm, emb = d["obs_track"][ok], d["obs_time"][ok], d["obs_embedding"][ok]
track_ids = sorted(set(int(t) for t in trk))
spans = {t: (tm[trk == t].min(), tm[trk == t].max()) for t in track_ids}
members = {t: np.where(trk == t)[0] for t in track_ids}

lines = []
def out(text=""):
    print(text)
    lines.append(text)


def same_person_scores(tracks):
    """Scores between two faces of the SAME track, at least MIN_GAP_S apart."""
    scores = []
    for t in tracks:
        idx = members[t]
        if len(idx) < 2:
            continue
        sim = emb[idx] @ emb[idx].T
        dt = np.abs(tm[idx][:, None] - tm[idx][None, :])
        iu = np.triu_indices(len(idx), k=1)
        keep = dt[iu] >= MIN_GAP_S
        scores.append(sim[iu][keep])
    return np.concatenate(scores) if scores else np.array([])


def different_person_scores(tracks):
    """Scores between faces of two DIFFERENT tracks that were alive together."""
    scores = []
    for a_pos, a in enumerate(tracks):
        for b in tracks[a_pos + 1:]:
            overlap = min(spans[a][1], spans[b][1]) - max(spans[a][0], spans[b][0])
            if overlap >= MIN_OVERLAP_S:
                scores.append((emb[members[a]] @ emb[members[b]].T).ravel())
    return np.concatenate(scores) if scores else np.array([])


def rates_at(threshold, pos, neg):
    return float((pos >= threshold).mean()), float((neg >= threshold).mean())


pos = same_person_scores(track_ids)
neg = different_person_scores(track_ids)
if len(pos) == 0 or len(neg) == 0:
    print("Not enough pairs to calibrate.")
    sys.exit(1)

out("THRESHOLD CALIBRATION (a measurement and a suggestion; nothing is changed)")
out(f"Source: {data_path}")
out(f"Faces used: {ok.sum()} that pass the quality filter, from {len(track_ids)} tracks\n")
out("How the pairs were made (no manual labels):")
out(f"  SAME person      : two faces of one track, >= {MIN_GAP_S:.0f} s apart        -> {len(pos):,} pairs")
out(f"  DIFFERENT people : faces of two tracks alive together >= {MIN_OVERLAP_S:.0f} s -> {len(neg):,} pairs\n")


def describe(name, v):
    p = lambda q: np.percentile(v, q)
    out(f"  {name:17s} min {v.min():.3f} | 1% {p(1):.3f} | 5% {p(5):.3f} | median {np.median(v):.3f} "
        f"| 95% {p(95):.3f} | 99% {p(99):.3f} | max {v.max():.3f}")


out("Similarity scores:")
describe("same person", pos)
describe("different people", neg)

out("\nWhat the current threshold does on these pairs:")
tar, far = rates_at(CURRENT_THRESHOLD, pos, neg)
out(f"  threshold {CURRENT_THRESHOLD}: same person accepted {100 * tar:.1f}% (TAR), "
    f"different people wrongly accepted {100 * far:.3f}% (FAR)")

out("\nThreshold needed for a target false-accept rate (FAR), and the true-accept rate (TAR) you get:")
suggestions = {}
for target in FAR_TARGETS:
    thr = float(np.quantile(neg, 1 - target))
    tar, far = rates_at(thr, pos, neg)
    reliable = len(neg) * target >= 10     # at least ~10 false accepts expected, else the estimate is shaky
    suggestions[target] = thr
    out(f"  FAR {target:g}: threshold {thr:.3f} -> TAR {100 * tar:.1f}%"
        + ("" if reliable else "   (too few different-people pairs to trust this row)"))

# Equal error rate: where false accepts and false rejects are equal.
grid = np.linspace(-0.2, 1.0, 1201)
gap = [abs((1 - (pos >= g).mean()) - (neg >= g).mean()) for g in grid]
eer_thr = float(grid[int(np.argmin(gap))])
out(f"  equal-error threshold (false rejects = false accepts): {eer_thr:.3f}")

# Stability: do the first and the second half of the video agree?
starts = {t: spans[t][0] for t in track_ids}
cut = float(np.median(list(starts.values())))
halves = {"first half": [t for t in track_ids if starts[t] < cut],
          "second half": [t for t in track_ids if starts[t] >= cut]}
out("\nDoes the answer hold in both halves of the video? (threshold for FAR 0.01 / 0.001)")
for name, tracks in halves.items():
    p_h, n_h = same_person_scores(tracks), different_person_scores(tracks)
    if len(p_h) and len(n_h):
        out(f"  {name:11s}: {len(tracks)} tracks, {len(n_h):,} different-people pairs -> "
            f"{np.quantile(n_h, 0.99):.3f} / {np.quantile(n_h, 0.999):.3f}")

# Pair FAR is not the same as the alarm rate of a real 1:N system.
out("\nWhy gallery size matters: every face is compared with EVERYONE enrolled.")
out("Chance that one UNKNOWN face is wrongly matched to somebody, at pair FAR = 0.001 (about 1 - (1-FAR)^N):")
for n in GALLERY_SIZES:
    out(f"  gallery of {n:5d} people: {100 * (1 - (1 - 1e-3) ** n):5.1f}%")

out("\nHONEST LIMITS:")
out("  - Same-person pairs come from ONE track in minutes of one camera: those faces look alike")
out("    (same clothes, light, nearby moments). Real re-identification (another day, another camera,")
out("    another angle) scores LOWER, so the TAR above is OPTIMISTIC.")
out("  - Different-person pairs are people seen at the same moment, which is a fair but small sample.")
out("  - Tracks are not perfect (see PROJECT_LOG.md): a split track makes a few 'same person' pairs")
out("    that count as different, and a mixed track the opposite.")
out("  - Treat the suggested thresholds as a starting range, then confirm with hand-labeled people")
out("    from several days/cameras and the real client footage before relying on any number.")

# ---- Picture: the two score distributions ------------------------------------
W, H, L, B = 900, 430, 50, 50
canvas = np.full((H, W, 3), 255, np.uint8)
bins = np.linspace(-0.2, 1.0, 61)
for values, color in ((neg, (60, 60, 220)), (pos, (60, 160, 60))):
    hist, _ = np.histogram(values, bins=bins)
    hist = hist / hist.max()
    for i, h in enumerate(hist):
        x0 = L + int((W - L - 20) * i / (len(bins) - 1))
        x1 = L + int((W - L - 20) * (i + 1) / (len(bins) - 1))
        cv2.rectangle(canvas, (x0, H - B - int((H - B - 40) * h)), (x1, H - B), color, -1)
overlay = canvas.copy()
def x_of(v):
    return L + int((W - L - 20) * (v + 0.2) / 1.2)
for value, label, color in ((CURRENT_THRESHOLD, "current 0.4", (0, 0, 0)),
                            (suggestions[1e-3], f"FAR 0.001: {suggestions[1e-3]:.2f}", (200, 100, 0))):
    cv2.line(canvas, (x_of(value), 30), (x_of(value), H - B), color, 2)
    cv2.putText(canvas, label, (x_of(value) + 4, 45 if color == (0, 0, 0) else 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)   # soften bars so lines stay visible
for tick in np.arange(-0.2, 1.01, 0.2):
    cv2.putText(canvas, f"{tick:.1f}", (x_of(tick) - 12, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
cv2.putText(canvas, "cosine similarity  (green = same person, red = different people)",
            (L, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(canvas, "Score distributions (each scaled to its own tallest bar)", (L, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

OUT_DIR.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT_DIR / "calibration.png"), canvas)
(OUT_DIR / "report.txt").write_text("\n".join(lines) + "\n")
print("\nSaved:", OUT_DIR / "report.txt", "and", OUT_DIR / "calibration.png")
