"""A simple face tracker: "the same person across video frames".

WHY: in a video the same person shows up in many frames. Without tracking we
would treat every frame as a new face. With it, each person becomes one TRACK,
and we keep only the best few faces of that track (frontal, sharp, big) to use
for enrollment or recognition. This is the "best frame per person" idea
(MaxMember) from the Person Re-ID app.

HOW (kept deliberately simple, no extra dependency beyond scipy):
  1. For each existing track, predict where its face should be now (last
     position + a little of its recent movement).
  2. Build a cost for every (track, new face) pair: how far apart they are,
     measured in face widths, plus a small penalty if the ArcFace embeddings
     look different.
  3. Pair them up with the Hungarian algorithm (best overall assignment),
     ignoring pairs that are too far apart or very different in size.
  4. A track that finds no face for more than `max_missed` updates is closed,
     so a person may be missed a few times (blur, head turn) and still keep
     the same track id.

LIMITS: geometry-based, so two people who cross paths closely can swap ids.
Check the tracks by eye (the strips written by 10_track_faces.py).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6  # cost for "not allowed to match"

# Rank-score scales (heuristics): a face counts as "full marks" at these values.
ANGLE_SCALE_DEG = 60.0   # frontalness goes to 0 at this head angle
SHARP_CAP = 300.0        # sharpness at 112x112 above this earns no extra credit
WIDTH_CAP = 100.0        # face width (px) above this earns no extra credit


def crop_with_margin(frame: np.ndarray, bbox, margin: float = 0.2) -> np.ndarray:
    """Face crop with a margin around the box, clipped to the frame borders."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    return frame[max(0, y1 - my):min(h, y2 + my),
                 max(0, x1 - mx):min(w, x2 + mx)].copy()


def rank_score(q: dict, width: float) -> float:
    """How good is this face for recognition? Higher = better (0 to 1).

    = frontalness x sharpness x size. A frontal, sharp, big face scores near 1;
    a turned, blurry or tiny one scores near 0. Faces that pass the quality
    filter always outrank ones that fail it (see Track.add).
    """
    if q["yaw"] is None:
        frontal = 0.0
    else:
        frontal = max(0.0, 1.0 - max(abs(q["yaw"]), abs(q["pitch"])) / ANGLE_SCALE_DEG)
    sharp = min(q["sharpness"], SHARP_CAP) / SHARP_CAP
    size = min(width, WIDTH_CAP) / WIDTH_CAP
    return frontal * sharp * size


class Track:
    """One person followed across frames."""

    def __init__(self, track_id: int, top_k: int, keep_all_crops: bool = True,
                 max_obs: int = None):
        self.id = track_id
        self.obs = []        # observations in time order (all, or the last max_obs)
        self.best = []       # the top_k best observations (see add)
        self.missed = 0      # updates in a row with no matching face
        self.closed = False
        self.n_obs = 0       # how many faces this track has seen in total
        self._top_k = top_k
        self._keep_all_crops = keep_all_crops
        self._max_obs = max_obs

    @property
    def last(self) -> dict:
        return self.obs[-1]

    @property
    def hits(self) -> int:
        return self.n_obs

    def add(self, ob: dict) -> None:
        self.obs.append(ob)
        self.n_obs += 1
        self.missed = 0
        self.best.append(ob)
        # Passing the quality filter beats everything; then the rank score.
        self.best.sort(key=lambda o: (o["quality"]["passed"], o["rank"]), reverse=True)
        dropped = self.best[self._top_k:]
        del self.best[self._top_k:]
        if not self._keep_all_crops:
            # Live use: only the best faces keep their picture, so memory stays flat.
            for o in dropped:
                o["crop"] = None
        if self._max_obs and len(self.obs) > self._max_obs:
            del self.obs[:len(self.obs) - self._max_obs]

    def predicted_center(self, frame_idx: int) -> np.ndarray:
        """Where the face should be at frame_idx: last position + half the speed."""
        last = self.last
        if len(self.obs) < 2:
            return last["center"]
        prev = self.obs[-2]
        gap = max(1, last["frame"] - prev["frame"])
        velocity = (last["center"] - prev["center"]) / gap
        return last["center"] + 0.5 * velocity * (frame_idx - last["frame"])

    def mean_embedding(self, n: int = 5) -> np.ndarray:
        """Average embedding of the last n observations (unit length)."""
        m = np.mean([o["embedding"] for o in self.obs[-n:]], axis=0)
        return m / (np.linalg.norm(m) + 1e-9)


class FaceTracker:
    def __init__(self, max_missed: int = 5, max_dist: float = 2.0,
                 appearance_weight: float = 0.3, top_k: int = 3,
                 keep_all_crops: bool = True, max_obs: int = None):
        """
        max_missed        : a track survives this many updates with no face
        max_dist          : biggest allowed jump, in face widths (grows a bit
                            for each missed update)
        appearance_weight : how much a different-looking embedding adds to the cost
        top_k             : how many best faces each track remembers
        keep_all_crops    : False for live use: only the top_k faces keep a picture
        max_obs           : keep only the last max_obs observations per track
                            (None = all). Both options exist so a stream that
                            runs for hours does not grow memory without limit.
        """
        self.max_missed = max_missed
        self.max_dist = max_dist
        self.appearance_weight = appearance_weight
        self.top_k = top_k
        self.keep_all_crops = keep_all_crops
        self.max_obs = max_obs
        self.tracks = []     # every track ever created (open and closed)
        self._next_id = 1

    def _observation(self, det: dict, frame_idx: int, time_s: float) -> dict:
        x1, y1, x2, y2 = det["bbox"]
        width = float(x2 - x1)
        return {
            "frame": frame_idx,
            "time": time_s,
            "bbox": [float(v) for v in det["bbox"]],
            "center": np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]),
            "width": width,
            "det_score": det["det_score"],
            "embedding": det["embedding"],
            "quality": det["quality"],
            "crop": det.get("crop"),
            "rank": rank_score(det["quality"], width),
        }

    def update(self, detections: list, frame_idx: int, time_s: float) -> None:
        """Feed the faces found in ONE frame (call once per processed frame).

        Each detection is a dict with: bbox [x1,y1,x2,y2], embedding (unit
        length), quality (from core.quality.assess_quality), det_score, and
        optionally crop.
        """
        obs = [self._observation(d, frame_idx, time_s) for d in detections]
        active = [t for t in self.tracks if not t.closed]
        matched_tracks, matched_obs = set(), set()

        if active and obs:
            cost = np.full((len(active), len(obs)), BIG)
            for i, t in enumerate(active):
                predicted = t.predicted_center(frame_idx)
                track_w = t.last["width"]
                gate = self.max_dist + 0.5 * t.missed
                reference = t.mean_embedding()
                for j, o in enumerate(obs):
                    ratio = o["width"] / max(track_w, 1e-6)
                    if ratio < 0.5 or ratio > 2.0:      # too different in size
                        continue
                    dist = np.linalg.norm(o["center"] - predicted) / max(track_w, o["width"])
                    if dist > gate:                      # jumped too far
                        continue
                    similarity = float(reference @ o["embedding"])
                    cost[i, j] = dist + self.appearance_weight * (1.0 - similarity)

            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] >= BIG:
                    continue
                active[i].add(obs[j])
                matched_tracks.add(i)
                matched_obs.add(j)

        # Tracks that found nothing: count the miss, close them if too many.
        for i, t in enumerate(active):
            if i not in matched_tracks:
                t.missed += 1
                if t.missed > self.max_missed:
                    t.closed = True

        # Faces that matched no track start a new one.
        for j, o in enumerate(obs):
            if j not in matched_obs:
                t = Track(self._next_id, self.top_k, self.keep_all_crops, self.max_obs)
                self._next_id += 1
                t.add(o)
                self.tracks.append(t)

    def pop_closed(self) -> list:
        """Remove and return the tracks that have ended (for live use: save them,
        then let them go, so finished tracks do not pile up in memory)."""
        done = [t for t in self.tracks if t.closed]
        self.tracks = [t for t in self.tracks if not t.closed]
        return done

    def finish(self) -> list:
        """Close everything (call at the end of the video). Returns all tracks."""
        for t in self.tracks:
            t.closed = True
        return self.tracks
