"""Frame-quality check for a detected face.

Decides whether a face is good enough to be used for enrollment or
recognition. It does NOT detect or match faces; it only looks at what
InsightFace already computed (head angle, landmarks, detection score)
plus one cheap image measurement (sharpness).

All input stays 2D: the head angle comes from buffalo_l's landmark model
looking at the 2D image. No depth data and no new model is used.
"""

import cv2
import numpy as np
from insightface.utils.face_align import norm_crop

# PROVISIONAL limits. Like the 0.4 recognition threshold, these are starting
# points to be calibrated on the real footage (see 09_quality_report.py).
MAX_YAW_DEG = 30.0       # turned left/right more than this -> reject
MAX_PITCH_DEG = 30.0     # looking up/down more than this -> reject
MIN_SHARPNESS = 50.0     # blur limit at 112x112 (~5th percentile of the railway footage)
MIN_DET_SCORE = 0.6      # same default as 06_extract_faces.py

ALIGNED_SIZE = 112       # ArcFace input size; sharpness is measured at this size


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_quality(img_bgr: np.ndarray, face) -> dict:
    """Score one InsightFace face found in `img_bgr` (the FULL image/frame).

    Returns a dict with the raw measurements, a `passed` flag, and a list of
    human-readable `reasons` when it failed.

    KNOWN LIMITATION: occlusion (a hand or object over the face) is NOT
    filtered. On the railway footage no cheap 2D proxy (detection score,
    embedding norm, mouth/eye sharpness) separated hand-covered faces from
    normal ones without also rejecting many good faces, so emb_norm and
    mouth_eye_ratio are only recorded for further study, never used to reject.
    Low sharpness can also come from a dark, low-contrast face, not only blur.

    Measurements:
      yaw, pitch, roll : head angle in degrees (roll is reported, not used:
                         alignment already straightens a tilted head)
      sharpness        : Laplacian variance of the aligned 112x112 face, so
                         crops of different sizes are measured fairly
      det_score        : detector confidence
      emb_norm         : length of the raw embedding (occlusion proxy, experimental)
      mouth_eye_ratio  : sharpness of the mouth band / eye band (occlusion
                         proxy, experimental)
    """
    # Align to the same 112x112 canonical face that ArcFace sees.
    aligned = norm_crop(img_bgr, face.kps, ALIGNED_SIZE)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    sharp = _laplacian_var(gray)

    # Rows of the aligned template: eyes sit near y=51, mouth near y=92.
    eye_band = _laplacian_var(gray[30:62])
    mouth_band = _laplacian_var(gray[78:108])
    mouth_eye_ratio = mouth_band / eye_band if eye_band > 0 else 0.0

    pose = getattr(face, "pose", None)  # [pitch, yaw, roll] in degrees
    if pose is None:
        pitch = yaw = roll = None
    else:
        pitch, yaw, roll = (float(v) for v in pose)

    reasons = []
    if yaw is None:
        reasons.append("head pose unavailable")
    else:
        if abs(yaw) > MAX_YAW_DEG:
            reasons.append(f"yaw {abs(yaw):.0f} deg > {MAX_YAW_DEG:.0f}")
        if abs(pitch) > MAX_PITCH_DEG:
            reasons.append(f"pitch {abs(pitch):.0f} deg > {MAX_PITCH_DEG:.0f}")
    if sharp < MIN_SHARPNESS:
        reasons.append(f"sharpness {sharp:.0f} < {MIN_SHARPNESS:.0f}")
    if float(face.det_score) < MIN_DET_SCORE:
        reasons.append(f"detection {float(face.det_score):.2f} < {MIN_DET_SCORE:.2f}")

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "sharpness": sharp,
        "det_score": float(face.det_score),
        "emb_norm": float(np.linalg.norm(face.embedding)),
        "mouth_eye_ratio": float(mouth_eye_ratio),
        "passed": not reasons,
        "reasons": reasons,
    }
