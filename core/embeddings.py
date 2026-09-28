import numpy as np

from core.models import get_face_app


def extract_best_face(img_bgr: np.ndarray, use_gpu: bool = False):
    """Detect faces in an image and return data for the largest one.

    Returns None if no face is found. Otherwise a dict with:
      embedding : 512-D unit-length vector (the face fingerprint)
      det_score : detector confidence (0 to 1)
      bbox      : [x1, y1, x2, y2] in pixels
      crop      : face crop with a small margin, for humans to look at
      num_faces : how many faces were found in the image
    """
    app = get_face_app(use_gpu)
    faces = app.get(img_bgr)  # detect -> align -> embed, all in one call
    if not faces:
        return None

    # Keep the largest face by box area.
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    # Build a crop with a 20% margin, clipped to the image borders.
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    margin_x = int((x2 - x1) * 0.2)
    margin_y = int((y2 - y1) * 0.2)
    cx1 = max(0, x1 - margin_x)
    cy1 = max(0, y1 - margin_y)
    cx2 = min(w, x2 + margin_x)
    cy2 = min(h, y2 + margin_y)
    crop = img_bgr[cy1:cy2, cx1:cx2].copy()

    return {
        "embedding": face.normed_embedding,
        "det_score": float(face.det_score),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "crop": crop,
        "num_faces": len(faces),
    }
