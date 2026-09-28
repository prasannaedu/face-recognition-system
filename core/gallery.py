import os
import re
from pathlib import Path

import cv2
import numpy as np

# gallery/ sits next to core/, at the project root.
GALLERY_DIR = Path(__file__).resolve().parent.parent / "gallery"

_VALID_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_id(person_id: str) -> None:
    if not _VALID_ID.match(person_id):
        raise ValueError(
            "person_id may only contain letters, numbers, '_' and '-'"
        )


def enroll(person_id: str, embedding: np.ndarray, crop: np.ndarray) -> int:
    """Save one embedding + face crop for a person. Returns the new index."""
    _check_id(person_id)
    person_dir = GALLERY_DIR / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    index = len(list(person_dir.glob("emb_*.npy")))

    # Write to a temporary name first, then rename: a crash mid-write
    # can never leave a half-written file in the gallery.
    emb_path = person_dir / f"emb_{index}.npy"
    tmp_emb = person_dir / f"tmp_emb_{index}.npy"
    np.save(tmp_emb, embedding.astype(np.float32))
    os.replace(tmp_emb, emb_path)

    crop_path = person_dir / f"crop_{index}.jpg"
    tmp_crop = person_dir / f"tmp_crop_{index}.jpg"
    cv2.imwrite(str(tmp_crop), crop)
    os.replace(tmp_crop, crop_path)

    return index


def load_gallery() -> dict:
    """Load every enrolled person: {person_id: array of shape (n, 512)}."""
    gallery = {}
    if not GALLERY_DIR.exists():
        return gallery

    for person_dir in sorted(GALLERY_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        files = sorted(person_dir.glob("emb_*.npy"))
        if files:
            gallery[person_dir.name] = np.stack([np.load(f) for f in files])
    return gallery


def search(query_embedding: np.ndarray, gallery: dict, top_k: int = 3):
    """Compare a face against everyone enrolled.

    Each person's score is their BEST-matching embedding (MaxMember idea).
    Returns [(person_id, score), ...] sorted from best to worst.
    We do NOT apply a threshold here; the caller decides.
    """
    results = []
    for person_id, embeddings in gallery.items():
        # Unit-length vectors: dot product == cosine similarity.
        scores = embeddings @ query_embedding
        results.append((person_id, float(scores.max())))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]
