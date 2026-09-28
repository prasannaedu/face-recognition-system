"""FastAPI wrapper around the EXISTING face-recognition pipeline.

This file adds NO recognition logic of its own. It only:
  1. accepts an uploaded image,
  2. calls the existing core functions (extract_best_face, load_gallery, search),
  3. applies the SAME decision rule as 04_recognize.py (best_score >= THRESHOLD),
  4. returns the result as JSON for the frontend.

Run from the project root:
    uvicorn api.main:app --reload
"""

import base64
import re
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.embeddings import extract_best_face
from core.gallery import GALLERY_DIR, load_gallery, search

# person_id may only contain letters, numbers, '_' and '-' (same rule the
# gallery uses). This also prevents path traversal in the reference-image route.
_VALID_ID = re.compile(r"^[A-Za-z0-9_-]+$")

# Mirrors the PROVISIONAL threshold in 04_recognize.py. Same value, same meaning:
# it only demonstrates the MATCH / UNKNOWN boundary and will be calibrated on
# real footage later. We define it here (rather than importing 04_recognize.py,
# which runs recognition at import time) so importing this module has no side
# effects.
THRESHOLD = 0.4
TOP_K = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(title="Face Recognition System", version="0.1.0")

# Load the gallery once at startup and reuse it for every request, so we do not
# re-read the .npy files on each call. The InsightFace model itself is already
# cached inside core.models.get_face_app().
_GALLERY = load_gallery()


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    """Uniform error shape the frontend can rely on."""
    return JSONResponse(status_code=status_code, content={"error": code, "message": message})


def _crop_to_data_url(crop: np.ndarray) -> str | None:
    """Encode the detected face crop as a base64 JPEG data URL for preview.

    Only the human-viewable crop is sent — never the 512-D embedding and never
    any filesystem path.
    """
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/api/health")
def health() -> dict:
    """Simple check: is the server up and how many identities are enrolled?"""
    return {"status": "ok", "enrolled_identities": sorted(_GALLERY.keys())}


@app.get("/gallery/{person_id}/crop")
def gallery_crop(person_id: str):
    """Serve the enrolled reference face crop for one person.

    Used by the UI to show the matched person's stored photo next to the query.
    Only the human-viewable crop is served — never the embedding.
    """
    if not _VALID_ID.match(person_id):
        return _error(400, "bad_id", "Invalid person id.")

    person_dir = GALLERY_DIR / person_id
    crops = sorted(person_dir.glob("crop_*.jpg")) if person_dir.is_dir() else []
    if not crops:
        return _error(404, "no_crop", "No reference image for this person.")

    return FileResponse(str(crops[0]), media_type="image/jpeg")


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)) -> JSONResponse:
    # 1. Basic upload validation.
    if file is None or not file.filename:
        return _error(400, "no_image", "No image was uploaded.")

    raw = await file.read()
    if not raw:
        return _error(400, "empty_file", "The uploaded file is empty.")

    # 2. Decode the bytes into an image (BGR, like cv2.imread in the scripts).
    buffer = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img is None:
        return _error(400, "invalid_image", "The file could not be read as an image.")

    # 3. The gallery must have someone to compare against.
    if not _GALLERY:
        return _error(503, "empty_gallery", "No identities are enrolled in the gallery yet.")

    # 4. Reuse the EXISTING pipeline: detect -> align -> embed -> best face.
    result = extract_best_face(img)
    if result is None:
        return _error(422, "no_face", "No face was detected in the image.")

    # 5. Reuse the EXISTING search (cosine similarity, MaxMember, top-K).
    matches = search(result["embedding"], _GALLERY, top_k=TOP_K)
    if not matches:
        return _error(500, "search_failed", "Recognition produced no matches.")

    best_id, best_score = matches[0]
    runner_up, runner_up_score = matches[1] if len(matches) > 1 else (None, None)
    gap = round(best_score - runner_up_score, 4) if runner_up_score is not None else None

    # 6. SAME decision rule as 04_recognize.py.
    is_match = best_score >= THRESHOLD

    payload = {
        "status": "MATCH" if is_match else "UNKNOWN",
        "identity": best_id if is_match else None,
        "match_score": round(best_score, 4),
        "runner_up": runner_up,
        "runner_up_score": round(runner_up_score, 4) if runner_up_score is not None else None,
        "gap": gap,
        "threshold": THRESHOLD,
        "detection_confidence": round(result["det_score"], 4),
        "num_faces": result["num_faces"],
        "matches": [
            {"rank": i, "person_id": pid, "score": round(score, 4)}
            for i, (pid, score) in enumerate(matches, start=1)
        ],
        "face_crop": _crop_to_data_url(result["crop"]),
    }
    return JSONResponse(content=payload)


# Serve the static frontend at "/". Mounted last so the API routes above win.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
