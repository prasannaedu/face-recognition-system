from functools import lru_cache

from insightface.app import FaceAnalysis


@lru_cache(maxsize=1)
def get_face_app(use_gpu: bool = False) -> FaceAnalysis:
    """Load the InsightFace model pack once and reuse it everywhere.

    buffalo_l is a pretrained bundle: face detector (SCRFD), 5-point
    landmarks, alignment, and the ArcFace embedding model.
    """
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app
