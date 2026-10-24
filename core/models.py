from functools import lru_cache

from insightface.app import FaceAnalysis


@lru_cache(maxsize=1)
def get_face_app(use_gpu: bool = False) -> FaceAnalysis:
    """Load the InsightFace model pack once and reuse it everywhere.

    buffalo_l is a pretrained bundle: face detector (SCRFD), 5-point
    landmarks, alignment, and the ArcFace embedding model.

    We load ONLY the models we use:
      detection        : SCRFD (face box + 5 landmarks)
      recognition      : ArcFace (512-D embedding)
      landmark_3d_68   : head angle (yaw/pitch) for the quality filter
    The pack's 106-point landmark model and gender/age model are skipped:
    nothing here uses them, so running them only wastes time, and guessing
    people's gender and age is not something this system should do.
    """
    if use_gpu:
        import onnxruntime

        # CUDA/cuDNN installed with pip (onnxruntime-gpu[cuda,cudnn]) must be
        # loaded first, or onnxruntime cannot find them. Only runs on the GPU path.
        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls(directory="")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition", "landmark_3d_68"],
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app
