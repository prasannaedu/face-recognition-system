import itertools
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Load the same pretrained model pack as before.
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# 2. Ground truth: which photo belongs to which person.
identity = {
    "face1.jpeg": "A",
    "face-2.jpeg": "A",
    "face3.jpeg": "B",
    "face-4.jpeg": "C",
    "face-5.jpeg": "C",
}

# 3. For each photo: detect -> align -> embed (all done inside app.get).
embeddings = {}
for name in identity:
    path = f"test_images/{name}"
    img = cv2.imread(path)
    if img is None:
        print("Could not read:", path)
        continue

    faces = app.get(img)
    if len(faces) == 0:
        print("No face found in:", name)
        continue

    # If there are several faces, keep the largest one.
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embeddings[name] = face.normed_embedding
    print(f"{name}: faces={len(faces)}, confidence={face.det_score:.3f}, "
          f"embedding size={face.normed_embedding.shape[0]}")

# 4. Cosine similarity for every pair (dot product of unit vectors).
print("\nSimilarity for every pair:")
for a, b in itertools.combinations(embeddings.keys(), 2):
    sim = float(np.dot(embeddings[a], embeddings[b]))
    truth = "SAME person" if identity[a] == identity[b] else "different people"
    print(f"{a:12s} vs {b:12s} : {sim:6.3f}   ({truth})")
