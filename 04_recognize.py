import cv2

from core.embeddings import extract_best_face
from core.gallery import load_gallery, search

# PROVISIONAL threshold, only to demonstrate the "unknown" logic.
# The real value will be calibrated on the company's own footage.
THRESHOLD = 0.4

# Query photos the gallery has never seen, with the expected answer.
queries = {
    "test_images/face-2.jpeg": "person_a",
    "test_images/face-5.jpeg": "person_c",
}

gallery = load_gallery()
print("People in gallery:", list(gallery.keys()), "\n")

for path, expected in queries.items():
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        continue

    result = extract_best_face(img)
    if result is None:
        print(f"{path}: no face found")
        continue

    matches = search(result["embedding"], gallery, top_k=3)
    best_id, best_score = matches[0]

    if best_score >= THRESHOLD:
        decision = best_id
    else:
        decision = "UNKNOWN"

    print(f"Query: {path}  (expected: {expected})")
    for rank, (person_id, score) in enumerate(matches, start=1):
        print(f"  #{rank}  {person_id:10s} score = {score:.3f}")
    verdict = "CORRECT" if decision == expected else "WRONG"
    print(f"  Decision: {decision}  ->  {verdict}\n")
