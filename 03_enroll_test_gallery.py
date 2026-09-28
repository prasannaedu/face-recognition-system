import cv2

from core.embeddings import extract_best_face
from core.gallery import GALLERY_DIR, enroll

# Ground truth: which photo we enroll for each person.
# The other photos (face-2, face-5) are held back for the recognition test.
to_enroll = {
    "person_a": "test_images/face1.jpeg",
    "person_b": "test_images/face3.jpeg",
    "person_c": "test_images/face-4.jpeg",
}

for person_id, path in to_enroll.items():
    # Skip people who are already in the gallery (avoids duplicates).
    if (GALLERY_DIR / person_id).exists():
        print(f"{person_id}: already enrolled, skipping")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"{person_id}: could not read {path}")
        continue

    result = extract_best_face(img)
    if result is None:
        print(f"{person_id}: no face found in {path}")
        continue

    index = enroll(person_id, result["embedding"], result["crop"])
    print(
        f"{person_id}: enrolled from {path} "
        f"(confidence {result['det_score']:.3f}, saved as emb_{index}.npy)"
    )

print("\nDone. Gallery folder:", GALLERY_DIR)
