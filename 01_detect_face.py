import sys
import cv2
from insightface.app import FaceAnalysis

# 1. Load the pretrained model pack (downloads on first run).
#    CPU is enough for Phase 1; we'll use the GPU later for video.
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# 2. Read one image from disk.
image_path = "test_images/face1.jpeg"
img = cv2.imread(image_path)
if img is None:
    print("Could not read image:", image_path)
    sys.exit(1)

# 3. Detect faces.
faces = app.get(img)

print("Image:", image_path)
print("Image size (height, width):", img.shape[:2])
print("Faces found:", len(faces))

for i, face in enumerate(faces):
    print(f"--- Face {i + 1} ---")
    print("Box [x1, y1, x2, y2]:", face.bbox.astype(int))
    print("Detection confidence:", round(float(face.det_score), 3))
    print("5 landmarks (x, y):")
    print(face.kps.astype(int))
