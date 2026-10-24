import sys
from pathlib import Path

import cv2
import numpy as np

FACES_DIR = Path("videos/faces")
OUT_PATH = FACES_DIR / "comparison_grid.jpg"

CELL_HEIGHT = 220   # each crop resized to this height
LABEL_HEIGHT = 24   # space for the filename label above each crop
PADDING = 6         # gap between crops
TAG_WIDTH = 100      # width of the "Person A" tag on the left


def load_and_resize(path, height):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, height))


def make_row(names, label):
    cells = []
    for name in names:
        img = load_and_resize(FACES_DIR / name, CELL_HEIGHT)
        if img is None:
            print("Could not read:", name)
            continue
        canvas = np.zeros((CELL_HEIGHT + LABEL_HEIGHT, img.shape[1], 3), dtype=np.uint8)
        canvas[LABEL_HEIGHT:, :, :] = img
        short_name = name.split("_w")[0]  # drop the "_w56.jpg" width suffix
        cv2.putText(canvas, short_name, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(canvas)

    if not cells:
        return None

    row_height = CELL_HEIGHT + LABEL_HEIGHT
    total_w = sum(c.shape[1] for c in cells) + PADDING * (len(cells) - 1)
    row = np.zeros((row_height, total_w, 3), dtype=np.uint8)
    x = 0
    for c in cells:
        row[:, x:x + c.shape[1], :] = c
        x += c.shape[1] + PADDING

    tag = np.zeros((row_height, TAG_WIDTH, 3), dtype=np.uint8)
    cv2.putText(tag, label, (4, row_height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return np.hstack([tag, row])


# Split arguments into groups, one group = one person.
groups, current = [], []
for arg in sys.argv[1:]:
    if arg == "vs":
        groups.append(current)
        current = []
    else:
        current.append(arg)
groups.append(current)

if not any(groups):
    print("Give crop file names, grouped with 'vs'. Example:")
    print("  python 08_compare_grid.py a1.jpg a2.jpg vs b1.jpg b2.jpg")
    sys.exit(1)

rows = []
for i, group in enumerate(groups):
    label = f"Person {chr(ord('A') + i)}"
    row = make_row(group, label)
    if row is not None:
        rows.append(row)

if not rows:
    print("No valid crops found.")
    sys.exit(1)

max_width = max(r.shape[1] for r in rows)
final_rows = []
for row in rows:
    if row.shape[1] < max_width:
        pad = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
        row = np.hstack([row, pad])
    final_rows.append(row)
    final_rows.append(np.full((2, max_width, 3), 60, dtype=np.uint8))  # divider

grid = np.vstack(final_rows[:-1])  # drop the trailing divider
cv2.imwrite(str(OUT_PATH), grid)
print("Saved comparison grid:", OUT_PATH)
print("Rows:", len(rows))
