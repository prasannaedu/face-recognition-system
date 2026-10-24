import sys
import itertools
from pathlib import Path

import numpy as np

# Usage: each PERSON is a group of crop files; groups are separated by "vs".
#   python 07_compare_crops.py a1.jpg a2.jpg a3.jpg vs b1.jpg b2.jpg
# Crops are the files inside videos/faces/.
DATA = Path("videos/faces/faces_data.npz")

if not DATA.exists():
    print("Missing", DATA, "- run 06_extract_faces.py first.")
    sys.exit(1)

data = np.load(DATA)
files = [str(f) for f in data["files"]]
emb = data["embeddings"]
widths = data["widths"]
sharp = data["sharpness"]
index = {name: i for i, name in enumerate(files)}

# Split the arguments into groups (one group = one person).
groups, current = [], []
for arg in sys.argv[1:]:
    if arg == "vs":
        groups.append(current)
        current = []
    else:
        current.append(arg)
groups.append(current)

if len(groups) < 1 or not any(groups):
    print("Give at least two crops. Example:")
    print("  python 07_compare_crops.py a1.jpg a2.jpg vs b1.jpg")
    sys.exit(1)

items = []  # (label, name, embedding row)
for g, group in enumerate(groups):
    label = chr(ord("A") + g)
    for path in group:
        name = Path(path).name
        if name not in index:
            print("Crop not found in faces_data.npz:", name)
            sys.exit(1)
        items.append((label, name, index[name]))

print("Crops used:")
for n, (label, name, i) in enumerate(items, start=1):
    print(f"  #{n}  person {label}  {name}  "
          f"(width {widths[i]:.0f} px, sharpness {sharp[i]:.0f})")

# Similarity between every pair (dot product of unit-length vectors).
print("\nSimilarity for every pair:")
same, diff = [], []
for (a, b) in itertools.combinations(range(len(items)), 2):
    la, na, ia = items[a]
    lb, nb, ib = items[b]
    sim = float(np.dot(emb[ia], emb[ib]))
    kind = "SAME person" if la == lb else "different people"
    (same if la == lb else diff).append(sim)
    print(f"  #{a + 1} vs #{b + 1}: {sim:6.3f}   ({kind})")

print("\nSummary:")
if same:
    print(f"  SAME person pairs      : lowest {min(same):.3f}, "
          f"average {np.mean(same):.3f}, highest {max(same):.3f}")
if diff:
    print(f"  DIFFERENT people pairs : lowest {min(diff):.3f}, "
          f"average {np.mean(diff):.3f}, highest {max(diff):.3f}")
if same and diff:
    gap = min(same) - max(diff)
    print(f"  Gap (lowest same - highest different): {gap:.3f}")
    print("  A positive gap means the two groups are separable "
          "on these crops." if gap > 0 else
          "  A negative gap means some same-person pairs score BELOW some "
          "different-person pairs: there is no safe threshold on these crops.")
