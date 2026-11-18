"""Quality report for the African Wildlife dataset (YOLO format).

Checks: split sizes, class counts, object sizes, images with no label,
boxes outside the image, and near-duplicate photos that cross a split.

Usage:  python scripts/check_dataset.py african-wildlife
"""
import os, sys, glob, collections, statistics
import numpy as np
from PIL import Image

CLASSES = ["buffalo", "elephant", "rhino", "zebra"]
SPLITS = ["train", "val", "test"]
DHASH_MAX = 8          # below this distance two photos are "the same"


def label_path(image_path):
    return image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def read_boxes(image_path):
    """Returns the boxes of one image as (class id, cx, cy, w, h), still 0..1."""
    path = label_path(image_path)
    if not os.path.exists(path):
        return None                      # no label file at all
    out = []
    for line in open(path):
        if line.strip():
            k, x, y, w, h = line.split()
            out.append((int(k), float(x), float(y), float(w), float(h)))
    return out


def dhash(path):
    """64-bit hash: compares every pixel with the pixel on its right."""
    a = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def report(root):
    images = {s: sorted(glob.glob(f"{root}/images/{s}/*")) for s in SPLITS}

    print(f"{'split':8}{'images':>8}{'objects':>9}   " + "  ".join(f"{c:>9}" for c in CLASSES))
    totals = collections.Counter()
    missing, empty, outside = [], [], 0
    sides = collections.defaultdict(list)
    per_image = []

    for s in SPLITS:
        counts = collections.Counter()
        for p in images[s]:
            boxes = read_boxes(p)
            if boxes is None:
                missing.append(p)
                continue
            if not boxes:
                empty.append(p)
            per_image.append(len(boxes))
            for k, x, y, w, h in boxes:
                counts[CLASSES[k]] += 1
                totals[CLASSES[k]] += 1
                sides[CLASSES[k]].append((w * h) ** 0.5)
                if x - w / 2 < -0.01 or y - h / 2 < -0.01 or x + w / 2 > 1.01 or y + h / 2 > 1.01:
                    outside += 1
        print(f"{s:8}{len(images[s]):>8}{sum(counts.values()):>9}   "
              + "  ".join(f"{counts[c]:>9}" for c in CLASSES))

    n_images = sum(len(images[s]) for s in SPLITS)
    print(f"{'total':8}{n_images:>8}{sum(totals.values()):>9}   "
          + "  ".join(f"{totals[c]:>9}" for c in CLASSES))

    print()
    print("images with no label file:", len(missing))
    print("images with an empty label:", len(empty))
    print("boxes outside the image   :", outside)
    print("objects per image: min", min(per_image),
          " median", statistics.median(per_image), " max", max(per_image))

    print()
    print(f"{'class':10}{'objects':>9}{'median box side, % of image':>30}")
    for c in CLASSES:
        print(f"{c:10}{len(sides[c]):>9}{statistics.median(sides[c]) * 100:>30.0f}")

    near_duplicates(images)


def near_duplicates(images):
    paths, where = [], {}
    for s in SPLITS:
        for p in images[s]:
            paths.append(p)
            where[p] = s

    H = np.array([dhash(p) for p in paths])
    pairs = []
    for i in range(len(paths)):
        distance = (H[i + 1:] != H[i]).sum(1)
        for k, d in enumerate(distance):
            if d <= DHASH_MAX:
                pairs.append((int(d), paths[i], paths[i + 1 + k]))
    pairs.sort()

    crossing = [p for p in pairs if where[p[1]] != where[p[2]]]
    kinds = collections.Counter(tuple(sorted((where[a], where[b]))) for d, a, b in pairs)

    print()
    print(f"near-duplicate pairs (dHash distance <= {DHASH_MAX}): {len(pairs)}")
    print("  by split pair:", dict(kinds))
    print("  crossing a split boundary:", len(crossing), " <-- these leak")
    for d, a, b in crossing[:10]:
        print(f"    d={d:>2}  {os.path.basename(a)} ({where[a]})"
              f"  <->  {os.path.basename(b)} ({where[b]})")
    if len(crossing) > 10:
        print(f"    ... and {len(crossing) - 10} more")
    return crossing


if __name__ == "__main__":
    report(sys.argv[1])
