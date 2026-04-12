"""Convert RescueNet semantic segmentation masks to YOLO-seg polygon format.

RescueNet masks are PNG images where each pixel value represents a class ID (1-11).
YOLO-seg format requires polygon coordinates per instance in .txt label files:
    class_id x1 y1 x2 y2 ... xn yn  (normalized 0-1)

For semantic segmentation classes (water, road, etc.), each connected component
of the same class becomes a separate polygon instance.

Usage (run on Kaggle kernel via kaggle_exec):
    This script is designed to be sent to the Kaggle kernel for execution.

RescueNet classes:
    0: Background (skip)
    1: Water
    2: Building No Damage
    3: Building Minor Damage
    4: Building Major Damage
    5: Building Total Destruction
    6: Road-Clear
    7: Road-Blocked
    8: Vehicle
    9: Tree (skip — not SAR-relevant, too noisy)
    10: Pool (skip — not SAR-relevant)

We remap to YOLO class IDs (0-indexed, excluding skipped classes):
    0: Water
    1: Building No Damage
    2: Building Minor Damage
    3: Building Major Damage
    4: Building Total Destruction
    5: Road-Clear
    6: Road-Blocked
    7: Vehicle
"""

# This code is meant to be run on the Kaggle kernel.
# It will be sent via kaggle_exec.run_code().

CONVERT_CODE = '''
import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json

# RescueNet pixel ID → YOLO class ID mapping
# Skip: 0 (background), 10 (tree), 11 (pool)
CLASS_MAP = {
    2: 0,   # Water → 0
    3: 1,   # Building No Damage → 1
    4: 2,   # Building Minor Damage → 2
    5: 3,   # Building Major Damage → 3
    6: 4,   # Building Total Destruction → 4
    7: 5,   # Road-Clear → 5
    8: 6,   # Road-Blocked → 6
    9: 7,   # Vehicle → 7
}

YOLO_NAMES = [
    'water',
    'building-no-damage',
    'building-minor-damage',
    'building-major-damage',
    'building-total-destruction',
    'road-clear',
    'road-blocked',
    'vehicle',
]

MIN_CONTOUR_AREA = 100  # Skip tiny fragments (noise)
MIN_POLYGON_POINTS = 3  # Minimum vertices for a valid polygon
EPSILON_FACTOR = 0.002  # Contour approximation factor (relative to perimeter)

SRC_ROOT = Path("/kaggle/working/rescuenet_raw/RescueNet")
DST_ROOT = Path("/kaggle/working/rescuenet_yolo")

SPLIT_MAP = {
    "train": ("train-org-img", "train-label-img"),
    "val": ("val-org-img", "val-label-img"),
    "test": ("test-org-img", "test-label-img"),
}


def mask_to_yolo_polygons(mask_path, img_h, img_w):
    """Convert a semantic mask PNG to YOLO polygon lines."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    lines = []
    for pixel_id, yolo_cls in CLASS_MAP.items():
        binary = (mask == pixel_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            # Approximate to reduce points
            epsilon = EPSILON_FACTOR * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) < MIN_POLYGON_POINTS:
                continue

            # Normalize coordinates
            points = approx.squeeze()
            if points.ndim != 2:
                continue

            coords = []
            for x, y in points:
                coords.append(f"{x / img_w:.6f}")
                coords.append(f"{y / img_h:.6f}")

            line = f"{yolo_cls} " + " ".join(coords)
            lines.append(line)

    return lines


def process_split(split_name):
    org_dir_name, lab_dir_name = SPLIT_MAP[split_name]
    org_dir = SRC_ROOT / split_name / org_dir_name
    lab_dir = SRC_ROOT / split_name / lab_dir_name

    if not org_dir.exists():
        print(f"  [skip] {org_dir} not found")
        return 0, 0

    dst_img_dir = DST_ROOT / split_name / "images"
    dst_lbl_dir = DST_ROOT / split_name / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    org_files = sorted(org_dir.glob("*.jpg")) + sorted(org_dir.glob("*.png"))
    converted = 0
    skipped = 0

    for img_path in org_files:
        stem = img_path.stem
        # RescueNet naming: image = "12345.jpg", label = "12345_lab.png"
        mask_path = lab_dir / f"{stem}_lab.png"
        if not mask_path.exists():
            # Try without _lab suffix
            mask_path = lab_dir / f"{stem}.png"
        if not mask_path.exists():
            skipped += 1
            continue

        # Get image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        # Convert mask to polygons
        lines = mask_to_yolo_polygons(mask_path, h, w)

        # Symlink image (save disk space)
        dst_img = dst_img_dir / img_path.name
        if not dst_img.exists():
            os.symlink(str(img_path), str(dst_img))

        # Write label
        dst_lbl = dst_lbl_dir / f"{stem}.txt"
        dst_lbl.write_text("\\n".join(lines) if lines else "")

        converted += 1

    return converted, skipped


def main():
    print("=== RescueNet → YOLO-seg conversion ===")

    stats = {}
    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        converted, skipped = process_split(split)
        stats[split] = {"converted": converted, "skipped": skipped}
        print(f"  {split}: {converted} converted, {skipped} skipped")

    # Write data.yaml
    yaml_path = DST_ROOT / "data.yaml"
    yaml_path.write_text(f"""path: {DST_ROOT}
train: train/images
val: val/images
test: test/images

nc: {len(YOLO_NAMES)}
names: {YOLO_NAMES}
""")
    print(f"Wrote {yaml_path}")

    # Write stats
    stats_path = DST_ROOT / "conversion_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")
    print(json.dumps(stats, indent=2))
    print("=== Done ===")


main()
'''

if __name__ == "__main__":
    print(CONVERT_CODE)
