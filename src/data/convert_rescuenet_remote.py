"""Send RescueNet mask→YOLO-seg conversion code to Kaggle kernel and execute it.

Usage:
    KAGGLE_PROXY_URL=... python -m src.data.convert_rescuenet_remote
"""
import os
import sys
from src.tools.kaggle_exec import run_code

CONVERT_CODE = r"""
import os, cv2, json
import numpy as np
from pathlib import Path

CLASS_MAP = {
    2: 0,   # Water
    3: 1,   # Building No Damage
    4: 2,   # Building Minor Damage
    5: 3,   # Building Major Damage
    6: 4,   # Building Total Destruction
    7: 5,   # Road-Clear
    8: 6,   # Road-Blocked
    9: 7,   # Vehicle
}
YOLO_NAMES = ["water","building-no-damage","building-minor-damage","building-major-damage",
              "building-total-destruction","road-clear","road-blocked","vehicle"]

MIN_CONTOUR_AREA = 100
EPSILON_FACTOR = 0.002

SRC = Path("/kaggle/working/rescuenet_parts")
DST = Path("/kaggle/working/rescuenet_yolo")

def mask_to_polygons(mask_path, h, w):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    lines = []
    for pid, yid in CLASS_MAP.items():
        binary = (mask == pid).astype(np.uint8)
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            eps = EPSILON_FACTOR * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) < 3:
                continue
            pts = approx.squeeze()
            if pts.ndim != 2:
                continue
            coords = []
            for x, y in pts:
                coords.append(f"{x/w:.6f}")
                coords.append(f"{y/h:.6f}")
            lines.append(f"{yid} " + " ".join(coords))
    return lines

def convert_split(name, img_dir, lab_dir):
    dst_img = DST / name / "images"
    dst_lbl = DST / name / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted(list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png")))
    done, skip = 0, 0
    for ip in imgs:
        stem = ip.stem
        mp = Path(lab_dir) / f"{stem}_lab.png"
        if not mp.exists():
            mp = Path(lab_dir) / f"{stem}.png"
        if not mp.exists():
            skip += 1
            continue
        img = cv2.imread(str(ip))
        if img is None:
            skip += 1
            continue
        h, w = img.shape[:2]
        lines = mask_to_polygons(mp, h, w)
        dst_file = dst_img / ip.name
        if not dst_file.exists():
            os.symlink(str(ip), str(dst_file))
        (dst_lbl / f"{stem}.txt").write_text("\n".join(lines) if lines else "")
        done += 1
    return done, skip

print("=== Converting RescueNet -> YOLO-seg ===")
stats = {}

# Train: images in 1/ and 2/, labels in train-label-img/
train_merged = SRC / "train-merged"
train_merged.mkdir(exist_ok=True)
for part in ["1", "2"]:
    pdir = SRC / part
    if pdir.exists():
        for f in pdir.iterdir():
            dst = train_merged / f.name
            if not dst.exists():
                os.symlink(str(f), str(dst))

d, s = convert_split("train", str(train_merged), str(SRC / "train-label-img"))
stats["train"] = {"converted": d, "skipped": s}
print(f"  train: {d} converted, {s} skipped")

d, s = convert_split("val", str(SRC / "val" / "val-org-img"), str(SRC / "val" / "val-label-img"))
stats["val"] = {"converted": d, "skipped": s}
print(f"  val: {d} converted, {s} skipped")

d, s = convert_split("test", str(SRC / "test" / "test-org-img"), str(SRC / "test" / "test-label-img"))
stats["test"] = {"converted": d, "skipped": s}
print(f"  test: {d} converted, {s} skipped")

yaml_content = f"path: {DST}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: {len(YOLO_NAMES)}\nnames: {YOLO_NAMES}\n"
(DST / "data.yaml").write_text(yaml_content)

print(json.dumps(stats, indent=2))
print("=== Done ===")

import subprocess
result = subprocess.run(["df", "-h", "/kaggle/working"], capture_output=True, text=True)
print(result.stdout)
"""


def main():
    base = os.environ.get("KAGGLE_PROXY_URL")
    if not base:
        sys.exit("KAGGLE_PROXY_URL env var required")

    print("[convert] Sending conversion code to Kaggle kernel...")
    result = run_code(base, CONVERT_CODE, timeout=600, quiet=False)

    if result["error"]:
        err = result["error"]
        print(f"\n[ERROR] {err['ename']}: {err['evalue']}", file=sys.stderr)
        sys.exit(1)

    print("\n[convert] Done.")


if __name__ == "__main__":
    main()
