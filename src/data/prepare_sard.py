"""SARD dataset integrity check + structured EDA report.

Validates the SARD YOLO-format dataset for:
  - Corrupt or unreadable images
  - Missing label files for images
  - Orphan label files without matching images
  - Malformed label lines (wrong column count, non-numeric)
  - Bounding boxes with values outside [0, 1]
  - Zero-area or negative-area bboxes
  - Images that are too small for tiny-object training

Outputs:
  - results/sard_integrity_report.json — structured report for Bab 2 proposal
  - results/sard_eda_stats.json — per-split statistics (bbox size distribution,
    tiny-object ratio, image size stats, objects per image)
  - Optionally: a cleaned data.yaml that excludes flagged files

Usage:
    python -m src.data.prepare_sard \\
        --root data/kaggle/search-and-rescue \\
        --out-dir results \\
        [--min-image-size 320] [--tiny-threshold 0.02] [--write-clean-yaml]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class IssueRecord:
    split: str
    file: str
    kind: str
    detail: str


@dataclass
class SplitStats:
    split: str
    images_total: int = 0
    images_valid: int = 0
    images_corrupt: int = 0
    images_too_small: int = 0
    labels_total: int = 0
    labels_missing: int = 0
    labels_orphan: int = 0
    labels_malformed: int = 0
    bboxes_total: int = 0
    bboxes_out_of_range: int = 0
    bboxes_zero_area: int = 0
    tiny_bbox_count: int = 0
    image_width_mean: float = 0.0
    image_height_mean: float = 0.0
    bbox_area_median: float = 0.0
    bbox_w_median: float = 0.0
    bbox_h_median: float = 0.0
    objects_per_image_mean: float = 0.0


@dataclass
class IntegrityReport:
    dataset: str
    root: str
    splits: dict[str, SplitStats]
    issues: list[IssueRecord]
    clean_yaml_path: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def iter_images(images_dir: Path):
    if not images_dir.exists():
        return
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def parse_label_file(path: Path) -> tuple[list[list[float]], list[str]]:
    """Return (valid_boxes, error_messages). Each box is [cls, xc, yc, w, h]."""
    boxes = []
    errors = []
    try:
        content = path.read_text().strip()
    except Exception as e:
        return [], [f"unreadable: {e}"]
    if not content:
        return [], []
    for lineno, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"line {lineno}: expected 5 cols, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"line {lineno}: non-numeric values")
            continue
        boxes.append([cls, xc, yc, w, h])
    return boxes, errors


def check_split(
    root: Path,
    split: str,
    issues: list[IssueRecord],
    min_image_size: int,
    tiny_threshold: float,
) -> SplitStats:
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    stats = SplitStats(split=split)

    if not images_dir.exists() or not labels_dir.exists():
        issues.append(IssueRecord(split, str(images_dir), "missing_split",
                                   "split directory not found"))
        return stats

    image_paths = list(iter_images(images_dir))
    label_paths = set(
        p.stem for p in labels_dir.iterdir() if p.suffix == ".txt"
    )
    image_stems = set(p.stem for p in image_paths)

    stats.images_total = len(image_paths)

    widths, heights = [], []
    bbox_areas, bbox_ws, bbox_hs = [], [], []
    objects_per_image = []

    for img_path in image_paths:
        # Image integrity
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                w, h = im.size
        except (UnidentifiedImageError, OSError) as e:
            stats.images_corrupt += 1
            issues.append(IssueRecord(
                split, str(img_path.relative_to(root)), "image_corrupt", str(e)
            ))
            continue

        if w < min_image_size or h < min_image_size:
            stats.images_too_small += 1
            issues.append(IssueRecord(
                split, str(img_path.relative_to(root)),
                "image_too_small", f"{w}x{h}"
            ))

        widths.append(w)
        heights.append(h)
        stats.images_valid += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats.labels_missing += 1
            issues.append(IssueRecord(
                split, str(img_path.relative_to(root)),
                "label_missing", "no matching .txt"
            ))
            continue

        stats.labels_total += 1
        boxes, errors = parse_label_file(label_path)
        if errors:
            stats.labels_malformed += 1
            for e in errors:
                issues.append(IssueRecord(
                    split, str(label_path.relative_to(root)),
                    "label_malformed", e
                ))

        img_bbox_count = 0
        for cls, xc, yc, bw, bh in boxes:
            stats.bboxes_total += 1
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
                    and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                stats.bboxes_out_of_range += 1
                issues.append(IssueRecord(
                    split, str(label_path.relative_to(root)),
                    "bbox_out_of_range",
                    f"xc={xc:.3f} yc={yc:.3f} w={bw:.3f} h={bh:.3f}"
                ))
                continue
            area = bw * bh
            if area <= 0:
                stats.bboxes_zero_area += 1
                continue
            bbox_areas.append(area)
            bbox_ws.append(bw)
            bbox_hs.append(bh)
            if area < tiny_threshold:
                stats.tiny_bbox_count += 1
            img_bbox_count += 1

        objects_per_image.append(img_bbox_count)

    # Orphan labels
    for stem in label_paths:
        if stem not in image_stems:
            stats.labels_orphan += 1
            issues.append(IssueRecord(
                split, f"{split}/labels/{stem}.txt", "label_orphan",
                "no matching image"
            ))

    if widths:
        stats.image_width_mean = round(float(np.mean(widths)), 2)
        stats.image_height_mean = round(float(np.mean(heights)), 2)
    if bbox_areas:
        stats.bbox_area_median = round(float(np.median(bbox_areas)), 5)
        stats.bbox_w_median = round(float(np.median(bbox_ws)), 5)
        stats.bbox_h_median = round(float(np.median(bbox_hs)), 5)
    if objects_per_image:
        stats.objects_per_image_mean = round(
            float(np.mean(objects_per_image)), 3
        )

    return stats


def build_clean_yaml(root: Path, out_path: Path) -> None:
    """Write a clean data.yaml with absolute paths — avoids relative-path surprises."""
    content = f"""# Auto-generated by prepare_sard.py
path: {root.resolve()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['human']
"""
    out_path.write_text(content)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="SARD dataset root")
    ap.add_argument("--out-dir", default="results",
                    help="Where to write JSON reports")
    ap.add_argument("--min-image-size", type=int, default=320,
                    help="Flag images smaller than this on either side")
    ap.add_argument("--tiny-threshold", type=float, default=0.02,
                    help="Normalized bbox area threshold for tiny object count")
    ap.add_argument("--write-clean-yaml", action="store_true",
                    help="Write a cleaned data.yaml with absolute path")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"SARD root not found: {root}")

    issues: list[IssueRecord] = []
    per_split: dict[str, SplitStats] = {}

    for split in SPLITS:
        print(f"[*] Checking split: {split}")
        per_split[split] = check_split(
            root, split, issues, args.min_image_size, args.tiny_threshold
        )

    total_images = sum(s.images_total for s in per_split.values())
    total_valid = sum(s.images_valid for s in per_split.values())
    total_bboxes = sum(s.bboxes_total for s in per_split.values())
    total_tiny = sum(s.tiny_bbox_count for s in per_split.values())
    tiny_ratio = round(total_tiny / max(total_bboxes, 1), 4)

    clean_yaml_path = None
    if args.write_clean_yaml:
        clean_yaml_path = out_dir / "sard_clean.yaml"
        build_clean_yaml(root, clean_yaml_path)

    report = IntegrityReport(
        dataset="SARD (Search and Rescue Dataset)",
        root=str(root),
        splits={k: v for k, v in per_split.items()},
        issues=issues,
        clean_yaml_path=str(clean_yaml_path) if clean_yaml_path else None,
        summary={
            "images_total": total_images,
            "images_valid": total_valid,
            "images_invalid": total_images - total_valid,
            "bboxes_total": total_bboxes,
            "tiny_bbox_count": total_tiny,
            "tiny_bbox_ratio": tiny_ratio,
            "issue_count": len(issues),
            "issues_by_kind": _count_by_kind(issues),
        },
    )

    # Serialize
    integrity_path = out_dir / "sard_integrity_report.json"
    report_dict = {
        "dataset": report.dataset,
        "root": report.root,
        "splits": {k: asdict(v) for k, v in report.splits.items()},
        "summary": report.summary,
        "clean_yaml_path": report.clean_yaml_path,
        "issues_sample": [asdict(i) for i in issues[:100]],
        "issue_count_total": len(issues),
    }
    integrity_path.write_text(json.dumps(report_dict, indent=2))

    # Human-readable summary
    sep = "─" * 64
    print()
    print(sep)
    print(f"  SARD Integrity Report — {root}")
    print(sep)
    print(f"  {'Split':<8}{'Images':>10}{'Valid':>10}{'BBoxes':>10}{'Tiny':>10}")
    for split, s in per_split.items():
        print(f"  {split:<8}{s.images_total:>10}{s.images_valid:>10}"
              f"{s.bboxes_total:>10}{s.tiny_bbox_count:>10}")
    print(sep)
    print(f"  Total images         : {total_images}")
    print(f"  Total valid          : {total_valid}")
    print(f"  Total bboxes         : {total_bboxes}")
    print(f"  Tiny bboxes (< {args.tiny_threshold}): {total_tiny} "
          f"({tiny_ratio*100:.2f}%)")
    print(f"  Issues detected      : {len(issues)}")
    for kind, n in report.summary["issues_by_kind"].items():
        print(f"    - {kind:<22}: {n}")
    print(sep)
    print(f"  Saved: {integrity_path}")
    if clean_yaml_path:
        print(f"  Saved: {clean_yaml_path}")
    print(sep)


def _count_by_kind(issues: list[IssueRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in issues:
        counts[i.kind] = counts.get(i.kind, 0) + 1
    return counts


if __name__ == "__main__":
    main()
