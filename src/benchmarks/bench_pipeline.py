"""End-to-end pipeline benchmark for DroneAId dual-model architecture.

Simulates the single-sample (1 frame) pipeline latency:
  Stage 1: YOLOv12-det (SARD, human detection) @ imgsz_det
  Stage 2: YOLOv12-seg (RescueNet, terrain segmentation) @ imgsz_seg
  Stage 3: Post-processing (Farneback optical flow on bbox crops, bbox-ratio
           pose estimation, rule-based status classifier)
  Stage 4: World Map Buffer update (stub)

Stages can be enabled/disabled via CLI to explore budget allocations.
Primary validator for Track A C-A3 (≤3.0 s per sample on CPU).

Usage (after models exist):
    python -m src.benchmarks.bench_pipeline \\
        --det models/yolov12s_sard.onnx --det-imgsz 1280 \\
        --seg models/yolov12s_rescuenet.onnx --seg-imgsz 640 \\
        --enable-flow --enable-pose \\
        --threads 4 --runs 20 \\
        --out results/pipeline_full.json

Can also run with stages disabled for dry-run profiling on a dev machine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmarks.bench_cpu import (
    C_A1_MAX_SIZE_MB,
    C_A3_MAX_LATENCY_S,
    collect_system_info,
    load_onnx_session,
    lock_threads,
    make_dummy_input,
    model_size_mb,
    peak_ram_mb,
)


@dataclass
class StageTimings:
    detection_s: float = 0.0
    segmentation_s: float = 0.0
    optical_flow_s: float = 0.0
    pose_s: float = 0.0
    status_s: float = 0.0
    world_map_s: float = 0.0

    def total(self) -> float:
        return (
            self.detection_s
            + self.segmentation_s
            + self.optical_flow_s
            + self.pose_s
            + self.status_s
            + self.world_map_s
        )


@dataclass
class PipelineResult:
    det_model: str | None
    seg_model: str | None
    det_imgsz: int
    seg_imgsz: int
    det_size_mb: float
    seg_size_mb: float
    total_size_mb: float
    threads: int
    runs: int
    per_run_totals_s: list[float]
    stage_averages: dict[str, float]
    total_min_s: float
    total_p50_s: float
    total_p95_s: float
    total_max_s: float
    total_mean_s: float
    peak_ram_mb: float
    system: dict[str, Any]
    constraint_c_a1_pass: bool
    constraint_c_a3_pass: bool
    assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------- Stage implementations ----------

def stage_detection(sess, input_name: str, imgsz: int) -> tuple[float, np.ndarray]:
    """Run detection model, return latency and fake bboxes for downstream."""
    x = make_dummy_input(imgsz)
    t = time.perf_counter()
    sess.run(None, {input_name: x})
    dt = time.perf_counter() - t
    # Stub: pretend we detected 3 humans at fixed locations
    fake_bboxes = np.array(
        [[100, 100, 180, 260], [300, 150, 380, 320], [500, 400, 580, 560]],
        dtype=np.float32,
    )
    return dt, fake_bboxes


def stage_segmentation(sess, input_name: str, imgsz: int) -> float:
    x = make_dummy_input(imgsz)
    t = time.perf_counter()
    sess.run(None, {input_name: x})
    return time.perf_counter() - t


def stage_optical_flow(bboxes: np.ndarray, crop_size: int = 96) -> float:
    """Farneback flow on each bbox crop. CPU-only OpenCV."""
    import cv2

    rng = np.random.default_rng(seed=0)
    total = 0.0
    for _ in bboxes:
        prev = rng.integers(0, 255, (crop_size, crop_size), dtype=np.uint8)
        curr = rng.integers(0, 255, (crop_size, crop_size), dtype=np.uint8)
        t = time.perf_counter()
        cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        total += time.perf_counter() - t
    return total


def stage_pose(bboxes: np.ndarray) -> float:
    """Bbox-ratio morphological pose classifier (pure math, O(n))."""
    t = time.perf_counter()
    for bbox in bboxes:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        ratio = w / max(h, 1e-6)
        # Simple rule: ratio > 1.0 = prone/horizontal, < 0.5 = standing, else = crouched
        _ = "prone" if ratio > 1.0 else "standing" if ratio < 0.5 else "crouched"
    return time.perf_counter() - t


def stage_status(bboxes: np.ndarray) -> float:
    """Rule-based status classifier using pose + flow magnitude (stub)."""
    t = time.perf_counter()
    for _ in bboxes:
        # Mobile / Distressed / Prone / Static decision tree
        _ = "static"
    return time.perf_counter() - t


def stage_world_map(bboxes: np.ndarray) -> float:
    """Update world map buffer with GPS-tagged victims and obstacles."""
    t = time.perf_counter()
    buffer = {}
    for i, bbox in enumerate(bboxes):
        buffer[f"victim_{i}"] = {
            "bbox": bbox.tolist(),
            "gps": (0.0, 0.0),  # stub
            "triage": "unknown",
        }
    return time.perf_counter() - t


# ---------- Benchmark driver ----------

def run_pipeline_once(
    det_sess, det_input: str, det_imgsz: int,
    seg_sess, seg_input: str, seg_imgsz: int,
    enable_flow: bool, enable_pose: bool,
    enable_status: bool, enable_world_map: bool,
) -> StageTimings:
    timings = StageTimings()

    if det_sess is not None:
        dt, bboxes = stage_detection(det_sess, det_input, det_imgsz)
        timings.detection_s = dt
    else:
        bboxes = np.array(
            [[100, 100, 180, 260], [300, 150, 380, 320], [500, 400, 580, 560]],
            dtype=np.float32,
        )

    if seg_sess is not None:
        timings.segmentation_s = stage_segmentation(seg_sess, seg_input, seg_imgsz)

    if enable_flow:
        timings.optical_flow_s = stage_optical_flow(bboxes)

    if enable_pose:
        timings.pose_s = stage_pose(bboxes)

    if enable_status:
        timings.status_s = stage_status(bboxes)

    if enable_world_map:
        timings.world_map_s = stage_world_map(bboxes)

    return timings


def run(args: argparse.Namespace) -> PipelineResult:
    lock_threads(args.threads)

    det_sess = seg_sess = None
    det_input = seg_input = ""
    det_size = seg_size = 0.0
    det_path = seg_path = None

    if args.det:
        det_path = Path(args.det)
        if not det_path.exists():
            raise FileNotFoundError(f"Detection model not found: {det_path}")
        det_sess = load_onnx_session(det_path, args.threads)
        det_input = det_sess.get_inputs()[0].name
        det_size = model_size_mb(det_path)

    if args.seg:
        seg_path = Path(args.seg)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation model not found: {seg_path}")
        seg_sess = load_onnx_session(seg_path, args.threads)
        seg_input = seg_sess.get_inputs()[0].name
        seg_size = model_size_mb(seg_path)

    total_size = round(det_size + seg_size, 3)

    # Warm up
    for _ in range(args.warmup):
        run_pipeline_once(
            det_sess, det_input, args.det_imgsz,
            seg_sess, seg_input, args.seg_imgsz,
            args.enable_flow, args.enable_pose,
            args.enable_status, args.enable_world_map,
        )

    all_runs: list[StageTimings] = []
    for _ in range(args.runs):
        all_runs.append(
            run_pipeline_once(
                det_sess, det_input, args.det_imgsz,
                seg_sess, seg_input, args.seg_imgsz,
                args.enable_flow, args.enable_pose,
                args.enable_status, args.enable_world_map,
            )
        )

    totals = [r.total() for r in all_runs]
    stage_avgs = {
        "detection_s": round(statistics.fmean(r.detection_s for r in all_runs), 4),
        "segmentation_s": round(statistics.fmean(r.segmentation_s for r in all_runs), 4),
        "optical_flow_s": round(statistics.fmean(r.optical_flow_s for r in all_runs), 4),
        "pose_s": round(statistics.fmean(r.pose_s for r in all_runs), 4),
        "status_s": round(statistics.fmean(r.status_s for r in all_runs), 4),
        "world_map_s": round(statistics.fmean(r.world_map_s for r in all_runs), 4),
    }

    p95 = statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else max(totals)
    sys_info = collect_system_info()

    assumptions = [
        "1 sample input = 1 frame (det + seg run once per frame)",
        "Optical flow runs per detected bbox (stubbed at 3 victims)",
        "Detection and segmentation run sequentially, not parallelized",
        "Post-processing stages use stubs; real values may differ",
    ]
    notes = []
    if sys_info.warning:
        notes.append(sys_info.warning)

    return PipelineResult(
        det_model=str(det_path) if det_path else None,
        seg_model=str(seg_path) if seg_path else None,
        det_imgsz=args.det_imgsz,
        seg_imgsz=args.seg_imgsz,
        det_size_mb=det_size,
        seg_size_mb=seg_size,
        total_size_mb=total_size,
        threads=args.threads,
        runs=args.runs,
        per_run_totals_s=[round(t, 4) for t in totals],
        stage_averages=stage_avgs,
        total_min_s=round(min(totals), 4),
        total_p50_s=round(statistics.median(totals), 4),
        total_p95_s=round(p95, 4),
        total_max_s=round(max(totals), 4),
        total_mean_s=round(statistics.fmean(totals), 4),
        peak_ram_mb=peak_ram_mb(),
        system=asdict(sys_info),
        constraint_c_a1_pass=total_size <= C_A1_MAX_SIZE_MB,
        constraint_c_a3_pass=p95 <= C_A3_MAX_LATENCY_S,
        assumptions=assumptions,
        notes=notes,
    )


def print_report(r: PipelineResult) -> None:
    sep = "─" * 72
    print(sep)
    print("  DroneAId End-to-End Pipeline Benchmark")
    print(sep)
    print(f"  Detection model  : {r.det_model or '(disabled)'}")
    print(f"  Segmentation mdl : {r.seg_model or '(disabled)'}")
    print(f"  Det imgsz        : {r.det_imgsz}")
    print(f"  Seg imgsz        : {r.seg_imgsz}")
    print(f"  Threads / runs   : {r.threads} / {r.runs}")
    print(sep)
    print(f"  CPU              : {r.system['cpu_brand']}")
    print(f"  Architecture     : {r.system['machine']}")
    print(sep)
    print("  Model sizes:")
    print(f"    Detection      : {r.det_size_mb:>8.3f} MB")
    print(f"    Segmentation   : {r.seg_size_mb:>8.3f} MB")
    print(f"    TOTAL          : {r.total_size_mb:>8.3f} MB  (C-A1 limit {C_A1_MAX_SIZE_MB} MB)")
    print(sep)
    print("  Stage latency (mean):")
    for k, v in r.stage_averages.items():
        print(f"    {k:<18s} : {v:>8.4f} s")
    print(sep)
    print("  End-to-end totals (1 sample = 1 frame):")
    print(f"    min            : {r.total_min_s:>8.4f} s")
    print(f"    p50            : {r.total_p50_s:>8.4f} s")
    print(f"    p95            : {r.total_p95_s:>8.4f} s  (C-A3 limit {C_A3_MAX_LATENCY_S} s)")
    print(f"    max            : {r.total_max_s:>8.4f} s")
    print(f"    mean           : {r.total_mean_s:>8.4f} s")
    print(f"    peak RAM       : {r.peak_ram_mb:>8.2f} MB")
    print(sep)
    print(f"  C-A1 (total size ≤ 50 MB)  : {'PASS' if r.constraint_c_a1_pass else 'FAIL'}")
    print(f"  C-A3 (p95 ≤ 3.0 s)         : {'PASS' if r.constraint_c_a3_pass else 'FAIL'}")
    print(sep)
    print("  Assumptions:")
    for a in r.assumptions:
        print(f"    • {a}")
    if r.notes:
        print(sep)
        for n in r.notes:
            print(f"  ! {n}")
    print(sep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det", type=str, default=None, help="Detection .onnx path")
    ap.add_argument("--seg", type=str, default=None, help="Segmentation .onnx path")
    ap.add_argument("--det-imgsz", type=int, default=1280)
    ap.add_argument("--seg-imgsz", type=int, default=640)
    ap.add_argument("--enable-flow", action="store_true",
                    help="Include Farneback optical flow stage")
    ap.add_argument("--enable-pose", action="store_true",
                    help="Include bbox-ratio pose stage")
    ap.add_argument("--enable-status", action="store_true",
                    help="Include status classification stage")
    ap.add_argument("--enable-world-map", action="store_true",
                    help="Include world map buffer update")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    result = run(args)
    print_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(result), indent=2))
        print(f"  Saved JSON report: {out_path}")


if __name__ == "__main__":
    main()
