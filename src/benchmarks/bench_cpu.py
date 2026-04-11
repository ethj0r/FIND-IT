"""CPU inference benchmark harness for DroneAId Track A constraint validation.

Measures single-sample inference latency and model size against C-A1 (≤50 MB)
and C-A3 (≤3s per sample on i5 Gen 8 class CPU).

Primary target runtime: Kaggle CPU notebook (Intel Xeon x86_64).
Secondary target: any x86_64 CPU. ARM64 (M-series) results are flagged as
non-authoritative since the judges validate on x86 Intel i5 Gen 8.

Usage:
    python -m src.benchmarks.bench_cpu \\
        --model models/yolov12s_sard.onnx \\
        --imgsz 1280 \\
        --runs 20 \\
        --threads 4 \\
        --out results/yolov12s_sard_onnx.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------- Constraint references ----------
C_A1_MAX_SIZE_MB = 50.0
C_A3_MAX_LATENCY_S = 3.0
REFERENCE_CPU = "Intel Core i5 Gen 8 (4C/6T, ~3.6 GHz base)"


@dataclass
class SystemInfo:
    platform: str
    machine: str
    processor: str
    cpu_brand: str
    physical_cores: int
    logical_cores: int
    ram_gb: float
    python_version: str
    is_x86_64: bool
    is_kaggle: bool
    warning: str | None = None


@dataclass
class BenchmarkResult:
    model_path: str
    model_format: str
    model_size_mb: float
    imgsz: int
    threads: int
    runs: int
    warmup: int
    cold_start_s: float
    latencies_s: list[float]
    latency_min_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_max_s: float
    latency_mean_s: float
    latency_stdev_s: float
    peak_ram_mb: float
    system: dict[str, Any]
    constraint_c_a1_pass: bool
    constraint_c_a3_pass: bool
    notes: list[str] = field(default_factory=list)


def collect_system_info() -> SystemInfo:
    import os
    import sys

    import psutil

    try:
        import cpuinfo
        brand = cpuinfo.get_cpu_info().get("brand_raw", "unknown")
    except Exception:
        brand = platform.processor() or "unknown"

    machine = platform.machine().lower()
    is_x86 = machine in {"x86_64", "amd64"}
    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists()

    warning = None
    if not is_x86:
        warning = (
            f"Non-x86_64 architecture ({machine}) detected. Results are "
            "NOT authoritative for Track A C-A3 validation. Re-run on a "
            "Kaggle CPU notebook or any x86 Intel CPU before reporting."
        )

    return SystemInfo(
        platform=platform.platform(),
        machine=machine,
        processor=platform.processor(),
        cpu_brand=brand,
        physical_cores=psutil.cpu_count(logical=False) or 0,
        logical_cores=psutil.cpu_count(logical=True) or 0,
        ram_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        python_version=sys.version.split()[0],
        is_x86_64=is_x86,
        is_kaggle=is_kaggle,
        warning=warning,
    )


def lock_threads(n: int) -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


def model_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024**2), 3)


def make_dummy_input(imgsz: int, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((1, channels, imgsz, imgsz), dtype=np.float32)


def load_onnx_session(path: Path, threads: int):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(
        str(path), sess_options=sess_options, providers=providers
    )


def bench_onnx(path: Path, imgsz: int, threads: int, warmup: int, runs: int):
    import onnxruntime as ort

    t0 = time.perf_counter()
    sess = load_onnx_session(path, threads)
    cold = time.perf_counter() - t0

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    if len(input_shape) == 4 and isinstance(input_shape[2], int):
        imgsz = int(input_shape[2])

    x = make_dummy_input(imgsz)

    # Warm up — exclude from measurement
    for _ in range(warmup):
        sess.run(None, {input_name: x})

    latencies = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {input_name: x})
        latencies.append(time.perf_counter() - t)

    return cold, latencies, imgsz


def load_pt_model(path: Path, threads: int):
    import torch

    torch.set_num_threads(threads)
    # Ultralytics .pt loading
    from ultralytics import YOLO

    model = YOLO(str(path))
    return model


def bench_pt(path: Path, imgsz: int, threads: int, warmup: int, runs: int):
    import torch

    t0 = time.perf_counter()
    model = load_pt_model(path, threads)
    cold = time.perf_counter() - t0

    # Build dummy CPU tensor, BCHW float32 [0, 1]
    dummy = torch.rand(1, 3, imgsz, imgsz, dtype=torch.float32)

    for _ in range(warmup):
        _ = model.predict(dummy, verbose=False, device="cpu", imgsz=imgsz)

    latencies = []
    for _ in range(runs):
        t = time.perf_counter()
        _ = model.predict(dummy, verbose=False, device="cpu", imgsz=imgsz)
        latencies.append(time.perf_counter() - t)

    return cold, latencies, imgsz


def peak_ram_mb() -> float:
    import resource
    import sys

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return round(usage / divisor, 2)


def summarize(latencies: list[float]) -> dict[str, float]:
    return {
        "min": min(latencies),
        "p50": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20
        else max(latencies),
        "max": max(latencies),
        "mean": statistics.fmean(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    lock_threads(args.threads)

    path = Path(args.model)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    size_mb = model_size_mb(path)
    fmt = path.suffix.lstrip(".").lower()

    if fmt == "onnx":
        cold, latencies, imgsz = bench_onnx(
            path, args.imgsz, args.threads, args.warmup, args.runs
        )
    elif fmt in {"pt", "pth"}:
        cold, latencies, imgsz = bench_pt(
            path, args.imgsz, args.threads, args.warmup, args.runs
        )
    else:
        raise ValueError(f"Unsupported model format: {fmt}")

    stats = summarize(latencies)
    sys_info = collect_system_info()

    notes = []
    if sys_info.warning:
        notes.append(sys_info.warning)
    if sys_info.logical_cores < args.threads:
        notes.append(
            f"Requested threads={args.threads} but only "
            f"{sys_info.logical_cores} logical cores available."
        )

    result = BenchmarkResult(
        model_path=str(path),
        model_format=fmt,
        model_size_mb=size_mb,
        imgsz=imgsz,
        threads=args.threads,
        runs=args.runs,
        warmup=args.warmup,
        cold_start_s=round(cold, 4),
        latencies_s=[round(x, 4) for x in latencies],
        latency_min_s=round(stats["min"], 4),
        latency_p50_s=round(stats["p50"], 4),
        latency_p95_s=round(stats["p95"], 4),
        latency_max_s=round(stats["max"], 4),
        latency_mean_s=round(stats["mean"], 4),
        latency_stdev_s=round(stats["stdev"], 4),
        peak_ram_mb=peak_ram_mb(),
        system=asdict(sys_info),
        constraint_c_a1_pass=size_mb <= C_A1_MAX_SIZE_MB,
        constraint_c_a3_pass=stats["p95"] <= C_A3_MAX_LATENCY_S,
        notes=notes,
    )
    return result


def print_report(r: BenchmarkResult) -> None:
    sep = "─" * 64
    print(sep)
    print(f"  DroneAId CPU Benchmark Report")
    print(sep)
    print(f"  Model            : {r.model_path}")
    print(f"  Format           : {r.model_format}")
    print(f"  Image size       : {r.imgsz}")
    print(f"  Threads locked   : {r.threads}")
    print(f"  Warmup / runs    : {r.warmup} / {r.runs}")
    print(sep)
    print(f"  CPU              : {r.system['cpu_brand']}")
    print(f"  Architecture     : {r.system['machine']}")
    print(f"  Cores (phys/log) : {r.system['physical_cores']}/{r.system['logical_cores']}")
    print(f"  RAM              : {r.system['ram_gb']} GB")
    print(f"  Kaggle runtime   : {r.system['is_kaggle']}")
    print(sep)
    print(f"  Model size       : {r.model_size_mb:>8.3f} MB  (C-A1 limit {C_A1_MAX_SIZE_MB} MB)")
    print(f"  Cold start       : {r.cold_start_s:>8.4f} s")
    print(f"  Latency  min     : {r.latency_min_s:>8.4f} s")
    print(f"  Latency  p50     : {r.latency_p50_s:>8.4f} s")
    print(f"  Latency  p95     : {r.latency_p95_s:>8.4f} s  (C-A3 limit {C_A3_MAX_LATENCY_S} s)")
    print(f"  Latency  max     : {r.latency_max_s:>8.4f} s")
    print(f"  Latency  mean    : {r.latency_mean_s:>8.4f} s  (± {r.latency_stdev_s:.4f})")
    print(f"  Peak RAM         : {r.peak_ram_mb:>8.2f} MB")
    print(sep)
    ok = "PASS"
    fail = "FAIL"
    print(f"  C-A1 (size ≤ 50 MB)  : {ok if r.constraint_c_a1_pass else fail}")
    print(f"  C-A3 (p95 ≤ 3.0 s)   : {ok if r.constraint_c_a3_pass else fail}")
    if r.notes:
        print(sep)
        for n in r.notes:
            print(f"  ! {n}")
    print(sep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to .onnx or .pt model")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--threads", type=int, default=4,
                    help="Intra-op thread count. i5 Gen 8 = 4 cores.")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSON output path")
    args = ap.parse_args()

    result = run_benchmark(args)
    print_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(result), indent=2))
        print(f"  Saved JSON report: {out_path}")


if __name__ == "__main__":
    main()
