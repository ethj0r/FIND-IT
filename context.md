# DroneAId — Session Context Summary

## Project Overview
Hackathon "DroneAId" — multi-UAV swarm SAR system untuk disaster response di Indonesia.
**Track A: Edge Vision** — fokus pada model ringan yang berjalan di edge device (CPU).

## Team
- Semua anggota pakai macOS M-series (M4, 16 GB RAM)
- Workflow: kode lokal di VSCode → eksekusi di Kaggle GPU kernel via Jupyter proxy

## Constraint (Dikonfirmasi Panitia 12 April 2026)
| ID | Constraint | Detail Panitia |
|---|---|---|
| C-A1 | Model ≤ 50 MB | **Total akumulasi** semua file bobot model yang di-load. Kode (OpenCV, pathfinding, rule-based) **tidak dihitung** |
| C-A2 | CPU only | Inference hanya di CPU |
| C-A3 | ≤ 3 detik | **Forward pass saja** (bukan pre/post-processing). Steady state (setelah warmup). Hardware: Intel i5 Gen 8 minimal |
| C-A4 | Framework bebas | - |
| C-A5 | Offline | Tidak boleh akses internet saat inference |

## Arsitektur Pipeline
```
Frame masuk
  ↓
[1] YOLOv12s Detection (SARD, imgsz=1280) → bbox semua manusia
  ↓
[2] Farneback Optical Flow (OpenCV) → per bbox, cek gerakan antar frame
     → "moving" atau "static"
  ↓
[3] Morphological Pose → aspect ratio bbox
     → w/h > 1.0 = "prone" (terbaring)
     → w/h ≤ 1.0 = "standing" (berdiri)
  ↓
[4] Status Classification (rule-based):
     → static + prone    = "critical"
     → static + standing = "needs_help"
     → moving + prone    = "injured"
     → moving + standing = "normal"
  ↓
[5] YOLO11s-seg (RescueNet, imgsz=640) → segmentasi terrain
     → safe: Road-Clear, Building No Damage
     → danger: Water, Road-Blocked, Total Destruction, Major Damage
  ↓
[6] World Map Buffer → GPS + triage status tiap orang
  ↓
[7] A* Pathfinding → rute evakuasi, hindari terrain bahaya
     prioritas ke korban "critical" dulu
```

## Model Training Results

### Model 1: Detection (YOLOv12s on SARD)
- **Dataset**: SARD — 5,755 images, 7,424 bboxes, 1 class ("human")
  - 83.50% bbox tiny (area < 0.02)
  - Split: train 4,041 / valid 1,144 / test 570
- **Training**: Kaggle P100, imgsz=1280, ~10 epochs
- **Test Metrics**:
  - mAP50: **0.687**
  - mAP50-95: **0.322**
  - Precision: **0.765**
  - Recall: **0.611**
- **ONNX size**: 18.093 MB
- **Local files**:
  - `models/yolov12s_sard_best.pt` (18.05 MB)
  - `models/yolov12s_sard_best.onnx` (18.09 MB)
  - `results/yolov12s_sard_baseline_v1_summary.json`

### Model 2: Segmentation (YOLO11s-seg on RescueNet)
- **Dataset**: RescueNet — 4,494 total images (kita pakai subset: 720 train, 449 val, 450 test)
  - 8 classes: water, building-no-damage, building-minor-damage, building-major-damage, building-total-destruction, road-clear, road-blocked, vehicle
  - Konversi dari semantic mask PNG → YOLO polygon format via `src/data/convert_rescuenet_remote.py`
- **Training**: Kaggle P100, imgsz=640, 10 epochs
- **Note**: YOLOv12 tidak punya varian -seg, dipakai YOLO11s-seg
- **ONNX size**: 19.401 MB (FP16 export karena FP32 = 38.7 MB melebihi budget)
- **Local files**:
  - `models/yolo11s_seg_rescuenet_best.pt` (19.55 MB)
  - `models/yolo11s_seg_rescuenet_best.onnx` (19.40 MB)
  - `results/yolov12s_seg_rescuenet_v1_summary.json`

## Constraint Validation Status
| Constraint | Limit | Actual | Status |
|---|---|---|---|
| C-A1 (total size) | ≤ 50 MB | 18.093 + 19.401 = **37.494 MB** | **PASS** |
| C-A3 (latency p95) | ≤ 3.0s | ~1.0s det on M4 ARM (belum x86 final) | **PASS (estimasi)** |

## CPU Benchmark (Lokal, BELUM FINAL — ARM64)
- Detection ONNX on Apple M4:
  - p50: 600 ms, p95: 1,007 ms, peak RAM: 1.7 GB
- **Perlu validasi ulang di Kaggle CPU x86** (Intel Xeon) untuk angka resmi C-A3
- Kaggle CPU session bermasalah ("Draft Session Starting" stuck)

## Tools yang Sudah Dibuat
| File | Fungsi |
|---|---|
| `src/tools/kaggle_exec.py` | Remote execute Python di Kaggle kernel via WebSocket |
| `src/tools/kaggle_sync.py` | Sync files (ls, pull-file, pull-dir, watch-log, ps) |
| `src/data/prepare_sard.py` | SARD integrity check + EDA report |
| `src/data/convert_rescuenet_remote.py` | Convert RescueNet mask → YOLO-seg format (remote) |
| `src/data/convert_rescuenet_to_yolo_seg.py` | Conversion code template |
| `src/benchmarks/bench_cpu.py` | Single-model CPU benchmark |
| `src/benchmarks/bench_pipeline.py` | End-to-end pipeline benchmark |

## Notebooks
| File | Fungsi | Status |
|---|---|---|
| `notebooks/00_kaggle_cpu_benchmark.ipynb` | C-A1 + C-A3 validation di x86 | Belum berhasil (Kaggle CPU session stuck) |
| `notebooks/01_train_yolov12s_sard_kaggle.ipynb` | Detection training | Selesai |
| `notebooks/02_train_yolov12s_seg_rescuenet_kaggle.ipynb` | Segmentation training | Selesai (skip test validation karena OOM) |

## Key Files
```
FINDIT/
├── .env                          # Kaggle proxy URLs (train + benchmark)
├── requirements.txt
├── models/
│   ├── yolov12s_sard_best.pt
│   ├── yolov12s_sard_best.onnx
│   ├── yolo11s_seg_rescuenet_best.pt
│   └── yolo11s_seg_rescuenet_best.onnx
├── results/
│   ├── yolov12s_sard_baseline_v1_summary.json
│   ├── yolov12s_seg_rescuenet_v1_summary.json
│   ├── benchmark_local_det.json
│   ├── sard_integrity_report.json
│   └── 1/                        # User-uploaded weights
├── runs/
│   ├── yolov12s_sard_baseline_v1/  # Det training artifacts
│   └── yolov12s_seg_rescuenet_v1/  # Seg training artifacts
├── notebooks/
├── src/
│   ├── tools/
│   ├── data/
│   └── benchmarks/
├── ABSTRACT.MD
├── CONSTRAINT.MD
└── PROPOSAL.MD
```

## Dataset Locations
- **SARD** (lokal): `data/kaggle/search-and-rescue/` — train/valid/test splits
- **RescueNet**: di-download langsung di Kaggle kernel dari `yaroslavchyrko/rescuenet` (Kaggle dataset)
  - Subset: train parts 1+2 (720 images) + val (449) + test (450) + train-labels (3,595)
  - Converted ke YOLO-seg format di `/kaggle/working/rescuenet_yolo/`

## Pending / Belum Selesai
1. **Benchmark CPU x86 final** — Kaggle CPU session tidak bisa start. Perlu coba lagi nanti
2. **Pipeline script utama** — belum dibuat. Detection → optical flow → pose → status → segmentation → pathfinding
3. **Abstract koreksi**:
   - "7.424 gambar" → "5.755 gambar dengan 7.424 bounding box"
   - "YOLOv12-seg" → "YOLO11s-seg" (YOLOv12 tidak punya varian seg)
4. **Test validation seg model** — skip karena OOM di GPU + terlalu lama di CPU
5. **RescueNet full train** — hanya pakai 720/3,595 train images. Bisa ditambah kalau ada waktu
6. **Git commit** — belum commit progress apapun

## Workflow Notes
- Kaggle session sering disconnect/expire. Proxy URL berubah tiap session baru
- File besar (>10 MB) tidak bisa di-transfer via WebSocket kaggle_sync. Gunakan Jupyter REST API (`{base}/files/...`)
- Kaggle working disk = 20 GB. RescueNet full = 20 GB (tidak cukup). Pakai subset
- Segmentation model validation OOM di GPU (mask interpolation butuh 13 GB untuk 1 image). Perlu device='cpu' tapi sangat lambat (~100 menit belum selesai)
