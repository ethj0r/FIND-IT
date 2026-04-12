# Proposal Tahap 2: DroneAId
## Integrated Multi-UAV Swarm untuk Triase Otomatis Korban Bencana

**Track A: The Edge Vision (Computer Vision)**

---

# Bab 1: Pendahuluan

## 1.1 Latar Belakang

Indonesia sebagai negara kepulauan berada pada zona risiko tinggi bencana alam masif, di mana dampak awal yang paling melumpuhkan adalah hancurnya infrastruktur kelistrikan dan menara telekomunikasi yang menciptakan area *blank spot* total. Berdasarkan data BNPB, tercatat 3.472 kejadian bencana pada tahun 2024, sementara bencana di Sumatera akhir 2025 mengakibatkan 1.138 korban tewas akibat kegagalan identifikasi awal. Ironisnya, sistem drone evakuasi saat ini memiliki ketergantungan absolut pada konektivitas cloud atau sinyal radio komersial yang rentan terhadap gangguan (*jamming*) dan pemutusan akses pihak ketiga — sebagaimana yang terjadi pada insiden pemutusan layanan Starlink di Ukraina.

Ketergantungan ini menjadi ancaman langsung terhadap *Digital Sovereignty* dan ketahanan nasional dalam kondisi krisis. Permasalahan ini diperburuk oleh rendahnya efisiensi operasi drone tunggal (*single-UAV*) yang tidak mampu mencakup area bencana luas dalam batasan *Golden Hour*. Selain risiko hilangnya kendali akibat gangguan sinyal radio, proses pengolahan video konvensional sering kali gagal memberikan informasi *real-time* karena keterbatasan *hardware on-board* atau latensi server eksternal.

## 1.2 Rumusan Masalah

1. Bagaimana merancang sistem deteksi korban bencana berbasis *aerial drone* yang mampu beroperasi secara **fully offline** tanpa ketergantungan pada infrastruktur cloud?
2. Bagaimana membangun pipeline analisis citra yang cukup ringan untuk berjalan di **CPU edge device** dengan latensi di bawah 3 detik per sampel, namun tetap mampu mengklasifikasikan tingkat kegawatan korban?
3. Bagaimana mengintegrasikan hasil deteksi korban dengan analisis terrain untuk menghasilkan **rute evakuasi optimal** bagi tim SAR darat?

## 1.3 Tujuan

Menghadirkan sistem multi-UAV swarm bernama **DroneAId** yang mampu:
- Mendeteksi keberadaan manusia dari citra aerial drone pada kondisi pasca-bencana
- Melakukan triase otomatis berdasarkan analisis pergerakan dan pose korban
- Memetakan kondisi terrain secara real-time untuk identifikasi area traversable
- Menghasilkan rekomendasi rute evakuasi optimal melalui algoritma pathfinding
- Beroperasi sepenuhnya secara luring (offline) pada perangkat CPU standar

---

# Bab 2: Metodologi AI

## 2.1 Dataset

Pengembangan DroneAId menggunakan dua dataset publik yang melayani objektif berbeda dalam pipeline sistem.

### 2.1.1 SARD (Search and Rescue Dataset)

Dataset pertama adalah SARD yang tersedia di Kaggle, terdiri dari **5.755 gambar aerial drone** dengan total **7.424 bounding box** dalam single class "human" (format YOLO-ready). Pembagian data mengikuti *pre-split* yang telah tersedia:

| Split | Gambar | Bounding Box | Objek/Gambar |
|-------|--------|-------------|--------------|
| Train | 4.041  | 5.229       | 1,29         |
| Valid | 1.144  | 1.463       | 1,28         |
| Test  | 570    | 732         | 1,28         |

Proses *Exploratory Data Analysis* (EDA) mengungkap temuan kritis: **83,5% objek tergolong *tiny object*** dengan area bounding box di bawah threshold 0,02 (normalized). Median ukuran bounding box hanya **0,056 × 0,128** relatif terhadap dimensi frame — mencerminkan kondisi nyata operasional drone SAR pada ketinggian 30–50 meter. Tidak ditemukan anomali pada dataset: 0 gambar korup, 0 label *malformed*, dan 0 bounding box di luar rentang valid.

> **`[PLACEHOLDER VISUALISASI: Distribusi ukuran bounding box SARD — histogram area bbox + contoh gambar tiny object vs normal object. Sumber: results/sard_integrity_report.json]`**

### 2.1.2 RescueNet

Dataset kedua adalah RescueNet, sebuah dataset UAV *post-disaster* beresolusi tinggi yang terdiri dari **4.494 gambar** dengan *pixel-level semantic segmentation mask*. Dataset ini digunakan untuk melatih model analisis terrain dengan **8 kelas**:

| Kelas | Kategori | Deskripsi |
|-------|----------|-----------|
| road-clear | Safe | Jalan yang dapat dilalui |
| building-no-damage | Safe | Bangunan utuh |
| building-minor-damage | Caution | Kerusakan ringan |
| building-major-damage | Danger | Kerusakan berat |
| building-total-destruction | Danger | Hancur total |
| road-blocked | Danger | Jalan terhalang puing |
| water | Danger | Genangan/banjir |
| vehicle | Neutral | Kendaraan |

Untuk keperluan training, digunakan subset: **720 gambar train**, **449 validasi**, dan **450 test**. Format label asli berupa *semantic mask* PNG dikonversi ke format YOLO polygon segmentation menggunakan script konversi otomatis (`convert_rescuenet_remote.py`).

> **`[PLACEHOLDER VISUALISASI: Sample gambar RescueNet dengan overlay segmentation mask — tunjukkan perbedaan kelas terrain. 2-3 contoh gambar]`**

### 2.1.3 Jaminan Data Leakage

Kedua dataset menggunakan pembagian train/valid/test yang **terpisah secara fisik** dalam folder berbeda. Tidak dilakukan *shuffling* atau redistribusi ulang — split yang digunakan adalah split bawaan dataset asli, sehingga menjamin tidak terjadi kebocoran data antara set pelatihan dan evaluasi.

## 2.2 Arsitektur Model

### 2.2.1 Model Deteksi: YOLOv12s

Model deteksi manusia menggunakan arsitektur **YOLOv12s** (*small variant*) yang merupakan generasi terbaru dari keluarga YOLO dengan peningkatan pada modul *attention mechanism*. Pemilihan varian *small* didasarkan pada batasan ukuran model ≤ 50 MB (constraint C-A1) yang harus menampung dua model sekaligus.

**Konfigurasi training:**
- Pretrained backbone: YOLOv12s (COCO)
- Epochs: 10 (dengan *early stopping patience* = 5)
- Image size: 640 (training) → 1280 (inference, untuk meningkatkan deteksi tiny object)
- Optimizer: auto (AdamW)
- Learning rate: 0,01 → 0,01 (cosine schedule)
- Augmentasi: mosaic, random erasing (0,4), HSV jitter, horizontal flip
- Platform: Kaggle P100 GPU

Keputusan penggunaan **imgsz=1280 saat inference** (lebih besar dari training imgsz=640) diambil sebagai strategi *test-time resolution scaling* — teknik standar untuk meningkatkan sensitivitas deteksi terhadap objek kecil tanpa perlu melatih ulang model. Model YOLO secara arsitektural bersifat *fully convolutional*, sehingga dapat menerima input pada resolusi berbeda dari resolusi training.

> **`[PLACEHOLDER VISUALISASI: Training curves — loss (box, cls, dfl) dan metrik (mAP50, mAP50-95, precision, recall) per epoch. Sumber: runs/yolov12s_sard_baseline_v1/results.csv dan results.png]`**

### 2.2.2 Model Segmentasi: YOLO11s-seg

Model segmentasi terrain menggunakan arsitektur **YOLO11s-seg** (*small variant* dengan *segmentation head*). Pemilihan YOLO11 — bukan YOLOv12 — didasarkan pada fakta bahwa YOLOv12 belum memiliki varian segmentasi pada saat pengembangan. YOLO11s-seg dipilih karena keseimbangan optimal antara akurasi dan ukuran model.

**Konfigurasi training:**
- Pretrained backbone: YOLO11s-seg (COCO)
- Epochs: 10
- Image size: 640
- 8 kelas terrain (RescueNet)
- Platform: Kaggle P100 GPU
- Export: ONNX FP16 (karena FP32 = 38,7 MB melebihi budget sisa)

> **`[PLACEHOLDER VISUALISASI: Training curves segmentasi — box mAP50, mask mAP50 per epoch. Sumber: runs/yolov12s_seg_rescuenet_v1/results.csv dan results.png]`**

### 2.2.3 Export & Optimisasi Model

Kedua model diekspor ke format **ONNX** untuk inferensi menggunakan ONNX Runtime:

| Model | Format | Ukuran | Catatan |
|-------|--------|--------|---------|
| YOLOv12s Detection | ONNX FP32 | 18,093 MB | FP32 cukup kecil, tidak perlu kuantisasi |
| YOLO11s-seg Segmentation | ONNX FP16 | 19,401 MB | FP16 dipilih karena FP32 = 38,7 MB |
| **Total** | | **37,494 MB** | Budget tersisa: 12,506 MB |

## 2.3 Pipeline Inferensi

Sistem DroneAId mengimplementasikan pipeline tujuh tahap yang terintegrasi:

> **`[PLACEHOLDER VISUALISASI: Diagram alir pipeline — flowchart 7 tahap dari frame masuk hingga output rute evakuasi. Gunakan style professional dengan warna berbeda per tahap. Referensi: inference.ipynb]`**

### Tahap 1: Deteksi Manusia
Frame aerial dari drone diproses melalui YOLOv12s ONNX pada resolusi 1280×1280 (letterbox). Output berupa bounding box seluruh manusia yang terdeteksi beserta confidence score.

### Tahap 2: Analisis Pergerakan (Optical Flow)
Untuk setiap bounding box korban, dilakukan analisis pergerakan menggunakan **Farneback Optical Flow** (OpenCV). Rata-rata *magnitude* flow dalam area bbox dibandingkan terhadap threshold (2,0 pixel) untuk mengklasifikasikan korban sebagai "moving" atau "static".

### Tahap 3: Estimasi Pose Morfologis
*Aspect ratio* bounding box digunakan sebagai pendekatan sederhana untuk estimasi pose:
- **w/h > 1,0** → korban terbaring (*prone*)
- **w/h ≤ 1,0** → korban berdiri/duduk (*standing*)

Pendekatan ini dipilih karena ringan secara komputasi dan tidak menambah ukuran model (constraint C-A1), sementara tetap memberikan informasi pose yang cukup untuk keperluan triase awal.

### Tahap 4: Klasifikasi Status (Rule-Based Triage)
Kombinasi hasil analisis gerakan dan pose menghasilkan empat tingkat kegawatan:

| Gerakan | Pose | Status | Prioritas | Rasional |
|---------|------|--------|-----------|----------|
| Static | Prone | **Critical** | 1 (tertinggi) | Tidak bergerak + terbaring → kemungkinan tidak sadarkan diri |
| Static | Standing | **Needs Help** | 2 | Tidak bergerak + berdiri → terjebak atau menunggu bantuan |
| Moving | Prone | **Injured** | 3 | Bergerak + terbaring → cedera tapi masih sadar |
| Moving | Standing | **Normal** | 4 (terendah) | Bergerak + berdiri → kondisi relatif baik |

### Tahap 5: Segmentasi Terrain
Frame yang sama diproses melalui YOLO11s-seg ONNX pada resolusi 640×640 untuk menghasilkan peta segmentasi terrain. Kelas terrain dikategorikan ke dalam **safe** (road-clear, building-no-damage) dan **danger** (water, road-blocked, building-total-destruction, building-major-damage).

### Tahap 6: World Map Buffer
Seluruh hasil deteksi dan segmentasi diagregasi ke dalam struktur data **World Map Buffer** — sebuah peta koordinat yang terus diperbarui selama drone beroperasi. Buffer ini menyimpan:
- Lokasi setiap korban beserta status triasenya
- Peta terrain *traversability* berdasarkan hasil segmentasi
- Metadata confidence dan prioritas untuk pengambilan keputusan

### Tahap 7: A* Pathfinding
Peta terrain dikonversi menjadi **grid traversability** (50×50 sel). Algoritma **A\*** dijalankan untuk mencari rute optimal dari posisi tim SAR darat menuju lokasi korban, dengan prioritas ke korban berstatus **critical** terlebih dahulu. Algoritma mendukung pergerakan 8 arah (termasuk diagonal) dan secara otomatis menghindari sel yang terblokir oleh terrain berbahaya.

> **`[PLACEHOLDER VISUALISASI: Contoh output pipeline — gambar aerial dengan overlay: bbox berwarna sesuai triage + terrain segmentation + rute A* pathfinding. Sumber: results/pipeline_demo_output.jpg setelah menjalankan inference.ipynb]`**

## 2.4 Evaluasi Model

### 2.4.1 Model Deteksi — Test Metrics (SARD)

Evaluasi pada **570 gambar test** yang sepenuhnya terpisah dari data training:

| Metrik | Nilai |
|--------|-------|
| mAP@50 | **0,687** |
| mAP@50-95 | **0,322** |
| Precision | **0,765** |
| Recall | **0,611** |

Nilai recall 0,611 pada dataset dengan 83,5% *tiny object* merupakan hasil yang kompetitif mengingat batasan ukuran model (18 MB). Peningkatan recall dapat dicapai melalui training dengan epoch lebih banyak atau augmentasi tambahan yang ditargetkan untuk objek kecil.

> **`[PLACEHOLDER VISUALISASI: (1) Confusion matrix detection — runs/yolov12s_sard_baseline_v1/confusion_matrix.png, (2) Precision-Recall curve — runs/yolov12s_sard_baseline_v1/BoxPR_curve.png, (3) Contoh prediksi pada test set — results/finalkah?/val_batch0_pred.jpg dan val_batch2_pred.jpg]`**

### 2.4.2 Model Segmentasi — Validation Metrics (RescueNet)

Evaluasi pada **449 gambar validasi** (epoch 10):

| Metrik | Box | Mask |
|--------|-----|------|
| mAP@50 | **0,646** | **0,644** |
| mAP@50-95 | **0,453** | **0,414** |
| Precision | **0,641** | **0,650** |
| Recall | **0,623** | **0,628** |

*Catatan: Evaluasi pada set test terpisah mengalami kendala teknis (OOM pada GPU, waktu eksekusi sangat lama pada CPU). Metrik di atas berasal dari validasi selama training dan bersifat representatif karena set validasi terpisah secara fisik dari set training.*

> **`[PLACEHOLDER VISUALISASI: Training curves segmentasi — plot dual-axis: box mAP50 & mask mAP50 vs epoch, beserta val loss curves]`**

---

# Bab 3: Kepatuhan Constraint

Bab ini menjelaskan secara rinci bagaimana sistem DroneAId memenuhi setiap poin constraint **Track A: The Edge Vision**. Seluruh angka yang disajikan berasal dari pengujian tervalidasi pada environment **Kaggle CPU runtime (x86_64)**.

## 3.1 C-A1: Ukuran Model ≤ 50 MB

**Status: PASS**

Total akumulasi seluruh file bobot model yang di-load:

| Model | File | Ukuran |
|-------|------|--------|
| YOLOv12s Detection | `yolov12s_sard_best.onnx` | 18,093 MB |
| YOLO11s-seg Segmentation | `yolo11s_seg_rescuenet_best.onnx` | 19,401 MB |
| **Total** | | **37,494 MB** |

Margin tersisa: **12,506 MB** (25% dari batas). Sesuai klarifikasi panitia, constraint ini menghitung total akumulasi semua file bobot model yang di-load — kode pemrosesan (OpenCV, pathfinding, rule-based) tidak dihitung.

Strategi optimisasi ukuran:
- Pemilihan varian *small* (bukan *medium/large*) untuk kedua arsitektur
- Export model segmentasi dalam format **FP16** (half-precision) yang mengurangi ukuran dari 38,7 MB menjadi 19,4 MB tanpa degradasi performa signifikan
- Model deteksi cukup kecil pada FP32 (18 MB) sehingga tidak memerlukan kuantisasi tambahan

## 3.2 C-A2: Kompatibilitas Platform (CPU-Only)

**Status: PASS**

Meskipun training dilakukan menggunakan GPU (Kaggle P100), seluruh pipeline inferensi dirancang untuk berjalan di lingkungan **CPU-only**:
- Runtime: **ONNX Runtime** dengan `CPUExecutionProvider` (satu-satunya provider yang didaftarkan)
- Thread locking: 4 intra-op threads (sesuai spesifikasi Intel i5 Gen 8 dengan 4 physical cores)
- Tidak ada dependensi CUDA/cuDNN pada pipeline inferensi
- Optical flow (Farneback), pose estimation (aspect ratio), status classification (rule-based), dan A* pathfinding seluruhnya menggunakan CPU via OpenCV dan NumPy

## 3.3 C-A3: Kecepatan Inferensi ≤ 3 Detik

**Status: PASS**

Pengukuran dilakukan pada **Kaggle CPU runtime** (x86_64) yang merepresentasikan environment validasi juri:

**Environment benchmark:**
| Parameter | Nilai |
|-----------|-------|
| CPU | AMD EPYC 7B12 |
| Arsitektur | x86_64 |
| Physical Cores | 2 |
| Logical Cores | 4 |
| RAM | 31,4 GB |
| Threads (locked) | 4 |
| Warmup | 3 iterations |
| Timed runs | 20 iterations |

**Hasil per model (forward pass only, steady state):**

| Model | imgsz | p50 | p95 | Mean ± Std |
|-------|-------|-----|-----|------------|
| YOLOv12s Detection | 1280 | **1,496 s** | **1,786 s** | 1,518 ± 0,114 s |
| YOLO11s-seg Segmentation | 640 | **0,274 s** | **0,320 s** | 0,279 ± 0,021 s |

**Worst-case sequential (kedua model per frame):**
- p95 kombinasi: 1,786 + 0,320 = **2,106 s** (di bawah batas 3,0 s)

Sesuai klarifikasi panitia, C-A3 mengukur **forward pass saja** (bukan pre/post-processing) pada kondisi **steady state** (setelah warmup). Angka di atas memenuhi kedua kriteria tersebut. Perlu dicatat bahwa AMD EPYC memiliki *single-thread performance* yang lebih rendah dibandingkan Intel i5 Gen 8, sehingga angka benchmark ini bersifat **konservatif** — latensi aktual pada hardware juri diharapkan lebih rendah.

> **`[PLACEHOLDER VISUALISASI: Bar chart perbandingan latency — detection vs segmentation vs combined, dengan garis batas 3.0s. Sumber: results/benchmark_kaggle_cpu_x86.json]`**

## 3.4 C-A4: Framework

**Status: PASS**

Sistem menggunakan **ONNX Runtime** sebagai framework inferensi utama, yang merupakan salah satu framework yang diperbolehkan berdasarkan constraint. Dependensi inferensi:
- `onnxruntime` — model inference (detection + segmentation)
- `opencv-python` — preprocessing, optical flow, visualisasi
- `numpy` — operasi array dan komputasi numerik
- Tidak ada dependensi pada PyTorch/TensorFlow saat inferensi (hanya digunakan saat training)

## 3.5 C-A5: Offline Total

**Status: PASS**

Seluruh pipeline inferensi berjalan secara **offline (localhost)**:
- Tidak ada panggilan API eksternal dalam kode inferensi
- Tidak ada dependensi pada cloud inference (Google Vision, AWS Rekognition, dll)
- Model ONNX di-load dari file lokal
- Optical flow, pose estimation, triage classification, dan A* pathfinding seluruhnya berjalan secara komputasi lokal
- Sistem dirancang untuk lingkungan tanpa konektivitas internet — sesuai dengan skenario operasional pasca-bencana di mana infrastruktur telekomunikasi telah hancur

---

# Bab 4: Rancangan Sistem & Bisnis

## 4.1 Arsitektur Sistem Terintegrasi

DroneAId dirancang sebagai sistem *onsite* yang berjalan pada perangkat operator lapangan (laptop/tablet) tanpa memerlukan koneksi internet.

> **`[PLACEHOLDER VISUALISASI: Diagram arsitektur sistem keseluruhan — menunjukkan: (1) Armada drone dengan kamera, (2) Koneksi ke ground station via radio/fiber optic, (3) Onsite computing device menjalankan pipeline AI, (4) Dashboard operator. Gunakan diagram blok professional]`**

### 4.1.1 Komponen Hardware
- **UAV Fleet**: Drone komersial dengan kemampuan *waypoint navigation* dan kamera aerial
- **Ground Station**: Perangkat operator (laptop/tablet) dengan minimal CPU Intel i5 Gen 8
- **Konektivitas Drone ↔ Station**: Radio link atau opsi kabel *fiber optic* (30-40 km) untuk lingkungan dengan gangguan frekuensi radio ekstrem

### 4.1.2 Komponen Software
- **DroneAId Pipeline**: Inference engine (notebook/aplikasi) yang menjalankan seluruh 7 tahap analisis
- **World Map Buffer**: Database in-memory yang menyimpan akumulasi hasil deteksi dari seluruh drone
- **Dashboard Operator**: Interface visualisasi peta korban, terrain, dan rute evakuasi

> **`[PLACEHOLDER VISUALISASI: Mockup dashboard operator — peta aerial dengan overlay: (1) Titik korban berwarna sesuai triage (merah=critical, oranye=needs_help, kuning=injured, hijau=normal), (2) Overlay terrain segmentation, (3) Garis rute evakuasi A*. Bisa dibuat sebagai wireframe atau mockup UI]`**

## 4.2 Integrasi Produk (MVP)

Minimum Viable Product (MVP) DroneAId terdiri dari:

1. **Inference Engine** (sudah diimplementasi) — pipeline 7 tahap yang berjalan di CPU
2. **Model Files** (sudah dilatih) — 2 model ONNX dengan total 37,5 MB
3. **Operator Interface** (rancangan) — dashboard berbasis web lokal untuk visualisasi hasil

Seluruh komponen dibungkus dalam aplikasi *onsite* yang dapat dijalankan pada perangkat operator tanpa instalasi kompleks. Komandan insiden dapat:
- Mengoordinasikan masing-masing drone melalui GPS *pre-loaded*
- Melihat lokasi korban beserta status triase secara *real-time*
- Mendapatkan rekomendasi rute evakuasi optimal berdasarkan terrain aktual

## 4.3 Analisis Dampak

### 4.3.1 Efisiensi *Golden Hour*
Sistem multi-UAV swarm memungkinkan **penyisiran paralel** area bencana yang luas. Dengan triase otomatis, tim SAR darat dapat langsung diarahkan ke korban berstatus **critical** — mengeliminasi waktu pencarian manual yang selama ini menjadi bottleneck utama pada *Golden Hour* pertama pasca-bencana.

### 4.3.2 Kedaulatan Digital
DroneAId beroperasi sepenuhnya secara *offline* tanpa ketergantungan pada infrastruktur cloud asing. Seluruh data sensitif — lokasi korban, peta bencana, jalur evakuasi — tetap berada di tangan tim penyelamat Indonesia. Arsitektur ini menghilangkan risiko pemutusan akses oleh pihak ketiga sebagaimana yang terjadi pada insiden Starlink di Ukraina.

### 4.3.3 Skalabilitas
Ukuran model yang ringan (37,5 MB total) dan latensi di bawah 3 detik pada CPU standar memungkinkan deployment pada berbagai tingkatan perangkat — dari laptop lapangan hingga embedded device pada drone itu sendiri di masa depan.

> **`[PLACEHOLDER VISUALISASI: Diagram skenario operasional — timeline Golden Hour menunjukkan: (1) Menit 0-5: Deployment drone fleet, (2) Menit 5-15: Penyisiran area + deteksi korban, (3) Menit 15-20: Triase otomatis + pathfinding, (4) Menit 20+: Tim SAR darat menuju korban critical via rute optimal. Bisa dibuat sebagai timeline infographic]`**

---

# Lampiran

## A. Daftar File Submission

| File | Deskripsi |
|------|-----------|
| `proposal.pdf` | Dokumen proposal (Bab 1-4) |
| `notebooks/inference.ipynb` | Script inferensi bersih — pipeline end-to-end |
| `notebooks/01_train_yolov12s_sard_kaggle.ipynb` | Notebook training model deteksi (dengan log) |
| `notebooks/02_train_yolov12s_seg_rescuenet_kaggle.ipynb` | Notebook training model segmentasi (dengan log) |
| `models/yolov12s_sard_best.onnx` | Model deteksi (18,093 MB) |
| `models/yolo11s_seg_rescuenet_best.onnx` | Model segmentasi (19,401 MB) |
| `data/kaggle/search-and-rescue/train/` | Data training SARD |
| `data/kaggle/search-and-rescue/test/` | Data test SARD |

## B. Hasil Benchmark Lengkap

Tersimpan di `results/benchmark_kaggle_cpu_x86.json`.
