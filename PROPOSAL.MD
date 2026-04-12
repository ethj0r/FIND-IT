Struktur Proposal Tahap 2: DroneAId
Bab 1: Pendahuluan
Latar Belakang & Rumusan Masalah: Ambil dari draf abstrak kita yang sudah kuat mengenai krisis konektivitas (kasus Starlink) dan pentingnya otonomi lokal di zona bencana.
Tujuan: Menghadirkan sistem multi-UAV swarm yang mampu melakukan triase otomatis secara luring total.
Bab 2: Metodologi AI
Dataset: Jelaskan detail SARD (7.424 gambar) untuk deteksi manusia dan RescueNet (4.494 gambar) untuk analisis terrain.
Arsitektur Model: Jabarkan penggunaan arsitektur YOLOv12. Tekankan pemilihan varian paling ringan (misal: YOLOv12s atau Nano) agar sesuai dengan batasan ukuran file.
Pre-processing & Augmentasi: Masukkan detail penanganan tiny objects dengan SAHI dan augmentasi Albumentations untuk simulasi cuaca buruk.
Pipeline Evaluasi: Jelaskan pembagian train-val-test secara fisik untuk menjamin tidak adanya kebocoran data (data leakage).
Bab 3: Kepatuhan Constraint (BAGIAN KRUSIAL)
Bagian ini wajib menjawab lima poin batasan teknis secara rinci:
C-A1: Ukuran Model: Buktikan total ukuran file .pt atau .onnx kita berada di bawah 50 MB.
C-A2: Kompatibilitas Platform: Jelaskan bahwa meskipun kita demo menggunakan GPU, model telah dioptimasi untuk berjalan di CPU-only environment.
C-A3: Kecepatan Inferensi: Sajikan data hasil pengujian internal yang menunjukkan inference time per satu sampel input di bawah 3 detik pada mesin CPU standar.
C-A4: Framework: Konfirmasi penggunaan salah satu dari PyTorch, TensorFlow, atau ONNX Runtime.
C-A5: Offline Total: Tegaskan bahwa pipeline inferensi bekerja secara localhost tanpa memanggil API eksternal seperti Google Vision atau AWS.
Bab 4: Rancangan Sistem & Bisnis
Integrasi Produk (MVP): Jelaskan bagaimana model AI akan dibungkus ke dalam aplikasi onsite pada perangkat operator untuk mengelola armada drone.
Analisis Dampak: Jelaskan efisiensi penyelamatan pada Golden Hour dan penguatan kedaulatan digital nasional.


