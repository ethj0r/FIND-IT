
Apa Itu Sealed Constraints?
Dalam semangat tema "Adaptive Intelligence", setiap tim peserta
tidak hanya ditantang untuk membangun model AI yang akurat, tetapi
juga model yang mampu beradaptasi dengan spesifikasi teknis
lingkungan tertentu. Inilah inti dari Sealed Constraints, yaitu batasan
teknis rahasia yang baru diungkap saat Tahap 2 dimulai.
Kegagalan memenuhi seluruh constraint yang telah ditentukan akan
berdampak fatal pada penilaian. Constraint bersifat wajib
(mandatory), bukan opsional.
Setiap track memiliki constraint masing-masing yang mencerminkan
tantangan dunia nyata di domain tersebut. Baca dan pahami
constraint track Anda dengan seksama sebelum memulai
pengembangan model.


Constraint Umum
1.Struktur File : Wajib mengumpulkan: proposal.pdf, training.ipynb
(dengan log terlihat), inference.ipynb (script bersih), file model
(.pt/.h5/dll), folder train data, dan folder test data secara terpisah.

2.Reproducibility: Model boleh memanfaatkan GPU localhost
bawaan laptop peserta saat demo. Namun, model wajib tetap
dapat berjalan di lingkungan CPU-only dan memenuhi batas waktu
inferensi pada kondisi CPU (lihat C-A3). Juri memvalidasi
constraint kecepatan pada kondisi CPU, bukan GPU.

3.Bahasa Kode: Python adalah bahasa utama yang
direkomendasikan. Penggunaan bahasa lain untuk komponen
tertentu diperbolehkan dengan dokumentasi yang jelas.

4.Larangan Cloud Inference: Dilarang menggunakan API inferensi
cloud (OpenAI, Google AI, AWS, Azure AI, Hugging Face Inference
API, dll) sebagai komponen inti model. Pelanggaran ini
mengakibatkan diskualifikasi.

5.Proposal Bab 3: Setiap tim wajib mendedikasikan Bab 3 proposal
untuk menjelaskan secara rinci bagaimana model mereka
memenuhi setiap poin constraint track masing-masing. Bab ini
akan menjadi dasar penilaian kepatuhan constraint.

Track A: The Edge Vision
(Computer Vision)
Konteks Tantangan:
Model Computer Vision pada umumnya membutuhkan sumber daya
komputasi yang besar. Namun di dunia nyata, solusi AI sering harus
berjalan di perangkat dengan sumber daya sangat terbatas, seperti
drone, kamera edge, atau sensor IoT. Track ini menguji kemampuan
tim membangun model yang cerdas sekaligus ringan.
Constraint:

1.Ukuran Model: Bobot model final (file .pt / .h5 / .onnx) TIDAK
BOLEH melebihi 50 MB.

2.Kompatibilitas Platform: Model boleh memanfaatkan GPU
localhost bawaan laptop peserta saat demo. Namun, model wajib
tetap dapat berjalan di lingkungan CPU-only dan memenuhi batas
waktu inferensi pada kondisi CPU (lihat C-A3). Juri memvalidasi
constraint kecepatan pada kondisi CPU, bukan GPU.

3.Kecepatan Inferensi: Waktu inferensi per satu sampel input
(single inference) TIDAK BOLEH melebihi 3 detik pada mesin CPU
standar (minimum Intel Core i5 Gen 8 / setara). Pengukuran
dilakukan pada kondisi CPU tanpa akselerasi GPU.

4.Framework: Model wajib menggunakan salah satu framework
berikut: PyTorch, TensorFlow/Keras, atau ONNX Runtime.
Penggunaan framework lain wajib mendapat persetujuan panitia
sebelum pengumpulan.

5.Offline Total: Seluruh proses inferensi berjalan secara offline
(localhost). Dilarang menggunakan API eksternal atau cloud
inference (misal: Google Vision API, AWS Rekognition, dll) dalam
pipeline inferensi.