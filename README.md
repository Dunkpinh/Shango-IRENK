# Project ERC team Shango
# 🍌 IRENK - Identifikasi Rona, Emisi, dan Noda Kebusukan
# Banana Detection (YOLO11)

Proyek deteksi kematangan pisang menggunakan YOLO11. Model dapat mendeteksi 3 kelas:
- **Fresh Banana** — pisang segar
- **Raw Banana** — pisang mentah
- **Rotten Banana** — pisang busuk

---

## 📁 Struktur Folder

```
Shango-IRENK/
├── configs/           ← konfigurasi tambahan
├── datasets/          ← dataset gambar & label (tidak di-push, download manual)
│   ├── images/
│   │   ├── train/     ← gambar training
│   │   ├── valid/     ← gambar validasi
│   │   └── test/      ← gambar testing
│   └── labels/
│       ├── train/     ← label YOLO (.txt)
│       ├── valid/
│       └── test/
├── models/            ← simpan model .pt di sini
├── runs/              ← hasil training & prediksi (tidak di-push)
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt    ← model terbaik
│               └── last.pt    ← model terakhir
├── scripts/
│   ├── train.py            ← script training
│   ├── detect.py           ← script deteksi
│   ├── val.py              ← script validasi
│   └── convert_to_yolo.py  ← script konversi dataset CSV ke YOLO
├── dataset.yaml       ← konfigurasi dataset
├── requirements.txt   ← daftar package Python
├── .gitignore
└── README.md
```

---

## ⚙️ Instalasi

### 1. Buat & Aktifkan Virtual Environment
```bash
python -m venv yolo-env
source yolo-env/bin/activate        # Linux/Mac
yolo-env\Scripts\activate           # Windows
```

### 2. Install PyTorch dengan CUDA
> Install jika device anda memiliki graphic card
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset & Model

- **Dataset:** 
- **Model (best.pt):** https://drive.google.com/drive/folders/1GieAJMuq-nLIfARA0c5Ej1w3kEOq5XOf?usp=sharing

Setelah download, letakkan:
- Dataset → `Shango-IRENK/datasets/`
- Model → `Shango-IRENK/runs/detect/train/weights/best.pt`

---

## 📄 dataset.yaml

```yaml
path: datasets
train: images/train
val: images/valid
test: images/test

nc: 3
names: ["fresh banana", "raw banana", "rotten banana"]
```

---

## 🔀 Pilihan Model YOLO11

| Model | Keterangan |
|---|---|
| yolo11n.pt | Nano — tercepat, kurang akurat |
| yolo11s.pt | Small |
| yolo11m.pt | Medium |
| yolo11l.pt | Large |
| yolo11x.pt | Extra — terlambat, paling akurat |

---

## 🏋️ Training

> Jalankan dari folder `Shango-IRENK/`

```bash
source yolo-env/bin/activate
python scripts/train.py
```

Akan muncul prompt interaktif:
```
Mode Training:
  1. scratch  - Training dari awal
  2. finetune - Lanjut dari best.pt
  3. resume   - Lanjut yang terhenti

Pilihan Model:
  1. yolo11n.pt  - Nano
  2. yolo11s.pt  - Small
  3. yolo11m.pt  - Medium
  4. yolo11l.pt  - Large
  5. yolo11x.pt  - Extra

Jumlah epochs [default: 100]:

Pilihan Device:
  1. GPU (CUDA)
  2. CPU
```

---

## ✅ Validasi

```bash
python scripts/val.py
```

Akan muncul prompt interaktif:
```
Pilihan Model:
  1. best.pt  - Model terbaik hasil training
  2. last.pt  - Model terakhir hasil training
  3. Custom   - Path model lain

Pilihan Split Dataset:
  1. val   - Dataset validasi
  2. test  - Dataset testing

Pilihan Device:
  1. GPU (CUDA)
  2. CPU
```

Hasil validasi:
```
========================================
  HASIL VALIDASI
========================================
  mAP50        : 0.xxxx
  mAP50-95     : 0.xxxx
  Precision    : 0.xxxx
  Recall       : 0.xxxx
========================================
```

---

## 🔍 Deteksi

```bash
python scripts/detect.py
```

Akan muncul prompt interaktif:
```
Sumber Input:
  1. Gambar tunggal  (path/gambar.jpg)
  2. Folder gambar   (path/folder/)
  3. Video           (path/video.mp4)
  4. Webcam realtime

Confidence threshold (0.0-1.0) [default: 0.5]:

Pilihan Device:
  1. GPU (CUDA)
  2. CPU
```

Untuk menghentikan webcam: tekan **Q** di jendela webcam atau **Ctrl+C** di terminal

---

## 🔄 Konversi Dataset CSV ke YOLO

Jika dataset dalam format klasifikasi CSV (dari Roboflow multiclass):

```bash
python scripts/convert_to_yolo.py
```

Edit bagian konfigurasi di dalam script sesuai path dataset anda

---

## 🗒️ Catatan

- Aktifkan virtual environment sebelum menjalankan script: `source yolo-env/bin/activate`
- Jalankan semua perintah dari folder `Shango-IRENK/`
- Urutan penggunaan: `train.py` → `val.py` → `detect.py`
- Suhu GPU normal saat training: 75–90°C
- Pantau suhu GPU: `watch -n 1 nvidia-smi`
