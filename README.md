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
│   └── convert_to_yolo.py  ← script konversi dataset CSV ke YOLO
├── dataset.yaml       ← konfigurasi dataset
├── .gitignore
└── README.md
```

---

## ⚙️ Instalasi

```bash
pip install ultralytics
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

- **yolo11n.pt** 
- **yolo11s.pt** 
- **yolo11m.pt** 
- **yolo11l.pt** 
- **yolo11x.pt** 

Untuk mengganti model, cukup ubah bagian `model=` pada perintah training:
```bash
# Contoh ganti ke yolo11s
MPLBACKEND=Agg yolo train data=dataset.yaml model=yolo11s.pt epochs=150 exist_ok=True
```

---

## 🏋️ Training

> Jalankan dari folder `Shango-IRENK/`

```bash
cd /path/to/Shango-IRENK/

# Training dari awal
MPLBACKEND=Agg yolo train data=dataset.yaml model=`modelyolo`.pt epochs=`jumlah epoch` exist_ok=True 

# Fine-tune dari model yang sudah ada
MPLBACKEND=Agg yolo train data=dataset.yaml model=runs/detect/train/weights/best.pt epochs=`jumlah epoch` exist_ok=True

# Resume training yang terhenti
MPLBACKEND=Agg yolo train resume model=runs/detect/train/weights/last.pt
```

> Ubah model dan jumlah epoch

---

## 🔍 Predict

```bash
# Gambar tunggal
MPLBACKEND=Agg yolo detect predict model=runs/detect/train/weights/best.pt source=/path/ke/gambar.jpeg exist_ok=True

# Folder gambar
MPLBACKEND=Agg yolo detect predict model=runs/detect/train/weights/best.pt source=/path/ke/folder/ exist_ok=True

# Real-time webcam
yolo detect predict model=runs/detect/train/weights/best.pt source=0 show=True conf=0.5 exist_ok=True
```

> Sesuaikan path ke source gambar

---

## 🔄 Konversi Dataset CSV ke YOLO

Jika dataset dalam format klasifikasi CSV (dari Roboflow multiclass):

```bash
python scripts/convert_to_yolo.py
```

Edit bagian konfigurasi di dalam script sesuai path dataset kamu.

---

## 🗒️ Catatan

- Gunakan `MPLBACKEND=Agg` saat training untuk menghindari error matplotlib di environment tanpa GUI
- Gunakan `exist_ok=True` agar tidak membuat folder baru setiap training/prediksi
- Jalankan semua perintah dari folder `Shango-IRENK/`
- Suhu GPU normal saat training: 75–90°C
- Pantau suhu GPU: `watch -n 1 nvidia-smi`
