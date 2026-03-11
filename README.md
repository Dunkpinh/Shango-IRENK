# Project ERC team Shango
# 🍌 IRENK - Identifikasi Rona, Emisi, dan Noda Kebusukan
# Banana Detection (YOLOv8)

Proyek deteksi kematangan pisang menggunakan YOLOv8. Model dapat mendeteksi 3 kelas:
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
- **Model (best.pt):** 

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

## 🏋️ Training

> Jalankan dari folder `Shango-IRENK/`

```bash
cd /path/to/Shango-IRENK/

# Training dari awal
MPLBACKEND=Agg yolo train data=dataset.yaml model=yolov8s.pt epochs=150

# Fine-tune dari model yang sudah ada
MPLBACKEND=Agg yolo train data=dataset.yaml model=runs/detect/train/weights/best.pt epochs=150

# Resume training yang terhenti
MPLBACKEND=Agg yolo train resume model=runs/detect/train/weights/last.pt
```


## 🔍 Prediksi

```bash
# Gambar tunggal
MPLBACKEND=Agg yolo detect predict model=runs/detect/train/weights/best.pt source=/path/ke/gambar.jpeg

# Folder gambar
MPLBACKEND=Agg yolo detect predict model=runs/detect/train/weights/best.pt source=/path/ke/folder/

# Real-time webcam
yolo detect predict model=runs/detect/train/weights/best.pt source=0 show=True conf=0.5
```

> Hasil prediksi tersimpan di `runs/detect/predictN/`

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
- Jalankan semua perintah dari folder `Shango-IRENK/`
- Suhu GPU normal saat training: 75–90°C
- Pantau suhu GPU: `watch -n 1 nvidia-smi`
