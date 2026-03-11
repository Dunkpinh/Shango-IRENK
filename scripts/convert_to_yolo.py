import os
import csv
import zipfile
import shutil
import random

# ========== KONFIGURASI ==========
ZIP_PATH = "/home/dunkpinh/Downloads/raw-fresh-rotten-banana.v3-gen_banana_v3.multiclass.zip"
EXTRACT_DIR = "/tmp/banana_dataset"
OUTPUT_DIR = "/home/dunkpinh/Documents/Programming/ERC/datasets"

# Rasio split train/valid/test
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO  = 0.1

# Mapping kelas
CLASS_NAMES = ["fresh banana", "raw banana", "rotten banana"]
# ==================================

# 1. Ekstrak zip
print("📦 Mengekstrak zip...")
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)

# 2. Baca _classes.csv
print("📄 Membaca _classes.csv...")
data = []
csv_path = os.path.join(EXTRACT_DIR, "train", "_classes.csv")

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        filename = row[0].strip()
        labels = [int(row[1].strip()), int(row[2].strip()), int(row[3].strip())]
        class_id = labels.index(1)  # ambil index kelas yang bernilai 1
        data.append((filename, class_id))

print(f"✅ Total gambar: {len(data)}")

# 3. Shuffle dan split
random.seed(42)
random.shuffle(data)

n = len(data)
n_train = int(n * TRAIN_RATIO)
n_valid = int(n * VALID_RATIO)

train_data = data[:n_train]
valid_data = data[n_train:n_train + n_valid]
test_data  = data[n_train + n_valid:]

print(f"   Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")

# 4. Buat folder output
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# 5. Salin gambar dan buat label YOLO
def process(dataset, split):
    for filename, class_id in dataset:
        src_img = os.path.join(EXTRACT_DIR, "train", filename)
        dst_img = os.path.join(OUTPUT_DIR, "images", split, filename)
        dst_lbl = os.path.join(OUTPUT_DIR, "labels", split, filename.rsplit(".", 1)[0] + ".txt")

        if not os.path.exists(src_img):
            print(f"⚠️  File tidak ditemukan: {filename}")
            continue

        # Salin gambar
        shutil.copy2(src_img, dst_img)

        # Buat label YOLO (full image bbox: x_center y_center width height)
        with open(dst_lbl, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("🔄 Memproses train...")
process(train_data, "train")

print("🔄 Memproses valid...")
process(valid_data, "valid")

print("🔄 Memproses test...")
process(test_data, "test")

print("\n✅ Selesai! Dataset tersimpan di:")
print(f"   {OUTPUT_DIR}")
print(f"\n📊 Ringkasan:")
for split in ["train", "valid", "test"]:
    count = len(os.listdir(os.path.join(OUTPUT_DIR, "images", split)))
    print(f"   {split}: {count} gambar")
