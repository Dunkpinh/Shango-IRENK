from ultralytics import YOLO
import os

# =====================
# KONFIGURASI VALIDASI
# =====================
MODEL = "runs/detect/train/weights/best.pt"  # Path model hasil training
DATA  = "dataset.yaml"                        # Konfigurasi dataset
IMGSZ = 640                                   # Ukuran gambar input
DEVICE = 0                                    # 0 = GPU, 'cpu' = CPU
SPLIT = "val"                                 # Split dataset: val / test


def validate():
    """Jalankan validasi model."""

    # Validasi model
    if not os.path.exists(MODEL):
        print(f"[ERROR] Model tidak ditemukan: {MODEL}")
        print("[INFO]  Jalankan train.py terlebih dahulu atau download best.pt")
        return

    # Validasi dataset
    if not os.path.exists(DATA):
        print(f"[ERROR] Dataset config tidak ditemukan: {DATA}")
        return

    print(f"\n[INFO] Memuat model  : {MODEL}")
    print(f"[INFO] Dataset       : {DATA}")
    print(f"[INFO] Split         : {SPLIT}")
    print(f"[INFO] Device        : {'GPU' if DEVICE == 0 else 'CPU'}")
    print("-" * 40)

    model = YOLO(MODEL)

    metrics = model.val(
        data=DATA,
        imgsz=IMGSZ,
        device=DEVICE,
        split=SPLIT,
    )

    # Tampilkan hasil metrik
    print("\n" + "=" * 40)
    print("  HASIL VALIDASI")
    print("=" * 40)
    print(f"  mAP50        : {metrics.box.map50:.4f}")
    print(f"  mAP50-95     : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("=" * 40)

    # Interpretasi mAP50
    map50 = metrics.box.map50
    print("\n[INTERPRETASI]")
    if map50 >= 0.9:
        print("  Sangat Bagus! Model siap digunakan.")
    elif map50 >= 0.75:
        print("  Bagus. Model cukup handal.")
    elif map50 >= 0.5:
        print("  Cukup. Pertimbangkan finetune lebih lanjut.")
    else:
        print("  Kurang. Disarankan training ulang dengan lebih banyak data/epoch.")


if __name__ == "__main__":
    os.environ["MPLBACKEND"] = "Agg"  # Hindari error matplotlib tanpa GUI

    # =====================
    # INPUT DARI PENGGUNA
    # =====================
    print("=" * 40)
    print("  IRENK - Banana Detection Validation")
    print("=" * 40)

    # Pilih model
    print("\nPilihan Model:")
    print("  1. best.pt  - Model terbaik hasil training")
    print("  2. last.pt  - Model terakhir hasil training")
    print("  3. Custom   - Path model lain")
    model_input = input("\nPilih model (1/2/3) [default: 1]: ").strip()

    if model_input == "2":
        MODEL = "runs/detect/train/weights/last.pt"
    elif model_input == "3":
        path = input("Masukkan path model (.pt): ").strip()
        if path != "":
            MODEL = path
    # default tetap best.pt

    # Pilih split
    print("\nPilihan Split Dataset:")
    print("  1. val   - Dataset validasi")
    print("  2. test  - Dataset testing")
    split_map = {"1": "val", "2": "test"}
    split_input = input("\nPilih split (1/2) [default: 1]: ").strip()
    SPLIT = split_map.get(split_input, "val")

    # Pilih device
    print("\nPilihan Device:")
    print("  1. GPU (CUDA)")
    print("  2. CPU")
    device_map = {"1": 0, "2": "cpu"}
    device_input = input("\nPilih device (1/2) [default: 1]: ").strip()
    DEVICE = device_map.get(device_input, 0)

    validate()
