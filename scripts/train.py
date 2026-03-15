from ultralytics import YOLO
import os

# =====================
# KONFIGURASI TRAINING
# =====================
MODEL = "yolo11n.pt"                # Ganti: yolo11n/s/m/l/x.pt
DATA = "dataset.yaml"               # Konfigurasi dataset
EPOCHS = 100                        # Jumlah epoch
IMGSZ = 640                         # Ukuran gambar input
BATCH = 16                          # Batch size (-1 untuk auto)
DEVICE = 0                          # 0 = GPU, 'cpu' = CPU
PROJECT = "runs/detect"             # Folder output
NAME = "train"                      # Nama eksperimen
EXIST_OK = True                     # Timpa hasil training sebelumnya

# =====================
# FINE-TUNE / RESUME
# =====================
FINETUNE_MODEL = "runs/detect/train/weights/best.pt"   # Untuk fine-tune
RESUME_MODEL  = "runs/detect/train/weights/last.pt"    # Untuk resume

# =====================
# MODE TRAINING
# =====================
MODE = "scratch"


def train_scratch():
    """Training dari awal menggunakan pretrained YOLO."""
    print(f"[INFO] Training dari awal dengan model: {MODEL}")
    model = YOLO(MODEL)
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=EXIST_OK,
    )
    return results


def train_finetune():
    """Fine-tune dari model best.pt yang sudah ada."""
    if not os.path.exists(FINETUNE_MODEL):
        print(f"[ERROR] Model tidak ditemukan: {FINETUNE_MODEL}")
        return
    print(f"[INFO] Fine-tuning dari model: {FINETUNE_MODEL}")
    model = YOLO(FINETUNE_MODEL)
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=EXIST_OK,
    )
    return results


def train_resume():
    """Melanjutkan training yang terhenti."""
    if not os.path.exists(RESUME_MODEL):
        print(f"[ERROR] Model tidak ditemukan: {RESUME_MODEL}")
        return
    print(f"[INFO] Melanjutkan training dari: {RESUME_MODEL}")
    model = YOLO(RESUME_MODEL)
    results = model.train(resume=True)
    return results


if __name__ == "__main__":
    os.environ["MPLBACKEND"] = "Agg"  # Hindari error matplotlib tanpa GUI

    # =====================
    # INPUT DARI PENGGUNA
    # =====================
    print("=" * 40)
    print("  IRENK - Banana Detection Training")
    print("=" * 40)

    # Pilih mode
    mode_map = {
        "1": "scratch",
        "2": "finetune",
        "3": "resume",
    }
    print("\nMode Training:")
    print("  1. scratch  - Training dari awal")
    print("  2. finetune - Lanjut dari best.pt")
    print("  3. resume   - Lanjut yang terhenti")
    mode_input = input("\nPilih mode (1/2/3) [default: 1]: ").strip()
    MODE = mode_map.get(mode_input, "scratch")

    if MODE in ("scratch", "finetune"):
        # Pilih model
        model_map = {
            "1": "yolo11n.pt",
            "2": "yolo11s.pt",
            "3": "yolo11m.pt",
            "4": "yolo11l.pt",
            "5": "yolo11x.pt",
        }
        print("\nPilihan Model:")
        print("  1. yolo11n.pt  - Nano   (tercepat, kurang akurat)")
        print("  2. yolo11s.pt  - Small")
        print("  3. yolo11m.pt  - Medium")
        print("  4. yolo11l.pt  - Large")
        print("  5. yolo11x.pt  - Extra  (terlambat, paling akurat)")
        model_input = input("\nPilih model (1-5) [default: 1]: ").strip()
        MODEL = model_map.get(model_input, "yolo11n.pt")

        # Pilih epochs
        epochs_input = input("Jumlah epochs [default: 100]: ").strip()
        if epochs_input != "":
            try:
                EPOCHS = int(epochs_input)
            except ValueError:
                print("[WARNING] Input tidak valid, menggunakan default: 100")
                EPOCHS = 100

    # Pilih device
    device_map = {"1": 0, "2": "cpu"}
    print("\nPilihan Device:")
    print("  1. GPU (CUDA)")
    print("  2. CPU")
    device_input = input("\nPilih device (1/2) [default: 1]: ").strip()
    DEVICE = device_map.get(device_input, 0)

    print(f"\n[INFO] Mode   : {MODE}")
    if MODE != "resume":
        print(f"[INFO] Model  : {MODEL}")
        print(f"[INFO] Epochs : {EPOCHS}")
    print(f"[INFO] Device : {'GPU' if DEVICE == 0 else 'CPU'}")
    print("-" * 40)

    if MODE == "scratch":
        train_scratch()
    elif MODE == "finetune":
        train_finetune()
    elif MODE == "resume":
        train_resume()
    else:
        print(f"[ERROR] MODE tidak valid: {MODE}. Pilih: 1 / 2 / 3")
