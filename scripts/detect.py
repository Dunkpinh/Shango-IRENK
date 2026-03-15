from ultralytics import YOLO
import os

# =====================
# KONFIGURASI DETEKSI
# =====================
MODEL = "runs/detect/train/weights/best.pt"  # Path model hasil training
CONF  = 0.7                                  # Confidence threshold (0.0 - 1.0)
IMGSZ = 640                                  # Ukuran gambar input
DEVICE = 0                                   # 0 = GPU, 'cpu' = CPU
PROJECT = "runs/detect"                      # Folder output hasil prediksi
NAME = "predict"                             # Nama folder hasil prediksi
EXIST_OK = True                              # Timpa hasil prediksi sebelumnya
SAVE = True                                  # Simpan hasil gambar/video
SHOW = False                                 # Tampilkan hasil (butuh GUI)
SOURCE = "datasets/images/test/"             # Default source


# =====================
# KELAS BANANA
# =====================
CLASSES = {
    0: "Fresh Banana",
    1: "Raw Banana",
    2: "Rotten Banana",
}


def detect():
    """Jalankan deteksi pisang."""

    # Validasi model
    if not os.path.exists(MODEL):
        print(f"[ERROR] Model tidak ditemukan: {MODEL}")
        print("[INFO]  Jalankan train.py terlebih dahulu atau download best.pt")
        return

    # Validasi source (kecuali webcam)
    if SOURCE != 0 and not os.path.exists(str(SOURCE)):
        print(f"[ERROR] Source tidak ditemukan: {SOURCE}")
        return

    print(f"\n[INFO] Memuat model  : {MODEL}")
    print(f"[INFO] Source        : {'Webcam' if SOURCE == 0 else SOURCE}")
    print(f"[INFO] Confidence    : {CONF}")
    print(f"[INFO] Device        : {'GPU' if DEVICE == 0 else 'CPU'}")
    print(f"[INFO] Kelas         : {list(CLASSES.values())}")
    print("-" * 40)

    model = YOLO(MODEL)

    results = model.predict(
        source=SOURCE,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=EXIST_OK,
        save=SAVE,
        show=SHOW,
    )

    # Tampilkan ringkasan hasil
    print("\n[HASIL DETEKSI]")
    for i, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  Gambar {i+1}: Tidak ada objek terdeteksi")
            continue

        print(f"  Gambar {i+1}: {len(boxes)} objek terdeteksi")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = CLASSES.get(cls_id, f"Class {cls_id}")
            print(f"    - {label} ({conf:.2f})")

    if SOURCE != 0:
        print(f"\n[INFO] Hasil disimpan di: {PROJECT}/{NAME}/")


if __name__ == "__main__":
    os.environ["MPLBACKEND"] = "Agg"  # Hindari error matplotlib tanpa GUI

    # =====================
    # INPUT DARI PENGGUNA
    # =====================
    print("=" * 40)
    print("  IRENK - Banana Detection")
    print("=" * 40)

    # Pilih sumber input
    source_map = {
        "1": "gambar",
        "2": "folder",
        "3": "video",
        "4": "webcam",
    }
    print("\nSumber Input:")
    print("  1. Gambar tunggal  (path/gambar.jpg)")
    print("  2. Folder gambar   (path/folder/)")
    print("  3. Video           (path/video.mp4)")
    print("  4. Webcam realtime")
    pilihan = input("\nPilih sumber (1/2/3/4) [default: 1]: ").strip()
    mode = source_map.get(pilihan, "gambar")

    if mode == "webcam":
        SOURCE = 0
        SHOW = True
        print("[INFO] Mode: Webcam realtime")
    elif mode == "video":
        path = input("Masukkan path video: ").strip()
        SOURCE = path if path != "" else SOURCE
        print(f"[INFO] Mode: Video -> {SOURCE}")
    elif mode == "folder":
        path = input("Masukkan path folder [default: datasets/images/test/]: ").strip()
        SOURCE = path if path != "" else "datasets/images/test/"
        print(f"[INFO] Mode: Folder -> {SOURCE}")
    else:
        path = input("Masukkan path gambar: ").strip()
        SOURCE = path if path != "" else SOURCE
        print(f"[INFO] Mode: Gambar -> {SOURCE}")

    # Pilih confidence
    conf_input = input("Confidence threshold (0.0-1.0) [default: 0.7]: ").strip()
    if conf_input != "":
        try:
            CONF = float(conf_input)
        except ValueError:
            print("[WARNING] Input tidak valid, menggunakan default: 0.7")
            CONF = 0.7

    # Pilih device
    device_map = {"1": 0, "2": "cpu"}
    print("\nPilihan Device:")
    print("  1. GPU (CUDA)")
    print("  2. CPU")
    device_input = input("\nPilih device (1/2) [default: 1]: ").strip()
    DEVICE = device_map.get(device_input, 0)

    detect()
