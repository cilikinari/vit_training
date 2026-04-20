import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

source_dir = os.path.join(BASE_DIR, "kaggle", "dataset-paddy")
train_dir = os.path.join(BASE_DIR, "kaggle", "train")
val_dir = os.path.join(BASE_DIR, "kaggle", "val")

print("SOURCE:", source_dir)
print("EXISTS:", os.path.exists(source_dir))

split_ratio = 0.8

# hapus folder lama biar tidak numpuk (opsional tapi disarankan)
shutil.rmtree(train_dir, ignore_errors=True)
shutil.rmtree(val_dir, ignore_errors=True)

# cek folder source
if not os.path.exists(source_dir):
    raise FileNotFoundError(f"Folder tidak ditemukan: {source_dir}")

print("Source dataset:", source_dir)

# looping tiap class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    # ambil hanya file gambar
    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    # buat folder class di train/val
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # copy train
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )

    # copy val
    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(val_dir, class_name, img)
        )

    print(f"{class_name}: train={len(train_images)}, val={len(val_images)}")

print("✅ Split dataset selesai!")