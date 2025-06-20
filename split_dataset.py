import os
import shutil
import random

# Base path
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Classes
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

for cls in classes:
    train_cls_path = os.path.join(train_dir, cls)
    val_cls_path = os.path.join(val_dir, cls)

    # Create validation folder if not exist
    os.makedirs(val_cls_path, exist_ok=True)

    # List images in training folder
    all_images = os.listdir(train_cls_path)
    random.shuffle(all_images)

    # 20% for validation
    val_count = int(0.2 * len(all_images))
    val_images = all_images[:val_count]

    for img in val_images:
        src_path = os.path.join(train_cls_path, img)
        dst_path = os.path.join(val_cls_path, img)
        shutil.move(src_path, dst_path)

    print(f"{cls}: Moved {val_count} images to validation folder ✅")

print("🔁 All categories split successfully!")
