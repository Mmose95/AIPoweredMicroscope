import shutil
from pathlib import Path

def move_images_and_labels(ids, target_img_dir, target_lbl_dir, image_dir, label_dir):
    Path(target_img_dir).mkdir(parents=True, exist_ok=True)
    Path(target_lbl_dir).mkdir(parents=True, exist_ok=True)

    for img_id in ids:
        img_path = Path(image_dir) / f"{img_id}.jpg"
        label_path = Path(label_dir) / f"{img_id}.txt"

        # Print for debugging
        print(f"🖼️ Checking: {img_path} | Exists: {img_path.exists()}")
        print(f"📄 Checking: {label_path} | Exists: {label_path.exists()}")

        try:
            if img_path.exists():
                shutil.copy(img_path, Path(target_img_dir) / img_path.name)
                print(f"✅ Copied image to {Path(target_img_dir) / img_path.name}")
            else:
                print(f"⚠️ Image NOT found: {img_path}")

            if label_path.exists():
                shutil.copy(label_path, Path(target_lbl_dir) / label_path.name)
                print(f"✅ Copied label to {Path(target_lbl_dir) / label_path.name}")
            else:
                print(f"⚠️ Label NOT found: {label_path}")
        except Exception as e:
            print(f"❌ Error copying {img_id}: {e}")



def fullDataSplits(image_dir, label_dir, basepath):

    import os
    from collections import defaultdict

    image_dir_Path = Path(image_dir)

    all_images = sorted([f.stem for f in image_dir_Path.glob("*.jpg")])  # No extensions
    import random
    random.shuffle(all_images)

    ssl_split_ratio = 0.7
    n_ssl = int(len(all_images) * ssl_split_ratio)

    ssl_image_ids = all_images[:n_ssl]
    supervised_image_ids = all_images[n_ssl:]

    from sklearn.model_selection import train_test_split

    # Split supervised set into train, val, and test (70/15/15 of the 30% chunk)
    train_ids, temp_ids = train_test_split(supervised_image_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"🔧 SSL images: {len(ssl_image_ids)}")
    print(f"✅ Supervised Train: {len(train_ids)}")
    print(f"🧪 Supervised Val: {len(val_ids)}")
    print(f"🧪 Supervised Test: {len(test_ids)}")

    base = Path(basepath)
    move_images_and_labels(ssl_image_ids, base / "SSL/images", base / "SSL/labels", image_dir, label_dir)
    move_images_and_labels(train_ids, base / "Supervised/Train/images", base / "Supervised/Train/labels", image_dir,
                           label_dir)
    move_images_and_labels(val_ids, base / "Supervised/Val/images", base / "Supervised/Val/labels", image_dir,
                           label_dir)
    move_images_and_labels(test_ids, base / "Supervised/Test/images", base / "Supervised/Test/labels", image_dir,
                           label_dir)


