"""
Dog vs Cat Image Classification - Data Pipeline
================================================
Handles:
1. Loading/organizing from 'archive' directory (archive/PetImages/Cat, archive/PetImages/Dog)
   or from ZIP files.
2. Organizing images into structured class directories: data/cats and data/dogs.
3. Strict image validation using Pillow to discard corrupted/unreadable files (e.g., 666.jpg, 11702.jpg).
4. Reproducible 80/20 train/validation splitting with fixed random seed (42).
5. Preventing data leakage: Mutual exclusivity between train and validation sets.
6. tf.data pipeline with 128x128 resizing, [0, 1] normalization, and train-only data augmentation.
7. Configurable image sampling for fast experiments.
"""

import os
import shutil
import zipfile
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Global configuration constants
IMAGE_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
CLASS_MAP = {0: "Cat", 1: "Dog"}
CLASS_NAMES = ["Cat", "Dog"]


def validate_image_file(image_path: str) -> bool:
    """
    Validates that an image file exists, is non-empty, and can be fully decoded by
    both Pillow and TensorFlow decode_image into an RGB tensor.
    Filters out corrupted headers, truncated files, channel mismatches, or 0-byte files.
    """
    try:
        if os.path.getsize(image_path) == 0:
            return False
        # 1. Verify with Pillow
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.convert("RGB")
        # 2. Verify with TensorFlow decode
        raw = tf.io.read_file(image_path)
        _ = tf.io.decode_image(raw, channels=3, expand_animations=False)
        return True
    except Exception:
        return False


def organize_from_archive(
    archive_dir: str = "archive",
    output_dir: str = "data",
    clean_existing: bool = True
) -> Tuple[int, int, int]:
    """
    Organizes images from an 'archive' directory (such as archive/PetImages) into
    output_dir/cats and output_dir/dogs. Validates every image file and discards corrupt ones.

    Returns:
        Tuple of (valid_cats, valid_dogs, corrupted_count)
    """
    print(f"[*] Organizing dataset from archive directory: '{archive_dir}'...")
    target_cats = os.path.join(output_dir, "cats")
    target_dogs = os.path.join(output_dir, "dogs")

    if clean_existing:
        if os.path.exists(target_cats):
            shutil.rmtree(target_cats)
        if os.path.exists(target_dogs):
            shutil.rmtree(target_dogs)

    os.makedirs(target_cats, exist_ok=True)
    os.makedirs(target_dogs, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    cats_count = 0
    dogs_count = 0
    corrupted_count = 0

    # Scan for cat and dog directories in archive
    for root, _, files in os.walk(archive_dir):
        root_lower = root.lower()
        is_cat_dir = "cat" in root_lower
        is_dog_dir = "dog" in root_lower

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_extensions:
                continue

            src_path = os.path.join(root, fname)
            fname_lower = fname.lower()

            is_cat = is_cat_dir or "cat." in fname_lower
            is_dog = is_dog_dir or "dog." in fname_lower

            if is_cat and not is_dog:
                if validate_image_file(src_path):
                    dest = os.path.join(target_cats, f"cat_{cats_count:06d}{ext}")
                    try:
                        os.link(src_path, dest)
                    except OSError:
                        shutil.copy2(src_path, dest)
                    cats_count += 1
                else:
                    corrupted_count += 1
            elif is_dog and not is_cat:
                if validate_image_file(src_path):
                    dest = os.path.join(target_dogs, f"dog_{dogs_count:06d}{ext}")
                    try:
                        os.link(src_path, dest)
                    except OSError:
                        shutil.copy2(src_path, dest)
                    dogs_count += 1
                else:
                    corrupted_count += 1

    print(f"[+] Archive organization complete:")
    print(f"    - Valid Cats:      {cats_count}")
    print(f"    - Valid Dogs:      {dogs_count}")
    print(f"    - Corrupt Skipped: {corrupted_count}")

    return cats_count, dogs_count, corrupted_count


def extract_and_organize_dataset(
    zip_path: str,
    output_dir: str = "data",
    clean_existing: bool = False
) -> Tuple[int, int]:
    """
    Extracts a ZIP archive and organizes images into output_dir/cats and output_dir/dogs.
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Dataset ZIP archive not found at: {zip_path}")

    target_cats = os.path.join(output_dir, "cats")
    target_dogs = os.path.join(output_dir, "dogs")

    if clean_existing:
        if os.path.exists(target_cats):
            shutil.rmtree(target_cats)
        if os.path.exists(target_dogs):
            shutil.rmtree(target_dogs)

    os.makedirs(target_cats, exist_ok=True)
    os.makedirs(target_dogs, exist_ok=True)

    temp_extract_dir = os.path.join(output_dir, "_temp_extract")
    os.makedirs(temp_extract_dir, exist_ok=True)

    print(f"[*] Extracting archive '{zip_path}' to temporary directory...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    print("[*] Scanning extracted files and organizing into 'cats/' and 'dogs/'...")
    cats_count = 0
    dogs_count = 0
    skipped_count = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for root, _, files in os.walk(temp_extract_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_extensions:
                continue

            full_path = os.path.join(root, fname)
            path_lower = full_path.lower()
            fname_lower = fname.lower()

            is_cat = "cat." in fname_lower or "/cat/" in path_lower or "/cats/" in path_lower
            is_dog = "dog." in fname_lower or "/dog/" in path_lower or "/dogs/" in path_lower

            if is_cat and not is_dog:
                if validate_image_file(full_path):
                    dest = os.path.join(target_cats, f"cat_{cats_count:06d}{ext}")
                    shutil.move(full_path, dest)
                    cats_count += 1
                else:
                    skipped_count += 1
            elif is_dog and not is_cat:
                if validate_image_file(full_path):
                    dest = os.path.join(target_dogs, f"dog_{dogs_count:06d}{ext}")
                    shutil.move(full_path, dest)
                    dogs_count += 1
                else:
                    skipped_count += 1

    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    return cats_count, dogs_count


def get_validated_image_paths(
    data_dir: str = "data",
    sample_limit: Optional[int] = None,
    seed: int = DEFAULT_SEED
) -> Tuple[List[str], List[int]]:
    """
    Retrieves and validates image paths for both cats and dogs.
    Automatically initializes from 'archive' if data/ is empty or missing.
    """
    cats_dir = os.path.join(data_dir, "cats")
    dogs_dir = os.path.join(data_dir, "dogs")

    # If data directories are missing or empty, check archive/
    needs_organize = (
        not os.path.isdir(cats_dir) or
        not os.path.isdir(dogs_dir) or
        len(os.listdir(cats_dir)) == 0 or
        len(os.listdir(dogs_dir)) == 0
    )

    if needs_organize:
        if os.path.isdir("archive"):
            organize_from_archive("archive", output_dir=data_dir)
        elif os.path.isfile(os.path.join(data_dir, "cats_and_dogs.zip")):
            extract_and_organize_dataset(os.path.join(data_dir, "cats_and_dogs.zip"), output_dir=data_dir)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    raw_cat_files = [
        os.path.join(cats_dir, f)
        for f in os.listdir(cats_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    raw_dog_files = [
        os.path.join(dogs_dir, f)
        for f in os.listdir(dogs_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    print(f"[*] Validating image files with Pillow...")
    valid_cats = []
    corrupted_cats = 0
    for p in raw_cat_files:
        if validate_image_file(p):
            valid_cats.append(p)
        else:
            corrupted_cats += 1

    valid_dogs = []
    corrupted_dogs = 0
    for p in raw_dog_files:
        if validate_image_file(p):
            valid_dogs.append(p)
        else:
            corrupted_dogs += 1

    print(f"[+] Validation complete:")
    print(f"    - Valid Cats: {len(valid_cats)} (Corrupted/Skipped: {corrupted_cats})")
    print(f"    - Valid Dogs: {len(valid_dogs)} (Corrupted/Skipped: {corrupted_dogs})")

    # Sort for deterministic behavior before sampling
    valid_cats.sort()
    valid_dogs.sort()

    rng = random.Random(seed)

    if sample_limit is not None and sample_limit > 0:
        if len(valid_cats) > sample_limit:
            valid_cats = rng.sample(valid_cats, sample_limit)
        if len(valid_dogs) > sample_limit:
            valid_dogs = rng.sample(valid_dogs, sample_limit)
        print(f"[*] Applied sample limit: {len(valid_cats)} cats, {len(valid_dogs)} dogs.")

    all_paths = valid_cats + valid_dogs
    all_labels = [0] * len(valid_cats) + [1] * len(valid_dogs)

    return all_paths, all_labels


def create_data_augmentation_pipeline():
    """
    Builds data augmentation sequential model applied ONLY to training data.
    Includes random flips, rotation, zoom, translation, and contrast variation.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def _process_image_path(file_path: tf.Tensor, label: tf.Tensor, img_size: Tuple[int, int]):
    """
    Loads an image file, decodes it, resizes to img_size, and normalizes pixel values to [0.0, 1.0].
    """
    img_bytes = tf.io.read_file(file_path)
    image = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, [None, None, 3])
    image = tf.image.resize(image, img_size, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_and_split_data(
    data_dir: str = "data",
    img_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = 0.2,
    seed: int = DEFAULT_SEED,
    sample_limit: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], int, int]:
    """
    Loads, validates, splits dataset (80% train, 20% val), and returns high-performance tf.data.Datasets.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    paths, labels = get_validated_image_paths(data_dir=data_dir, sample_limit=sample_limit, seed=seed)

    if len(paths) == 0:
        raise ValueError(
            f"No valid images found in '{data_dir}'. Ensure dataset is organized."
        )

    # Stratified 80/20 train/test split to guarantee balanced classes and zero leakage
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=validation_split,
        random_state=seed,
        stratify=labels,
        shuffle=True
    )

    assert set(train_paths).isdisjoint(set(val_paths)), "CRITICAL: Data leakage detected between train and val!"

    num_train = len(train_paths)
    num_val = len(val_paths)

    train_cats = train_labels.count(0)
    train_dogs = train_labels.count(1)
    val_cats = val_labels.count(0)
    val_dogs = val_labels.count(1)

    print(f"\n[+] Dataset Split Summary (Seed={seed}):")
    print(f"    - Total Valid Images: {len(paths)}")
    print(f"    - Training Set:       {num_train} ({train_cats} cats, {train_dogs} dogs)")
    print(f"    - Validation Set:     {num_val} ({val_cats} cats, {val_dogs} dogs)")

    # Build tf.data pipeline for Training
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(buffer_size=min(num_train, 1000), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(
        lambda p, l: _process_image_path(p, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Data augmentation on training dataset ONLY
    augmentation = create_data_augmentation_pipeline()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build tf.data pipeline for Validation (NO augmentation)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(
        lambda p, l: _process_image_path(p, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, CLASS_NAMES, num_train, num_val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Loader & Organizer for Dogs vs Cats")
    parser.add_argument("--archive-dir", type=str, default="archive", help="Path to archive directory")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit images per class for fast testing")
    args = parser.parse_args()

    if os.path.exists(args.archive_dir):
        organize_from_archive(args.archive_dir, args.output_dir)

    train_ds, val_ds, classes, n_train, n_val = load_and_split_data(
        data_dir=args.output_dir,
        sample_limit=args.sample_limit
    )
    print("\n[*] Validating one batch from training pipeline...")
    for batch_imgs, batch_lbls in train_ds.take(1):
        print(f"    - Batch Images Shape: {batch_imgs.shape} (dtype: {batch_imgs.dtype})")
        print(f"    - Batch Labels Shape: {batch_lbls.shape} (dtype: {batch_lbls.dtype})")
        print(f"    - Pixel Min: {np.min(batch_imgs):.3f}, Max: {np.max(batch_imgs):.3f}")
    print("[+] Pipeline verification SUCCESSFUL!")
