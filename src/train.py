"""
Dog vs Cat Image Classification - Training Pipeline
===================================================
Executes reproducible training for the 3-layer CNN model.

Features:
- Configurable epochs, batch size, and learning rate.
- GPU acceleration detection and utilization when available.
- Training-only data augmentation via tf.data pipeline.
- ModelCheckpoint callback saving the best model based on validation loss.
- EarlyStopping callback preventing overfitting and restoring best weights.
- Training history logging to JSON for reproducible academic documentation.
- Graceful error handling for missing dataset or invalid paths.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import tensorflow as tf

from src.data_loader import load_and_split_data, extract_and_organize_dataset, IMAGE_SIZE
from src.model import build_model, print_architecture_details


def train_model(
    data_dir: str = "data",
    model_save_path: str = "models/dog_cat_cnn.keras",
    history_save_path: str = "results/history.json",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.0007,
    patience: int = 5,
    sample_limit: int = 4000,
    seed: int = 42,
    l2_reg: float = 1e-4
) -> Dict[str, Any]:
    """
    Executes the complete model training workflow and persists the best weights and history.
    """
    print("=" * 80)
    print("                 STARTING ACCURACY-OPTIMIZED TRAINING")
    print("=" * 80)
    print(f"[*] Configuration:")
    print(f"    - Epochs:              {epochs}")
    print(f"    - Batch Size:          {batch_size}")
    print(f"    - Learning Rate:       {learning_rate}")
    print(f"    - L2 Regularization:   {l2_reg}")
    print(f"    - Early Stop Patience: {patience}")
    print(f"    - Sample Limit/Class:  {sample_limit}")
    print(f"    - Model Save Path:     {model_save_path}")
    print(f"    - History Save Path:   {history_save_path}")
    print(f"    - Random Seed:         {seed}")

    # Check for GPU devices and configure memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[+] GPU Acceleration ACTIVE on RTX 4050: {len(gpus)} device(s) detected: {[g.name for g in gpus]}")
        except Exception as e:
            print(f"[!] Warning configuring GPU memory growth: {e}")
    else:
        print("[!] No GPU acceleration found. Operating in optimized multi-threaded CPU mode.")

    # Validate dataset existence
    cats_dir = os.path.join(data_dir, "cats")
    dogs_dir = os.path.join(data_dir, "dogs")
    zip_path = os.path.join(data_dir, "cats_and_dogs.zip")

    if not os.path.isdir(cats_dir) or not os.path.isdir(dogs_dir):
        if os.path.isfile(zip_path):
            print(f"[*] Dataset directories not found, but archive '{zip_path}' exists. Extracting...")
            extract_and_organize_dataset(zip_path, output_dir=data_dir)
        else:
            raise FileNotFoundError(
                f"Dataset directories '{cats_dir}' and '{dogs_dir}' not found and '{zip_path}' is missing. "
                f"Please place the dataset in '{data_dir}'."
            )

    # Load dataset
    print("\n[*] Initializing Data Pipeline...")
    train_ds, val_ds, class_names, num_train, num_val = load_and_split_data(
        data_dir=data_dir,
        img_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        seed=seed,
        sample_limit=sample_limit
    )

    # Build CNN Architecture
    print("\n[*] Initializing 3-Layer Convolutional Neural Network (He-Normal + L2)...")
    model = build_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        learning_rate=learning_rate,
        padding="same",
        l2_reg=l2_reg
    )

    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)

    # Configure callbacks
    callbacks = [
        # Checkpoint: Save best model weights based on maximum validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        # Early Stopping: Prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce LR on Plateau: Adaptively lower learning rate for finer convergence
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n" + "=" * 80)
    print("                     TRAINING IN PROGRESS")
    print("=" * 80)
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"[+] Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")
    print("=" * 80)

    # Format and save training history
    history_dict = {
        "epochs_completed": len(history.history["loss"]),
        "training_time_seconds": round(elapsed_time, 2),
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "accuracy": [float(v) for v in history.history["accuracy"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "num_train": num_train,
            "num_val": num_val
        }
    }

    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)
    print(f"[+] Training history saved to: {history_save_path}")
    print(f"[+] Best model checkpoint saved to: {model_save_path}")

    # Best epoch summary
    best_val_loss = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(best_val_loss) + 1
    best_val_acc = history.history["val_accuracy"][best_epoch - 1]

    print("\n" + "-" * 80)
    print(f"Optimal Checkpoint: Epoch {best_epoch}/{len(history.history['loss'])}")
    print(f"  - Validation Loss:     {best_val_loss:.4f}")
    print(f"  - Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("-" * 80 + "\n")

    return history_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3-Layer CNN for Dog vs Cat Classification")
    parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs (default: 14)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0007, help="Learning rate (default: 0.0007)")
    parser.add_argument("--l2-reg", type=float, default=1e-4, help="L2 regularization factor (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (default: 'data')")
    parser.add_argument("--model-save-path", type=str, default="models/dog_cat_cnn.keras", help="Model output path")
    parser.add_argument("--history-save-path", type=str, default="results/history.json", help="History output path")
    parser.add_argument("--sample-limit", type=int, default=4000, help="Optional image limit per class (default: 4000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        history_save_path=args.history_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        l2_reg=args.l2_reg,
        patience=args.patience,
        sample_limit=args.sample_limit,
        seed=args.seed
    )
