"""
Dog vs Cat Image Classification - Evaluation Pipeline
=====================================================
Evaluates the trained 3-Layer CNN model on the validation dataset.

Calculates:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (True Negatives, False Positives, False Negatives, True Positives)
- Generates Publication-Ready Visualizations:
  1. Training vs. Validation Accuracy (results/accuracy.png)
  2. Training vs. Validation Loss (results/loss.png)
  3. Confusion Matrix Heatmap (results/confusion_matrix.png)
  4. Sample Validation Predictions Grid (results/sample_predictions.png)
- Exports quantitative evaluation results to results/metrics.json
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import argparse
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import tensorflow as tf

from src.data_loader import load_and_split_data, IMAGE_SIZE, CLASS_MAP


def plot_training_metrics(history_path: str = "results/history.json", output_dir: str = "results"):
    """
    Plots and saves Training vs. Validation Accuracy and Loss curves.
    """
    if not os.path.exists(history_path):
        print(f"[!] History file '{history_path}' not found, skipping training curves plot.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", {})
    epochs = range(1, len(history["loss"]) + 1)

    # 1. Accuracy Plot
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history["accuracy"], "o-", label="Training Accuracy", color="#2563EB", linewidth=2.2, markersize=5)
    plt.plot(epochs, history["val_accuracy"], "s--", label="Validation Accuracy", color="#10B981", linewidth=2.2, markersize=5)
    best_acc_epoch = np.argmax(history["val_accuracy"]) + 1
    best_val_acc = np.max(history["val_accuracy"])
    plt.axvline(best_acc_epoch, color="#EF4444", linestyle=":", label=f"Best Val Acc ({best_val_acc*100:.1f}%)")

    plt.title("Training vs. Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"[+] Saved accuracy curve: {acc_path}")

    # 2. Loss Plot
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history["loss"], "o-", label="Training Loss", color="#D97706", linewidth=2.2, markersize=5)
    plt.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#DC2626", linewidth=2.2, markersize=5)
    min_loss_epoch = np.argmin(history["val_loss"]) + 1
    min_val_loss = np.min(history["val_loss"])
    plt.axvline(min_loss_epoch, color="#6366F1", linestyle=":", label=f"Lowest Val Loss ({min_val_loss:.4f})")

    plt.title("Training vs. Validation Loss (Binary Cross-Entropy)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"[+] Saved loss curve:     {loss_path}")


def evaluate_model(
    model_path: str = "models/dog_cat_cnn.keras",
    data_dir: str = "data",
    output_dir: str = "results",
    sample_limit: int = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates saved model checkpoint, computes metrics, and produces evaluation artifacts.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train model first.")

    print("=" * 80)
    print("                  EVALUATING MODEL ON VALIDATION SET")
    print("=" * 80)
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[+] GPU Acceleration ACTIVE on RTX 4050: {[g.name for g in gpus]}")
        except Exception:
            pass

    print(f"[*] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"[*] Preparing validation dataset (seed={seed})...")
    _, val_ds, class_names, num_train, num_val = load_and_split_data(
        data_dir=data_dir,
        img_size=IMAGE_SIZE,
        batch_size=32,
        seed=seed,
        sample_limit=sample_limit
    )

    y_true_list = []
    y_prob_list = []
    sample_images = []
    sample_true = []
    sample_pred = []
    sample_conf = []

    print("[*] Running inference on validation batches...")
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0).flatten()
        y_prob_list.extend(preds.tolist())
        y_true_list.extend(batch_y.numpy().tolist())

        # Collect sample images for visualization
        if len(sample_images) < 12:
            needed = 12 - len(sample_images)
            for i in range(min(needed, len(preds))):
                sample_images.append(batch_x[i].numpy())
                sample_true.append(int(batch_y[i].numpy()))
                pred_cls = 1 if preds[i] >= 0.5 else 0
                sample_pred.append(pred_cls)
                conf = preds[i] if pred_cls == 1 else (1.0 - preds[i])
                sample_conf.append(float(conf))

    y_true = np.array(y_true_list, dtype=int)
    y_prob = np.array(y_prob_list, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Per-class metrics
    cat_precision = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    cat_recall = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    cat_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    metrics_dict = {
        "dataset": "Kaggle Dogs vs Cats",
        "validation_samples": len(y_true),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "precision_dog": round(prec, 4),
            "recall_dog": round(rec, 4),
            "f1_score_dog": round(f1, 4),
            "accuracy_percent": f"{acc * 100:.2f}%"
        },
        "per_class_metrics": {
            "Cat (Class 0)": {
                "precision": round(cat_precision, 4),
                "recall": round(cat_recall, 4),
                "f1_score": round(cat_f1, 4),
                "support": int((y_true == 0).sum())
            },
            "Dog (Class 1)": {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": int((y_true == 1).sum())
            }
        },
        "confusion_matrix": {
            "matrix": cm,
            "true_negatives_cats_correct": int(tn),
            "false_positives_cats_misclassified": int(fp),
            "false_negatives_dogs_misclassified": int(fn),
            "true_positives_dogs_correct": int(tp)
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[+] Saved metrics json:   {metrics_path}")

    # Generate Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cat (0)", "Dog (1)"],
        yticklabels=["Cat (0)", "Dog (1)"],
        annot_kws={"size": 14, "weight": "bold"},
        cbar=True
    )
    plt.title(f"Confusion Matrix (Acc: {acc*100:.2f}%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[+] Saved confusion matrix: {cm_path}")

    # Generate Sample Predictions Grid
    if sample_images:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), dpi=250)
        axes = axes.flatten()
        for idx in range(min(12, len(sample_images))):
            ax = axes[idx]
            img = sample_images[idx]
            true_cls = CLASS_MAP[sample_true[idx]]
            pred_cls = CLASS_MAP[sample_pred[idx]]
            conf = sample_conf[idx] * 100

            is_correct = (sample_true[idx] == sample_pred[idx])
            title_color = "#15803D" if is_correct else "#B91C1C"

            ax.imshow(img)
            ax.set_title(
                f"Pred: {pred_cls} ({conf:.1f}%)\nTrue: {true_cls}",
                color=title_color,
                fontsize=10,
                fontweight="bold"
            )
            ax.axis("off")
        plt.suptitle("Sample Validation Predictions (Green = Correct, Red = Incorrect)", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        sample_path = os.path.join(output_dir, "sample_predictions.png")
        plt.savefig(sample_path)
        plt.close()
        print(f"[+] Saved sample grid:     {sample_path}")

    # Plot training metrics (accuracy & loss)
    history_file = os.path.join(output_dir, "history.json")
    plot_training_metrics(history_path=history_file, output_dir=output_dir)

    # Print Report
    print("\n" + "=" * 80)
    print("                      MODEL EVALUATION REPORT")
    print("=" * 80)
    print(f"Validation Samples: {len(y_true)}")
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (Dog):    {prec:.4f}")
    print(f"Recall (Dog):       {rec:.4f}")
    print(f"F1-Score (Dog):     {f1:.4f}")
    print("-" * 80)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Cat (0)", "Dog (1)"], digits=4))
    print("Confusion Matrix:")
    print(f"  [TN={tn:3d}  FP={fp:3d}]  <-- Actual Cats (0)")
    print(f"  [FN={fn:3d}  TP={tp:3d}]  <-- Actual Dogs (1)")
    print("=" * 80 + "\n")

    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3-Layer CNN for Dog vs Cat Classification")
    parser.add_argument("--model-path", type=str, default="models/dog_cat_cnn.keras", help="Path to .keras model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation artifacts")
    parser.add_argument("--sample-limit", type=int, default=None, help="Sample limit")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_limit=args.sample_limit,
        seed=args.seed
    )
