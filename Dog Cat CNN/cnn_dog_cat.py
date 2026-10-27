import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import shutil
import zipfile
import random
import time
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


RANDOM_SEED      = 42
IMAGE_SIZE       = (128, 128)     # (height, width)
BATCH_SIZE       = 32
EPOCHS           = 15
LEARNING_RATE    = 0.0007
L2_REG           = 1e-4
LABEL_SMOOTH     = 0.05
PATIENCE         = 5
SAMPLE_LIMIT     = 4000           # max images per class; None = use all
VALIDATION_SPLIT = 0.2

DATA_DIR         = "data"
MODEL_FILE       = "dog_cat_cnn.keras"   # saved at project root
OUTPUT_DIR       = "Output"              # all plots go here

CLASS_MAP   = {0: "Cat", 1: "Dog"}
CLASS_NAMES = ["Cat", "Dog"]

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def _valid_image(path: str) -> bool:
    """Return True if the image file is readable and not corrupted."""
    try:
        if os.path.getsize(path) == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        raw = tf.io.read_file(path)
        tf.io.decode_image(raw, channels=3, expand_animations=False)
        return True
    except Exception:
        return False


def _organize_archive(archive_dir: str = "archive", out: str = "data") -> Tuple[int, int]:
    """Organise a Kaggle-style archive/ folder into data/cats and data/dogs."""
    cats_dir = os.path.join(out, "cats")
    dogs_dir = os.path.join(out, "dogs")
    os.makedirs(cats_dir, exist_ok=True)
    os.makedirs(dogs_dir, exist_ok=True)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    n_cats = n_dogs = 0
    for root, _, files in os.walk(archive_dir):
        rl = root.lower()
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_ext:
                continue
            src = os.path.join(root, fname)
            fl  = fname.lower()
            is_cat = ("cat" in rl) or ("cat." in fl)
            is_dog = ("dog" in rl) or ("dog." in fl)
            if is_cat and not is_dog and _valid_image(src):
                dest = os.path.join(cats_dir, f"cat_{n_cats:06d}{ext}")
                try:
                    os.link(src, dest)
                except OSError:
                    shutil.copy2(src, dest)
                n_cats += 1
            elif is_dog and not is_cat and _valid_image(src):
                dest = os.path.join(dogs_dir, f"dog_{n_dogs:06d}{ext}")
                try:
                    os.link(src, dest)
                except OSError:
                    shutil.copy2(src, dest)
                n_dogs += 1
    print(f"Organised: {n_cats} cats, {n_dogs} dogs")
    return n_cats, n_dogs


def _extract_zip(zip_path: str, out: str = "data") -> Tuple[int, int]:
    """Extract a zip file and sort images into data/cats and data/dogs."""
    cats_dir = os.path.join(out, "cats")
    dogs_dir = os.path.join(out, "dogs")
    os.makedirs(cats_dir, exist_ok=True)
    os.makedirs(dogs_dir, exist_ok=True)
    tmp = os.path.join(out, "_tmp_extract")
    os.makedirs(tmp, exist_ok=True)
    print(f"Extracting '{zip_path}'...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    n_cats = n_dogs = 0
    for root, _, files in os.walk(tmp):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_ext:
                continue
            full = os.path.join(root, fname)
            pl   = full.lower()
            fl   = fname.lower()
            is_cat = "cat." in fl or "/cat/" in pl or "/cats/" in pl
            is_dog = "dog." in fl or "/dog/" in pl or "/dogs/" in pl
            if is_cat and not is_dog and _valid_image(full):
                shutil.move(full, os.path.join(cats_dir, f"cat_{n_cats:06d}{ext}"))
                n_cats += 1
            elif is_dog and not is_cat and _valid_image(full):
                shutil.move(full, os.path.join(dogs_dir, f"dog_{n_dogs:06d}{ext}"))
                n_dogs += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Extracted: {n_cats} cats, {n_dogs} dogs")
    return n_cats, n_dogs


def _collect_paths(
    data_dir: str = DATA_DIR,
    sample_limit: Optional[int] = SAMPLE_LIMIT,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[int]]:
    cats_dir = os.path.join(data_dir, "cats")
    dogs_dir = os.path.join(data_dir, "dogs")

    if (
        not os.path.isdir(cats_dir) or not os.path.isdir(dogs_dir)
        or len(os.listdir(cats_dir)) == 0 or len(os.listdir(dogs_dir)) == 0
    ):
        if os.path.isdir("archive"):
            _organize_archive("archive", data_dir)
        elif os.path.isfile(os.path.join(data_dir, "cats_and_dogs.zip")):
            _extract_zip(os.path.join(data_dir, "cats_and_dogs.zip"), data_dir)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    def scan(folder: str) -> List[str]:
        return [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in valid_ext
            and _valid_image(os.path.join(folder, f))
        ]

    cats = scan(cats_dir)
    dogs = scan(dogs_dir)
    rng  = random.Random(seed)
    if sample_limit and sample_limit > 0:
        if len(cats) > sample_limit:
            cats = rng.sample(cats, sample_limit)
        if len(dogs) > sample_limit:
            dogs = rng.sample(dogs, sample_limit)

    return cats + dogs, [0] * len(cats) + [1] * len(dogs)


def _process_path(path: tf.Tensor, label: tf.Tensor, img_size: Tuple[int, int]):
    raw   = tf.io.read_file(path)
    img   = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img   = tf.ensure_shape(img, [None, None, 3])
    img   = tf.image.resize(img, img_size, method="bilinear")
    img   = tf.cast(img, tf.float32) / 255.0
    return img, label


def build_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Return (train_ds, val_ds, num_train, num_val)."""
    paths, labels = _collect_paths()
    if len(paths) == 0:
        raise ValueError(f"No valid images found in '{DATA_DIR}/cats' and '{DATA_DIR}/dogs'.")

    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        paths, labels,
        test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED,
        stratify=labels, shuffle=True,
    )
    assert set(tr_paths).isdisjoint(set(va_paths)), "Data leakage!"

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    n_train, n_val = len(tr_paths), len(va_paths)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
        .shuffle(min(n_train, 1000), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .map(lambda p, l: _process_path(p, l, IMAGE_SIZE), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((va_paths, va_labels))
        .map(lambda p, l: _process_path(p, l, IMAGE_SIZE), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, n_train, n_val


def build_model() -> models.Sequential:
    """Build and compile a 3-block CNN for cat vs dog binary classification."""
    reg = tf.keras.regularizers.l2(L2_REG) if L2_REG > 0 else None

    m = models.Sequential(name="dog_cat_cnn")
    m.add(layers.Input(shape=(*IMAGE_SIZE, 3)))

    m.add(layers.Conv2D(32,  (3, 3), activation="relu", padding="same",
                        kernel_initializer="he_normal", kernel_regularizer=reg))
    m.add(layers.MaxPooling2D((2, 2)))

    m.add(layers.Conv2D(64,  (3, 3), activation="relu", padding="same",
                        kernel_initializer="he_normal", kernel_regularizer=reg))
    m.add(layers.MaxPooling2D((2, 2)))

    m.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                        kernel_initializer="he_normal", kernel_regularizer=reg))
    m.add(layers.MaxPooling2D((2, 2)))

    m.add(layers.Flatten())
    m.add(layers.Dense(128, activation="relu",
                       kernel_initializer="he_normal", kernel_regularizer=reg))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(1, activation="sigmoid"))

    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=["accuracy"],
    )
    return m


def train():
    print("=" * 60)
    print("   Dog vs Cat CNN -- Training")
    print("=" * 60)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"Device: GPU ({gpus[0].name})")
        except Exception as e:
            print(f"GPU warning: {e}")
    else:
        print("Device: CPU")

    train_ds, val_ds, n_train, n_val = build_datasets()
    print(f"Train: {n_train} images | Val: {n_val} images")

    model = build_model()
    model.summary()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history_path = os.path.join(OUTPUT_DIR, "history.json")

    callbacks = [
        ModelCheckpoint(MODEL_FILE, monitor="val_accuracy", mode="max",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, callbacks=callbacks, verbose=1)
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({elapsed/60:.2f} min)")

    hist_json = {
        "epochs_completed": len(history.history["loss"]),
        "training_time_seconds": round(elapsed, 2),
        "history": {
            "loss":         [float(v) for v in history.history["loss"]],
            "accuracy":     [float(v) for v in history.history["accuracy"]],
            "val_loss":     [float(v) for v in history.history["val_loss"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
    }
    with open(history_path, "w") as f:
        json.dump(hist_json, f, indent=4)

    best = int(np.argmin(history.history["val_loss"])) + 1
    best_acc = history.history["val_accuracy"][best - 1]
    print(f"Best epoch: {best}  |  Val Accuracy: {best_acc*100:.2f}%")
    print(f"Model  -> {MODEL_FILE}")
    print(f"Output -> {OUTPUT_DIR}/\n")

    _plot_curves(hist_json["history"])


def _plot_curves(h: dict):
    """Save accuracy and loss curves to Output/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    epochs = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].plot(epochs, h["accuracy"],     "o-",  label="Train",      color="steelblue", linewidth=2)
    axes[0].plot(epochs, h["val_accuracy"], "s--", label="Validation",  color="coral",     linewidth=2)
    best_e = int(np.argmax(h["val_accuracy"])) + 1
    axes[0].axvline(best_e, color="red", linestyle=":", alpha=0.7, label=f"Best @ Epoch {best_e}")
    axes[0].set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, h["loss"],     "o-",  label="Train",      color="#D97706", linewidth=2)
    axes[1].plot(epochs, h["val_loss"], "s--", label="Validation",  color="#DC2626", linewidth=2)
    min_e = int(np.argmin(h["val_loss"])) + 1
    axes[1].axvline(min_e, color="purple", linestyle=":", alpha=0.7, label=f"Min Loss @ Epoch {min_e}")
    axes[1].set_title("Loss over Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/training_curves.png")


def evaluate():
    """Load saved model, compute metrics, and save all plots to Output/."""
    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(
            f"Model '{MODEL_FILE}' not found. Run training first:\n"
            "  python cnn_dog_cat.py train"
        )

    print(f"Evaluating '{MODEL_FILE}' ...")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    model = tf.keras.models.load_model(MODEL_FILE)
    _, val_ds, _, _ = build_datasets()

    y_true_list, y_prob_list = [], []
    sample_imgs, sample_true, sample_pred, sample_conf = [], [], [], []

    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0).flatten()
        y_prob_list.extend(preds.tolist())
        y_true_list.extend(batch_y.numpy().tolist())
        if len(sample_imgs) < 12:
            needed = 12 - len(sample_imgs)
            for i in range(min(needed, len(preds))):
                sample_imgs.append(batch_x[i].numpy())
                sample_true.append(int(batch_y[i].numpy()))
                pred_cls = 1 if preds[i] >= 0.5 else 0
                sample_pred.append(pred_cls)
                conf = preds[i] if pred_cls == 1 else (1.0 - preds[i])
                sample_conf.append(float(conf))

    y_true = np.array(y_true_list, dtype=int)
    y_pred = (np.array(y_prob_list) >= 0.5).astype(int)

    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred,    zero_division=0))
    f1   = float(f1_score(y_true, y_pred,        zero_division=0))
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat (0)", "Dog (1)"],
                yticklabels=["Cat (0)", "Dog (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Confusion Matrix  (Accuracy: {acc*100:.2f}%)",
              fontsize=13, fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/confusion_matrix.png")

    if sample_imgs:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), dpi=150)
        axes = axes.flatten()
        for idx in range(min(12, len(sample_imgs))):
            ax = axes[idx]
            true_name = CLASS_MAP[sample_true[idx]]
            pred_name = CLASS_MAP[sample_pred[idx]]
            conf_pct  = sample_conf[idx] * 100
            color     = "#15803D" if sample_true[idx] == sample_pred[idx] else "#B91C1C"
            ax.imshow(sample_imgs[idx])
            ax.set_title(f"Pred: {pred_name} ({conf_pct:.1f}%)\nTrue: {true_name}",
                         color=color, fontsize=9, fontweight="bold")
            ax.axis("off")
        plt.suptitle("Validation Sample Predictions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"))
        plt.close()
        print(f"Saved: {OUTPUT_DIR}/sample_predictions.png")

    hist_path = os.path.join(OUTPUT_DIR, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            _plot_curves(json.load(f)["history"])

    print(f"\nValidation Samples : {len(y_true)}")
    print(f"Accuracy           : {acc * 100:.2f}%")
    print(f"Precision (Dog)    : {prec:.4f}")
    print(f"Recall    (Dog)    : {rec:.4f}")
    print(f"F1-Score  (Dog)    : {f1:.4f}")
    print(f"Confusion Matrix   : TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")
    print(classification_report(y_true, y_pred,
                                target_names=["Cat (0)", "Dog (1)"], digits=4))


_model_cache: Optional[tf.keras.Model] = None
_model_cache_path: Optional[str]       = None


def _load_model() -> tf.keras.Model:
    global _model_cache, _model_cache_path
    if _model_cache is None or _model_cache_path != MODEL_FILE:
        if not os.path.isfile(MODEL_FILE):
            raise FileNotFoundError(
                f"Model '{MODEL_FILE}' not found. Train first:\n"
                "  python cnn_dog_cat.py train"
            )
        _model_cache      = tf.keras.models.load_model(MODEL_FILE)
        _model_cache_path = MODEL_FILE
    return _model_cache


def predict_image(
    image_input,
    save_viz: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify one image as Cat or Dog.

    Parameters
    ----------
    image_input : str (file path) or PIL.Image
    save_viz    : optional output path to save a labelled result image
    """
    model = _load_model()

    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Expected path or PIL.Image, got {type(image_input)}")

    arr     = np.array(pil_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR),
                       dtype=np.float32) / 255.0
    tensor  = np.expand_dims(arr, 0)
    sigmoid = float(model.predict(tensor, verbose=0)[0][0])

    if sigmoid >= 0.5:
        pred, cls_id, conf = "Dog", 1, sigmoid * 100.0
    else:
        pred, cls_id, conf = "Cat", 0, (1.0 - sigmoid) * 100.0

    result = {
        "predicted_class":    pred,
        "class_id":           cls_id,
        "confidence":         round(conf, 2),
        "raw_sigmoid_output": round(sigmoid, 4),
        "probabilities":      {
            "Cat": round((1.0 - sigmoid) * 100, 2),
            "Dog": round(sigmoid * 100,          2),
        },
    }

    if save_viz:
        os.makedirs(os.path.dirname(save_viz) or ".", exist_ok=True)
        color = "#15803D" if pred == "Dog" else "#1D4ED8"
        plt.figure(figsize=(5, 5.5), dpi=150)
        plt.imshow(pil_img)
        plt.title(
            f"Prediction: {pred}  ({conf:.1f}%)\n"
            f"Cat: {result['probabilities']['Cat']:.1f}%  |  "
            f"Dog: {result['probabilities']['Dog']:.1f}%",
            fontsize=12, fontweight="bold", color=color, pad=10,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_viz)
        plt.close()
        result["saved_visualization"] = save_viz

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dog vs Cat CNN  --  train | evaluate | predict"
    )
    sub = parser.add_subparsers(dest="cmd")

    tp = sub.add_parser("train", help="Train the CNN on your dataset")
    tp.add_argument("--epochs",        type=int,   default=EPOCHS)
    tp.add_argument("--batch-size",    type=int,   default=BATCH_SIZE)
    tp.add_argument("--lr",            type=float, default=LEARNING_RATE)
    tp.add_argument("--sample-limit",  type=int,   default=SAMPLE_LIMIT)
    tp.add_argument("--data-dir",      type=str,   default=DATA_DIR)

    ep = sub.add_parser("evaluate", help="Evaluate saved model and save plots")
    ep.add_argument("--data-dir", type=str, default=DATA_DIR)

    pp = sub.add_parser("predict", help="Classify a single image")
    pp.add_argument("--image",    type=str, required=True)
    pp.add_argument("--save-viz", type=str, default=None,
                    help="Path to save the labelled result image")

    args = parser.parse_args()

    if args.cmd == "train":
        EPOCHS       = args.epochs
        BATCH_SIZE   = args.batch_size
        LEARNING_RATE = args.lr
        SAMPLE_LIMIT  = args.sample_limit
        DATA_DIR      = args.data_dir
        train()

    elif args.cmd == "evaluate":
        DATA_DIR = args.data_dir
        evaluate()

    elif args.cmd == "predict":
        res = predict_image(args.image, save_viz=args.save_viz)
        print(f"\nPrediction  : {res['predicted_class']}")
        print(f"Confidence  : {res['confidence']}%")
        print(f"Sigmoid out : {res['raw_sigmoid_output']}")
        print(f"Cat prob    : {res['probabilities']['Cat']}%")
        print(f"Dog prob    : {res['probabilities']['Dog']}%")
        if "saved_visualization" in res:
            print(f"Saved viz   : {res['saved_visualization']}")

    else:
        parser.print_help()
