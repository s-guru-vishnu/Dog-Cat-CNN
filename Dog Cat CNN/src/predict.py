"""
Dog vs Cat Image Classification - Inference & Prediction Module
================================================================
Performs single-image inference using the trained 3-Layer CNN model.

Features:
- Validates and loads single image (JPG, JPEG, PNG, BMP).
- Preprocesses image: 128x128 resize and [0, 1] normalization.
- Forward passes through trained 'dog_cat_cnn.keras' model.
- Maps sigmoid output to Cat (0) or Dog (1) with calibrated confidence score.
- Optionally displays or saves annotated visualization image.
- Callable both as a modular Python function and as a command-line script.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import IMAGE_SIZE, CLASS_MAP


# Cached model instance for fast repeated predictions
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def get_model(model_path: str = "models/dog_cat_cnn.keras") -> tf.keras.Model:
    """
    Loads or retrieves the cached trained Keras model.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                f"Please run 'python -m src.train' first."
            )
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        _CACHED_MODEL = tf.keras.models.load_model(model_path)
        _CACHED_MODEL_PATH = model_path
    return _CACHED_MODEL


def preprocess_image(image_input) -> Tuple[np.ndarray, Image.Image]:
    """
    Loads, converts to RGB, resizes to (128, 128), and normalizes pixel values to [0.0, 1.0].

    Args:
        image_input: Either a file path string or a PIL Image instance.

    Returns:
        Tuple of (batch_tensor, original_pil_image)
    """
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image file does not exist: {image_input}")
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Expected str path or PIL.Image, got {type(image_input)}")

    resized_img = pil_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    batch_tensor = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

    return batch_tensor, pil_img


def predict_image(
    image_input,
    model_path: str = "models/dog_cat_cnn.keras",
    save_viz_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predicts whether the input image contains a Cat or Dog.

    Args:
        image_input: File path (str) or PIL Image.
        model_path: Path to the .keras model weights.
        save_viz_path: Optional output path to save the visualized prediction plot.

    Returns:
        Dict with prediction details.
    """
    model = get_model(model_path)
    tensor, pil_img = preprocess_image(image_input)

    # Sigmoid output in range [0.0, 1.0]
    sigmoid_output = float(model.predict(tensor, verbose=0)[0][0])

    # Class threshold: 0.5 (Consistent with training)
    if sigmoid_output >= 0.5:
        predicted_class = "Dog"
        class_id = 1
        confidence = sigmoid_output * 100.0
    else:
        predicted_class = "Cat"
        class_id = 0
        confidence = (1.0 - sigmoid_output) * 100.0

    dog_probability = sigmoid_output * 100.0
    cat_probability = (1.0 - sigmoid_output) * 100.0

    result = {
        "predicted_class": predicted_class,
        "class_id": class_id,
        "confidence": round(confidence, 2),
        "raw_sigmoid_output": round(sigmoid_output, 4),
        "probabilities": {
            "Cat": round(cat_probability, 2),
            "Dog": round(dog_probability, 2)
        }
    }

    # Save visual plot if requested
    if save_viz_path:
        os.makedirs(os.path.dirname(save_viz_path) or ".", exist_ok=True)
        plt.figure(figsize=(5, 5.5), dpi=200)
        plt.imshow(pil_img)
        color = "#15803D" if predicted_class == "Dog" else "#1D4ED8"
        plt.title(
            f"Prediction: {predicted_class} ({confidence:.1f}%)\n"
            f"[Cat: {cat_probability:.1f}% | Dog: {dog_probability:.1f}%]",
            fontsize=12,
            fontweight="bold",
            color=color,
            pad=10
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_viz_path)
        plt.close()
        result["saved_visualization"] = save_viz_path

    return result


def main():
    parser = argparse.ArgumentParser(description="Dog vs Cat Single-Image Classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model-path", type=str, default="models/dog_cat_cnn.keras", help="Path to model file")
    parser.add_argument("--save-viz", type=str, default=None, help="Path to save output visualization image")

    args = parser.parse_args()

    print("=" * 70)
    print("                 IMAGE CLASSIFICATION INFERENCE")
    print("=" * 70)
    print(f"[*] Input Image:  {args.image}")
    print(f"[*] Model Path:   {args.model_path}")

    try:
        res = predict_image(
            image_input=args.image,
            model_path=args.model_path,
            save_viz_path=args.save_viz
        )

        print("-" * 70)
        print(f"  PREDICTED CLASS:     {res['predicted_class'].upper()}")
        print(f"  CONFIDENCE:          {res['confidence']}%")
        print(f"  SIGMOID VALUE:       {res['raw_sigmoid_output']}")
        print(f"  CAT PROBABILITY:     {res['probabilities']['Cat']}%")
        print(f"  DOG PROBABILITY:     {res['probabilities']['Dog']}%")
        if "saved_visualization" in res:
            print(f"  SAVED VISUALIZATION: {res['saved_visualization']}")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"[!] Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
