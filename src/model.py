"""
Dog vs Cat Image Classification - CNN Architecture
===================================================
Implementation of the 3-Convolutional-Layer CNN Architecture for binary classification.

Architecture Specifications:
  Input: (128, 128, 3) RGB Image
  Layer 1: Conv2D(32, 3x3, ReLU) + MaxPooling2D(2x2)
  Layer 2: Conv2D(64, 3x3, ReLU) + MaxPooling2D(2x2)
  Layer 3: Conv2D(128, 3x3, ReLU) + MaxPooling2D(2x2)
  Dense & Regularization:
    - Flatten
    - Dense(128, ReLU)
    - Dropout(0.5)
    - Dense(1, Sigmoid)

Optimizer: Adam
Loss: Binary Cross-Entropy
Metric: Accuracy
Class labels: 0 = Cat, 1 = Dog
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Tuple, Dict
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    learning_rate: float = 0.0007,
    padding: str = "same",
    l2_reg: float = 1e-4,
    label_smoothing: float = 0.05
) -> models.Sequential:
    """
    Constructs and compiles the 3-Convolutional-Layer CNN Sequential model.
    Utilizes He-Normal initialization (optimal for ReLU) and L2 regularization
    to prevent dense overfitting and significantly boost generalization accuracy.

    Args:
        input_shape: Shape of input image (height, width, channels).
        learning_rate: Initial learning rate for Adam optimizer.
        padding: Padding mode for Conv2D layers ('same' preserves spatial dimensions before pooling).
        l2_reg: L2 weight decay penalty factor.
        label_smoothing: Label smoothing factor for Binary Cross-Entropy loss.

    Returns:
        Compiled tf.keras.models.Sequential instance.
    """
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = models.Sequential(name="Dog_Cat_3Layer_CNN")

    # Input Layer definition
    model.add(layers.Input(shape=input_shape, name="input_image"))

    # ==========================================
    # Layer 1: Feature Extraction (Low-level)
    # ==========================================
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="conv2d_layer_1"
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        name="maxpool_1"
    ))

    # ==========================================
    # Layer 2: Feature Extraction (Mid-level)
    # ==========================================
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="conv2d_layer_2"
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        name="maxpool_2"
    ))

    # ==========================================
    # Layer 3: Feature Extraction (High-level)
    # ==========================================
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding=padding,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="conv2d_layer_3"
    ))
    model.add(layers.MaxPooling2D(
        pool_size=(2, 2),
        name="maxpool_3"
    ))

    # ==========================================
    # Classification Head
    # ==========================================
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="dense_features"
    ))
    model.add(layers.Dropout(0.5, name="dropout_regularization"))
    model.add(layers.Dense(1, activation="sigmoid", name="output_prediction"))

    # Optimizer, Loss, and Evaluation Metric
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    metrics = ["accuracy"]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def explain_layers() -> Dict[str, str]:
    """
    Returns an academic, theoretical explanation for every functional block in the network.
    """
    return {
        "Input (128x128x3)": (
            "Receives standard 3-channel RGB image resized to 128x128 pixels with normalized [0, 1] floating-point values."
        ),
        "Conv2D Layer 1 (32 filters, 3x3, ReLU)": (
            "Detects fundamental low-level visual primitives including edges, gradients, color transitions, and texture patches. "
            "ReLU activation introduces non-linearity without gradient saturation."
        ),
        "MaxPooling2D 1 (2x2)": (
            "Downsamples feature map dimensions by a factor of 2 (128x128 -> 64x64), reducing computational complexity "
            "and introducing spatial translation invariance."
        ),
        "Conv2D Layer 2 (64 filters, 3x3, ReLU)": (
            "Combines low-level edge features into intermediate structural motifs such as corners, textures, contours, "
            "and localized facial outlines (whiskers, fur patterns, ear shapes)."
        ),
        "MaxPooling2D 2 (2x2)": (
            "Further downsamples spatial resolution (64x64 -> 32x32), condensing receptive field representations."
        ),
        "Conv2D Layer 3 (128 filters, 3x3, ReLU)": (
            "Extracts complex, semantically meaningful high-level domain representations such as snout geometry, "
            "eye positioning, canine muzzle shapes, and feline ear contours."
        ),
        "MaxPooling2D 3 (2x2)": (
            "Final spatial reduction (32x32 -> 16x16), producing a compact, information-dense 16x16x128 tensor."
        ),
        "Flatten": (
            "Unrolls the 3D tensor (16x16x128 = 32,768 scalar values) into a 1D vector suitable for dense neural layers."
        ),
        "Dense (128 units, ReLU)": (
            "Learns non-linear combinations of the extracted high-level features for class discrimination."
        ),
        "Dropout (0.5)": (
            "Randomly zeros out 50% of the hidden activations during training, breaking co-adaptation between neurons "
            "and preventing severe overfitting on the training set."
        ),
        "Dense Output (1 unit, Sigmoid)": (
            "Applies logistic sigmoid function to output a single scalar probability in [0.0, 1.0]. "
            "Values near 0.0 represent Cat (Class 0); values near 1.0 represent Dog (Class 1). Threshold is set at 0.5."
        )
    }


def print_architecture_details(model: models.Sequential):
    """
    Prints a formatted summary, parameter counts, and theoretical explanation.
    """
    print("=" * 80)
    print("           3-LAYER CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE")
    print("=" * 80)
    model.summary()

    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    print("\n" + "-" * 80)
    print(f"Total Parameters:          {total_params:,}")
    print(f"Trainable Parameters:      {trainable_params:,}")
    print(f"Non-Trainable Parameters:  {non_trainable_params:,}")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("                  LAYER-BY-LAYER ACADEMIC EXPLANATION")
    print("=" * 80)
    explanations = explain_layers()
    for layer_name, purpose in explanations.items():
        print(f"\n[+] {layer_name}:")
        print(f"    {purpose}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    cnn_model = build_model()
    print_architecture_details(cnn_model)
