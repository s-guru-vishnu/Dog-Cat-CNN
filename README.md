# 🐾 Dog vs. Cat Image Classification Using a 3-Layer Convolutional Neural Network (CNN)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic, production-grade deep learning project implementing a Convolutional Neural Network (CNN) with **exactly three convolutional layers** to classify images of cats and dogs. Built from scratch without transfer learning or pre-trained backbones, emphasizing mathematical rigor, data integrity, reproducible experimentation, and educational clarity.

---

## 📋 Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Project Objectives](#3-project-objectives)
- [4. Technologies Used](#4-technologies-used)
- [5. Project Directory Structure](#5-project-directory-structure)
- [6. Dataset & Pipeline Integrity](#6-dataset--pipeline-integrity)
- [7. CNN Architecture Details](#7-cnn-architecture-details)
- [8. Installation & Environment Setup](#8-installation--environment-setup)
- [9. How to Run](#9-how-to-run)
  - [Step 1: Dataset Validation & Preparation](#step-1-dataset-validation--preparation)
  - [Step 2: Inspect Model Architecture](#step-2-inspect-model-architecture)
  - [Step 3: Train the Model](#step-3-train-the-model)
  - [Step 4: Evaluate & Generate Plots](#step-4-evaluate--generate-plots)
  - [Step 5: Run Single-Image Predictions (CLI)](#step-5-run-single-image-predictions-cli)
  - [Step 6: Launch Streamlit Web Application](#step-6-launch-streamlit-web-application)
- [10. Empirical Results & Visualizations](#10-empirical-results--visualizations)
- [11. Limitations & Future Scope](#11-limitations--future-scope)

---

## 1. Project Overview
Binary image classification is a cornerstone benchmark in computer vision. This project designs, trains, evaluates, and deploys a lightweight Convolutional Neural Network capable of distinguishing between feline (*Felis catus*) and canine (*Canis lupus familiaris*) imagery.

Unlike modern trends that rely on multi-hundred-million-parameter transfer learning models (e.g., ResNet, EfficientNet), this project demonstrates how an intentionally constrained **3-layer Conv2D architecture** can effectively learn hierarchical spatial features directly from raw pixel data.

---

## 2. Problem Statement
Given an arbitrary RGB image of size $128 \times 128 \times 3$, predict whether the subject is a **Cat** (Class 0) or a **Dog** (Class 1) by outputting a calibrated posterior probability $\hat{y} \in [0.0, 1.0]$ using a decision threshold of $\tau = 0.50$. The system must handle real-world image variances, lighting shifts, pose deformations, and occlusions while guarding against corrupted image files and data leakage.

---

## 3. Project Objectives
1. **Strict Architectural Adherence:** Construct a Sequential CNN containing **exactly three Conv2D layers**, each paired with MaxPooling2D, followed by a dense classification head with Dropout regularization.
2. **Robust Data Pipeline:** Ingest data from local archives or ZIP packages, validate headers and pixel decoding with Pillow and TensorFlow, eliminate corrupted/truncated files, and enforce a clean 80/20 train/validation split with fixed random seed (`seed=42`).
3. **Data Leakage Prevention:** Ensure complete mutual exclusivity between training and validation images, applying data augmentation strictly during training.
4. **Reproducible Academic Workflow:** Record epoch-by-epoch loss and accuracy, checkpoint the best model weights based on minimum validation loss, and export standard evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
5. **Interactive Deployment:** Deliver a responsive Streamlit web application enabling real-time image uploads, instantaneous inference, and metric inspection.

---

## 4. Technologies Used
- **Language:** Python (3.10 - 3.14 compatible)
- **Deep Learning Framework:** TensorFlow 2.x & Keras 3.x
- **Image Processing:** Pillow (PIL) & NumPy
- **Scientific Computing & Metrics:** Scikit-learn
- **Data Visualization:** Matplotlib & Seaborn
- **Web Interface:** Streamlit

---

## 5. Project Directory Structure
```text
cnn/
├── data/                                 # Organized dataset directories
│   ├── cats/                             # Clean, validated Cat images
│   └── dogs/                             # Clean, validated Dog images
├── models/
│   └── dog_cat_cnn.keras                 # Checkpoint of the best-performing model
├── results/                              # Generated evaluation artifacts
│   ├── accuracy.png                      # Training vs. Validation Accuracy curve
│   ├── loss.png                          # Training vs. Validation Loss curve
│   ├── confusion_matrix.png              # Annotated Confusion Matrix heatmap
│   ├── sample_predictions.png            # Visual grid of validation predictions
│   ├── history.json                      # Epoch-by-epoch training logs
│   └── metrics.json                      # Quantitative evaluation benchmark
├── src/                                  # Modular source code
│   ├── __init__.py
│   ├── data_loader.py                    # Dataset extraction, validation & tf.data pipeline
│   ├── model.py                          # 3-Layer CNN architecture & layer explanations
│   ├── train.py                          # Training pipeline with checkpointing & early stopping
│   ├── evaluate.py                       # Evaluation pipeline & visualization generator
│   └── predict.py                        # Single-image inference module & CLI
├── archive/                              # Source dataset directory (PetImages/Cat, PetImages/Dog)
├── app.py                                # Streamlit web application
├── requirements.txt                      # Project dependencies
├── README.md                             # Comprehensive project documentation
├── REPORT.md                             # Formal academic project report
└── .gitignore                            # Git exclusion rules
```

---

## 6. Dataset & Pipeline Integrity
The project supports the **Kaggle Dogs vs. Cats** and **Microsoft PetImages** datasets.

### Automatic Ingestion & De-Corruption
Raw computer vision archives often contain corrupted images, truncated files, or channel mismatches (e.g., 2-channel images or invalid BMP headers). Our pipeline implements dual-stage validation:
1. **Pillow Verification:** Checks file integrity via `img.verify()` and `img.convert("RGB")`.
2. **TensorFlow Verification:** Tests `tf.io.decode_image(raw, channels=3, expand_animations=False)`.

During dataset preparation on the 25,000 image dataset, exactly 7 corrupted files (e.g., `cat_002434.jpg`, `dog_001372.jpg`) were identified and purged, guaranteeing 100% stable execution without mid-training crashes.

### Data Splitting & Leakage Prevention
- **Split Ratio:** 80% Training ($N_{train} = 3,200$), 20% Validation ($N_{val} = 800$) for the benchmark run.
- **Stratification:** Class balance is preserved at exactly 50% Cat and 50% Dog across both sets.
- **Seed:** Fixed `seed=42` guarantees reproducibility.
- **Augmentation:** Applied **only** to training batches (`RandomFlip("horizontal")`, `RandomRotation(0.15)`, `RandomZoom(0.1)`, `RandomTranslation(0.08, 0.08)`). Validation images are strictly normalized and resized.

---

## 7. CNN Architecture Details

```text
Input Image: (128, 128, 3)
     │
┌────▼────────────────────────────────────────┐
│ Conv2D (32 filters, 3×3, ReLU, same pad)    │ Output: (128, 128, 32)  | Params: 896
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ MaxPooling2D (2×2 pool size)                │ Output: (64, 64, 32)    | Params: 0
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ Conv2D (64 filters, 3×3, ReLU, same pad)    │ Output: (64, 64, 64)    | Params: 18,496
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ MaxPooling2D (2×2 pool size)                │ Output: (32, 32, 64)    | Params: 0
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ Conv2D (128 filters, 3×3, ReLU, same pad)   │ Output: (32, 32, 128)   | Params: 73,856
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ MaxPooling2D (2×2 pool size)                │ Output: (16, 16, 128)   | Params: 0
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ Flatten                                     │ Output: (32,768)        | Params: 0
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ Dense (128 units, ReLU activation)          │ Output: (128)           | Params: 4,194,432
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ Dropout (rate = 0.50 regularization)        │ Output: (128)           | Params: 0
└────┬────────────────────────────────────────┘
┌────▼────────────────────────────────────────┐
│ Dense Output (1 unit, Sigmoid activation)   │ Output: (1)             | Params: 129
└─────────────────────────────────────────────┘
Total Parameters: 4,287,809 (16.36 MB) | 100% Trainable
```

---

## 8. Installation & Environment Setup

### 1. Clone or Navigate to Directory
```bash
cd /home/bharath/Documents/cnn
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

*(If creating a new environment:)*
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. How to Run

### Step 1: Dataset Validation & Preparation
Scans the dataset (either `archive/` or a provided ZIP archive), purges corrupt files, and tests the `tf.data` pipeline:
```bash
python -m src.data_loader
```

### Step 2: Inspect Model Architecture
Verifies layer shapes, parameter counts, and theoretical descriptions:
```bash
python -m src.model
```

### Step 3: Train the Model
Trains the 3-layer CNN with EarlyStopping, ModelCheckpoint, and LR scheduling:
```bash
python -m src.train --epochs 10 --batch-size 32 --sample-limit 2000
```
- Best weights are automatically saved to `models/dog_cat_cnn.keras`.
- Training history is recorded in `results/history.json`.

### Step 4: Evaluate & Generate Plots
Calculates classification metrics on the validation dataset and saves plots to `results/`:
```bash
python -m src.evaluate --sample-limit 2000
```
Generated artifacts:
- `results/accuracy.png`
- `results/loss.png`
- `results/confusion_matrix.png`
- `results/sample_predictions.png`
- `results/metrics.json`

### Step 5: Run Single-Image Predictions (CLI)
Classify any local image file:
```bash
python -m src.predict --image data/cats/cat_000001.jpg --save-viz results/test_prediction.png
```

### Step 6: Launch Streamlit Web Application
Run the interactive graphical user interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test live image classifications, view metrics, and inspect the architecture.

---

## 10. Empirical Results & Visualizations

### Performance Benchmark (Validation/Test Set: $N = 2,000$ | 80/20 Split on RTX 4050 GPU)
| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **79.75%** |
| **Dog Precision** | 0.7717 (77.17%) |
| **Dog Recall** | 0.8450 (84.50%) |
| **Dog F1-Score** | 0.8067 (80.67%) |
| **Cat Precision** | 0.8287 (82.87%) |
| **Cat Recall** | 0.7500 (75.00%) |
| **Cat F1-Score** | 0.7874 (78.74%) |
| **Best Epoch Validation Loss** | 0.5059 |
| **Hardware Accelerator** | NVIDIA GeForce RTX 4050 Laptop GPU (104.2s total training) |

### Confusion Matrix Breakdown ($N = 2,000$)
| Actual \ Predicted | Predicted Cat (0) | Predicted Dog (1) |
| :--- | :---: | :---: |
| **Actual Cat (0)** | **750** (True Negative) | **250** (False Positive) |
| **Actual Dog (1)** | **155** (False Negative) | **845** (True Positive) |

### Key Artifacts
- **Accuracy Curve:** `results/accuracy.png` demonstrates steady training accuracy improvement with validation tracking.
- **Loss Curve:** `results/loss.png` shows smooth descent of Binary Cross-Entropy loss.
- **Heatmap:** `results/confusion_matrix.png` details balanced sensitivity between both classes.
- **Sample Predictions:** `results/sample_predictions.png` showcases qualitative model inferences.

---

## 11. Limitations & Future Scope
1. **Architectural Constraints:** A 3-layer CNN without residual connections or depthwise separable convolutions is capacity-bounded (~70-75% accuracy range).
2. **Resolution Limitations:** Images are downsampled to $128 \times 128$, which can discard subtle facial micro-textures.
3. **Future Improvements:**
   - Exploring Cosine Annealing learning rate schedules.
   - Adding Batch Normalization layers after each Conv2D operation.
   - Extending the architecture to 5 convolutional stages or residual blocks for higher capacity.
# CNN
