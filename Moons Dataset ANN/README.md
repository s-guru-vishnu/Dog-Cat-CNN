# Moons Dataset ANN

A self-contained **Artificial Neural Network (ANN)** that learns to classify the classic two-moons dataset — a non-linearly separable binary classification benchmark — using TensorFlow/Keras.

---

## Project Structure

```
Moons Dataset ANN/
├── ann_moons.py          # ← single-file: data, model, training, evaluation & plots
├── requirements.txt      # Python dependencies
├── ann_moons_model.keras # Saved model (generated after training)
└── Output/
    ├── training_curves.png    # Loss & Accuracy over epochs
    ├── confusion_matrix.png   # Confusion matrix heatmap
    └── decision_boundary.png  # 2-D decision boundary plot
```

---

## What Is the Moons Dataset?

`make_moons` from scikit-learn generates two interleaving half-circle clusters that **cannot** be separated by a straight line. It is a standard benchmark for testing non-linear classifiers.

```
Class 0  ●●●●●    Class 1  ○○○○○
        ●● ○○
       ● ○○○○ ●
      ●○○○   ●●
```

---

## ️ Model Architecture

| Layer            | Units / Filters | Activation | Notes              |
|------------------|-----------------|------------|--------------------|
| Dense (Input)    | 64              | ReLU       | input_shape = (2,) |
| Dropout          | —               | —          | rate = 0.3         |
| Dense (Hidden 1) | 32              | ReLU       |                    |
| Dropout          | —               | —          | rate = 0.3         |
| Dense (Hidden 2) | 16              | ReLU       |                    |
| Dense (Output)   | 1               | Sigmoid    | Binary output      |

- **Optimizer:** Adam  
- **Loss:** Binary Cross-Entropy  
- **Early Stopping:** patience = 15, monitoring `val_loss`  
- **Max Epochs:** 200  

---

## ️ Configuration

All hyperparameters are defined as constants at the top of [`ann_moons.py`](ann_moons.py):

| Parameter      | Default |
|----------------|---------|
| `RANDOM_SEED`  | 42      |
| `n_samples`    | 1 000   |
| `noise`        | 0.2     |
| `test_size`    | 0.2     |
| `batch_size`   | 32      |
| `epochs`       | 200     |
| `patience`     | 15      |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train & evaluate

```bash
python ann_moons.py
```

The script will:
1. Generate the moons dataset  
2. Scale features with `StandardScaler`  
3. Train the ANN with early stopping  
4. Print test accuracy and a classification report  
5. Save three plots to the `Output/` folder  
6. Save the model as `ann_moons_model.keras`  

---

## Outputs

| File | Description |
|------|-------------|
| `Output/training_curves.png` | Train vs Val loss and accuracy over epochs |
| `Output/confusion_matrix.png` | Heatmap of TP / TN / FP / FN |
| `Output/decision_boundary.png` | 2-D decision boundary learned by the ANN |
| `ann_moons_model.keras` | Trained Keras model |

---

## Requirements

```
tensorflow>=2.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Expected Results

| Metric   | Approximate Value |
|----------|-------------------|
| Accuracy | ~ 97–99 %         |
| F1-Score | ~ 0.97–0.99       |

> Results may vary slightly based on the random seed and noise level.

---

## Key Concepts Demonstrated

- **Non-linear classification** with a multi-layer perceptron  
- **Dropout regularisation** to prevent overfitting  
- **Early stopping** for efficient training  
- **Decision boundary visualisation** in 2-D feature space  
- **StandardScaler** preprocessing for neural networks  
