# 🐾 Dog vs Cat CNN Classifier

A self-contained **Convolutional Neural Network** that classifies images as **Dog** or **Cat** — everything in one Python file.

---

## 📁 Project Structure

```
Dog Cat CNN/
├── cnn_dog_cat.py        # ← single file: all logic here
├── requirements.txt
├── README.md
├── dog_cat_cnn.keras     # saved model (generated after training)
└── Output/
    ├── training_curves.png    # accuracy & loss curves
    ├── confusion_matrix.png   # evaluation heatmap
    └── sample_predictions.png # 3×4 grid of val predictions
```

> **Dataset** goes in a `data/` folder (not committed to git):
> ```
> data/
> ├── cats/   ← all cat images
> └── dogs/   ← all dog images
> ```

---

## 🏗️ Model Architecture

| Block   | Layer                          | Output Shape        |
|---------|--------------------------------|---------------------|
| Input   | –                              | (None, 128, 128, 3) |
| Block 1 | Conv2D 32 filters · ReLU       | (None, 128, 128, 32)|
|         | MaxPooling2D 2×2               | (None, 64, 64, 32)  |
| Block 2 | Conv2D 64 filters · ReLU       | (None, 64, 64, 64)  |
|         | MaxPooling2D 2×2               | (None, 32, 32, 64)  |
| Block 3 | Conv2D 128 filters · ReLU      | (None, 32, 32, 128) |
|         | MaxPooling2D 2×2               | (None, 16, 16, 128) |
| Head    | Flatten → Dense 128 → Dropout 0.5 → Dense 1 Sigmoid | (None, 1) |

- **Loss:** Binary Cross-Entropy (label smoothing 0.05)  
- **Optimizer:** Adam · LR 0.0007  
- **Threshold:** sigmoid ≥ 0.5 → Dog, else Cat  

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# 1. Train
python cnn_dog_cat.py train

# 2. Evaluate (generates Output/ plots)
python cnn_dog_cat.py evaluate

# 3. Predict one image
python cnn_dog_cat.py predict --image my_pet.jpg --save-viz Output/result.png
```

---

## ⚙️ Dataset Setup

| Option | How |
|--------|-----|
| Kaggle archive | place extracted `archive/` folder next to `cnn_dog_cat.py` — auto-organised |
| Zip file | drop `cats_and_dogs.zip` inside `data/` — auto-extracted |
| Pre-organised | put images directly in `data/cats/` and `data/dogs/` |

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `dog_cat_cnn.keras` | Best checkpoint (by val accuracy) |
| `Output/training_curves.png` | Side-by-side accuracy & loss plots |
| `Output/confusion_matrix.png` | TP / TN / FP / FN heatmap |
| `Output/sample_predictions.png` | 12 validation image predictions |

---

## 📦 Requirements

```
tensorflow>=2.15.0
keras>=3.0.0
numpy>=1.26.0
pillow>=10.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
```
