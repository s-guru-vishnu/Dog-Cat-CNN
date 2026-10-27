# 🤖 ML Training Workshop

A collection of self-contained machine learning projects, each demonstrating a core deep learning concept with clean, runnable code.

---

## 📂 Projects

### 🌙 [Moons Dataset ANN](./Moons%20Dataset%20ANN/)

A fully-connected **Artificial Neural Network** that classifies the non-linearly separable two-moons dataset. Single file, zero boilerplate.

| Property | Value |
|----------|-------|
| Task | Binary Classification |
| Architecture | ANN (Dense + Dropout) |
| Framework | TensorFlow / Keras |
| Dataset | `make_moons` (scikit-learn) |
| Entry Point | `python ann_moons.py` |

---

### 🐾 [Dog Cat CNN](./Dog%20Cat%20CNN/)

A 3-block **Convolutional Neural Network** that distinguishes dog and cat images. Single file with train / evaluate / predict sub-commands.

| Property | Value |
|----------|-------|
| Task | Binary Image Classification |
| Architecture | CNN (3× Conv2D + MaxPool + Dense) |
| Framework | TensorFlow / Keras |
| Dataset | Kaggle Dogs vs. Cats |
| Entry Point | `python cnn_dog_cat.py train` |

---

### 🔤 [Transformer Based Next Token Prediction](./Transformer%20Based%20Next%20Token%20Prediction/)

*(Coming soon)*

---

## 🚀 Getting Started

Each project is self-contained with its own `requirements.txt`. Navigate into any project folder and install:

```bash
pip install -r requirements.txt
```

Then follow the instructions in that project's `README.md`.

---

## 🔑 Common Concepts Across Projects

| Concept | Moons ANN | Dog Cat CNN |
|---------|-----------|-------------|
| Single-file design | ✅ | ✅ |
| Reproducible seeds | ✅ | ✅ |
| Early stopping | ✅ | ✅ |
| Train/val split | ✅ | ✅ |
| Confusion matrix plot | ✅ | ✅ |
| Model checkpoint | ✅ | ✅ |
| Data augmentation | — | ✅ |
| tf.data pipeline | — | ✅ |
