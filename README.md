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

A decoder-only **Transformer Language Model** built from scratch using multi-head causal self-attention and positional encoding. Predicts the next token/character auto-regressively with temperature sampling. Single file, zero boilerplate.

| Property | Value |
|----------|-------|
| Task | Auto-regressive Next-Token Prediction |
| Architecture | Decoder-only Transformer (Causal MHSA + FFN + Positional Encoding) |
| Framework | TensorFlow / Keras |
| Dataset | Built-in literary corpus (or custom text file) |
| Entry Point | `python transformer_next_token.py train` |

---

## 🚀 Getting Started

Each project is self-contained with its own `requirements.txt`. Navigate into any project folder and install:

```bash
pip install -r requirements.txt
```

Then follow the instructions in that project's `README.md`.

---

## 🔑 Common Concepts Across Projects

| Concept | Moons ANN | Dog Cat CNN | Transformer Next Token |
|---------|-----------|-------------|------------------------|
| Single-file design | ✅ | ✅ | ✅ |
| Reproducible seeds | ✅ | ✅ | ✅ |
| Early stopping | ✅ | ✅ | ✅ |
| Train/val split | ✅ | ✅ | ✅ |
| Evaluation / Loss curves | ✅ | ✅ | ✅ |
| Model checkpoint | ✅ | ✅ | ✅ |
| Custom layer / block math | — | — | ✅ (Attention, PosEnc, Block) |
| Data augmentation / pipeline | — | ✅ (tf.data) | ✅ (tf.data) |
