# 🤖 Transformer-Based Next-Token Prediction

A self-contained **Transformer language model** that learns to predict the next character in a text sequence — built from scratch using TensorFlow/Keras with explicit Multi-Head Self-Attention and Positional Encoding. Everything is in one file.

---

## 📁 Project Structure

```
Transformer Based Next Token Prediction/
├── transformer_next_token.py   # ← single file: all logic here
├── requirements.txt
├── README.md
├── transformer_lm.keras        # saved model (generated after training)
└── Output/
    ├── training_curves.png     # loss & accuracy over epochs
    ├── attention_map.png       # multi-head attention heatmap
    ├── history.json            # per-epoch metrics
    └── tokenizer.json          # character vocabulary
```

---

## 🏗️ Model Architecture

```
Token IDs (B, L)
      │
      ▼
  Embedding (vocab_size → embed_dim)
      │
      ▼
  Positional Encoding  +  Dropout
      │
      ▼
  ┌─────────────────────────────┐
  │   Transformer Block × N    │  ← stacked N times
  │  ┌────────────────────────┐ │
  │  │  LayerNorm             │ │
  │  │  Causal MHSA           │ │  ← self-attention with causal mask
  │  │  Dropout + Residual    │ │
  │  │  LayerNorm             │ │
  │  │  Feed-Forward (MLP)    │ │
  │  │  Dropout + Residual    │ │
  │  └────────────────────────┘ │
  └─────────────────────────────┘
      │
      ▼
  LayerNorm → Dense(vocab_size)
      │
      ▼
  Logits (B, L, vocab_size)
```

### Default Hyperparameters

| Parameter     | Value | Description                          |
|---------------|-------|--------------------------------------|
| `SEQ_LEN`     | 64    | Context window (tokens per step)     |
| `EMBED_DIM`   | 128   | Token embedding dimension            |
| `NUM_HEADS`   | 4     | Parallel attention heads             |
| `FF_DIM`      | 256   | Feed-forward hidden size             |
| `NUM_LAYERS`  | 2     | Stacked Transformer blocks           |
| `DROPOUT_RATE`| 0.1   | Dropout applied after attention & FFN|
| `EPOCHS`      | 30    | Max training epochs                  |
| `LEARNING_RATE`| 3e-4 | Adam initial LR                      |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# 1. Train on built-in corpus (no dataset needed)
python transformer_next_token.py train

# 2. Train on your own text file
python transformer_next_token.py train --text-file my_book.txt

# 3. Generate text from a seed
python transformer_next_token.py generate --seed "The sun rose" --length 300 --temperature 0.8

# 4. Plot training curves + attention map
python transformer_next_token.py visualize --seed "The sun rose"
```

---

## 🌡️ Temperature Sampling

| Temperature | Effect |
|-------------|--------|
| `< 0.5`     | Very focused — repeats common patterns |
| `0.7–0.9`   | Balanced creativity (recommended) |
| `1.0`       | Raw model distribution |
| `> 1.2`     | Very random / creative |

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `transformer_lm.keras` | Best checkpoint (by val loss) |
| `Output/training_curves.png` | Loss & accuracy over epochs |
| `Output/attention_map.png` | Per-head attention weight heatmap |
| `Output/tokenizer.json` | Character vocabulary (char → id) |
| `Output/history.json` | Per-epoch training metrics |

---

## 📦 Requirements

```
tensorflow>=2.15.0
keras>=3.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

