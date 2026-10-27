import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time
import random
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, Model


RANDOM_SEED    = 42
SEQ_LEN        = 64        # tokens the model sees at once (context window)
EMBED_DIM      = 128       # token embedding size
NUM_HEADS      = 4         # parallel attention heads
FF_DIM         = 256       # feed-forward hidden size
NUM_LAYERS     = 2         # stacked Transformer blocks
DROPOUT_RATE   = 0.1
BATCH_SIZE     = 64
EPOCHS         = 30
LEARNING_RATE  = 3e-4

MODEL_FILE     = "transformer_lm.keras"
OUTPUT_DIR     = "Output"

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

_BUILTIN_TEXT = """
The sun rose over the mountains, casting golden light across the quiet valley.
Birds began to sing their morning songs as the mist slowly lifted from the river.
A young traveller walked the winding path, carrying nothing but a worn satchel and a curious mind.
She had set out before dawn, guided by the stars and the stories her grandmother had told her.
The road ahead was long, but every step brought new sights and sounds to wonder at.
Ancient trees lined the path, their gnarled roots weaving between stones worn smooth by centuries of rain.
She stopped by a clear stream to drink, watching the water tumble over pebbles, each one glinting in the early light.
A fox paused on the opposite bank, regarded her with calm amber eyes, then vanished silently into the ferns.
By midday she reached the crest of the first hill and looked back at the village far below.
Smoke curled from chimneys, and the church bell rang faintly across the distance, marking the hour.
She turned her face forward, breathed in the cool mountain air, and walked on.
The forest deepened as afternoon shadows lengthened.
Owls slept in hollow oaks. Mushrooms clustered at the roots of fallen logs.
The traveller hummed a tune her father had taught her, the notes weaving between the trees like a second path.
As dusk painted the sky in shades of rose and amber, she found a stone cottage beside a pond.
An old man sat outside, whittling a piece of wood. He looked up without surprise, as though he had been expecting her.
"You look hungry," he said. "Come in. There is soup on the fire."
She thanked him, ducked through the low doorway, and sat by the warm hearth.
They spoke little, but the silence was comfortable, full of the crackle of fire and the call of frogs outside.
That night she slept soundly on a cot of woven rushes, dreaming of the road that still lay ahead.
In the morning she left a small coin on the table, shouldered her satchel, and stepped back into the bright world.
The path continued north, climbing through meadows scattered with wildflowers, purple and white and yellow.
She counted the colours to pass the time: lavender, clover, yarrow, thistle.
A shepherd watched her from a hillside, his crook resting across his shoulders, his dog asleep at his feet.
She waved. He nodded. The dog opened one eye and closed it again.
At the edge of a pine forest a wooden sign pointed two ways. Left: the valley road, easy and familiar.
Right: the mountain pass, steep and uncertain.
She chose right without hesitation, because the stories worth telling always came from the harder road.
The wind picked up as she climbed, pulling at her coat and whistling through the rocks.
Below her the world spread out like a painted map, rivers threading between fields, villages like scattered seeds.
She thought about all the people down there, each one the centre of their own story, each one carrying their own small satchel of joys and worries.
The top of the pass came suddenly, a notch in the ridge where the wind howled and the view opened on both sides.
She stood there for a long moment, heart beating fast, cold air sharp in her lungs.
Then she began the descent, step by careful step, into the unknown country beyond.
"""


class CharTokenizer:
    """
    Maps every unique character in a corpus to an integer id and back.

    Attributes
    ----------
    vocab_size : int          - number of unique characters
    char2id    : dict[str,int]
    id2char    : dict[int,str]
    """

    def __init__(self):
        self.char2id: dict = {}
        self.id2char: dict = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        self.char2id  = {c: i for i, c in enumerate(chars)}
        self.id2char  = {i: c for c, i in self.char2id.items()}
        self.vocab_size = len(chars)
        return self

    def encode(self, text: str) -> List[int]:
        return [self.char2id[c] for c in text if c in self.char2id]

    def decode(self, ids) -> str:
        return "".join(self.id2char.get(i, "?") for i in ids)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2id": self.char2id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.char2id   = data["char2id"]
        tok.id2char   = {int(i): c for c, i in tok.char2id.items()}
        tok.vocab_size = len(tok.char2id)
        return tok


def make_dataset(
    ids: List[int],
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Slice a token sequence into (input, target) pairs for next-token prediction.

    Each input  = ids[i : i+seq_len]
    Each target = ids[i+1 : i+seq_len+1]   (shifted one step right)
    """
    total = len(ids) - seq_len
    ids_tensor = tf.constant(ids, dtype=tf.int32)

    def get_pair(i):
        x = ids_tensor[i     : i + seq_len]
        y = ids_tensor[i + 1 : i + seq_len + 1]
        return x, y

    ds = (
        tf.data.Dataset.range(total)
        .shuffle(min(total, 5000), seed=RANDOM_SEED)
        .map(get_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


class PositionalEncoding(layers.Layer):
    """
    Adds sinusoidal positional information to token embeddings.

    For position p and dimension d:
        PE[p, 2i]   = sin(p / 10000^(2i / d_model))
        PE[p, 2i+1] = cos(p / 10000^(2i / d_model))

    This gives every position a unique signature that the model
    can use to understand sequence order without recurrence.
    """

    def __init__(self, seq_len: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len   = seq_len
        self.embed_dim = embed_dim

        positions = np.arange(seq_len)[:, np.newaxis]          # (L, 1)
        dims      = np.arange(embed_dim)[np.newaxis, :]         # (1, D)
        angles    = positions / np.power(10000, (2 * (dims // 2)) / embed_dim)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.cast(angles[np.newaxis, :, :], tf.float32)  # (1, L, D)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "embed_dim": self.embed_dim})
        return cfg


class CausalMultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention with a causal (look-ahead) mask.

    Steps for each forward pass
    ---------------------------
    1. Project input X into Q, K, V matrices using learned linear layers.
    2. Split Q, K, V across `num_heads` attention heads.
    3. Compute scaled dot-product scores:  scores = QKᵀ / √d_k
    4. Apply causal mask: positions j > i get score −∞ (-> 0 after softmax)
       so token i can only attend to tokens 0 ... i (no future peeking).
    5. Softmax -> attention weights  α
    6. Weighted sum: output = α · V
    7. Concatenate heads and project back to embed_dim.

    Parameters
    ----------
    embed_dim  : total embedding dimension (must be divisible by num_heads)
    num_heads  : number of parallel attention heads
    dropout    : attention-weight dropout rate
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k       = embed_dim // num_heads

        self.W_q  = layers.Dense(embed_dim, use_bias=False)
        self.W_k  = layers.Dense(embed_dim, use_bias=False)
        self.W_v  = layers.Dense(embed_dim, use_bias=False)
        self.W_o  = layers.Dense(embed_dim)
        self.drop = layers.Dropout(dropout)

    def _split_heads(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.num_heads, self.d_k))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]

        Q = self._split_heads(self.W_q(x), B)   # (B, H, L, d_k)
        K = self._split_heads(self.W_k(x), B)
        V = self._split_heads(self.W_v(x), B)

        scale  = tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        scores = tf.matmul(Q, K, transpose_b=True) / scale   # (B, H, L, L)

        mask = 1 - tf.linalg.band_part(tf.ones((L, L)), -1, 0)
        scores -= mask * 1e9

        weights = tf.nn.softmax(scores, axis=-1)              # (B, H, L, L)
        weights = self.drop(weights, training=training)

        out = tf.matmul(weights, V)                           # (B, H, L, d_k)
        out = tf.transpose(out, [0, 2, 1, 3])                 # (B, L, H, d_k)
        out = tf.reshape(out, (B, -1, self.embed_dim))        # (B, L, D)
        return self.W_o(out), weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                    "dropout": self.drop.rate})
        return cfg


class TransformerBlock(layers.Layer):
    """
    One Transformer decoder block:
        x -> LayerNorm -> CausalMHSA -> Dropout -> residual
          -> LayerNorm -> FFN         -> Dropout -> residual

    The Feed-Forward Network (FFN) is a two-layer MLP applied
    independently to every position:
        FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂

    Pre-LayerNorm (norm before attention) tends to train more stably.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn   = CausalMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn    = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.norm2  = layers.LayerNormalization(epsilon=1e-6)
        self.drop1  = layers.Dropout(dropout)
        self.drop2  = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out, _ = self.attn(self.norm1(x), training=training)
        x = x + self.drop1(attn_out, training=training)
        ffn_out     = self.ffn(self.norm2(x))
        x = x + self.drop2(ffn_out,  training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.attn.embed_dim,
            "num_heads": self.attn.num_heads,
            "ff_dim":    self.ffn.layers[0].units,
            "dropout":   self.drop1.rate,
        })
        return cfg


def build_transformer_lm(
    vocab_size: int,
    seq_len: int      = SEQ_LEN,
    embed_dim: int    = EMBED_DIM,
    num_heads: int    = NUM_HEADS,
    ff_dim: int       = FF_DIM,
    num_layers: int   = NUM_LAYERS,
    dropout: float    = DROPOUT_RATE,
) -> Model:
    """
    Assemble the full Transformer Language Model.

    Input  : integer token ids  shape (B, L)
    Output : logits over vocab  shape (B, L, V)

    Pipeline
    --------
    token ids -> Embedding -> + PositionalEncoding -> Dropout
              -> TransformerBlock x num_layers
              -> LayerNorm -> Dense(vocab_size)  ← output logits
    """
    inp = layers.Input(shape=(seq_len,), dtype=tf.int32, name="token_ids")

    x = layers.Embedding(vocab_size, embed_dim, name="token_embedding")(inp)
    x = PositionalEncoding(seq_len, embed_dim, name="pos_encoding")(x)
    x = layers.Dropout(dropout, name="embed_dropout")(x)

    for i in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout,
                             name=f"transformer_block_{i}")(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)

    logits = layers.Dense(vocab_size, name="lm_head")(x)

    model = Model(inputs=inp, outputs=logits, name="transformer_lm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train(text_file: Optional[str] = None):
    """
    Train the Transformer language model.

    Parameters
    ----------
    text_file : path to a UTF-8 text file.  If None, the built-in corpus is used.
    """
    print("=" * 60)
    print("   Transformer Next-Token Prediction -- Training")
    print("=" * 60)

    if text_file and os.path.isfile(text_file):
        with open(text_file, encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded corpus: '{text_file}'  ({len(text):,} chars)")
    else:
        text = _BUILTIN_TEXT
        if text_file:
            print(f"'{text_file}' not found -- using built-in corpus.")
        else:
            print("Using built-in corpus.")
    print(f"Corpus length: {len(text):,} characters")

    tok = CharTokenizer().fit(text)
    print(f"Vocabulary size: {tok.vocab_size} characters")
    ids = tok.encode(text)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tok.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))

    split      = int(0.9 * len(ids))
    train_ids  = ids[:split]
    val_ids    = ids[split:]
    print(f"Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

    train_ds = make_dataset(train_ids)
    val_ds   = make_dataset(val_ids)

    model = build_transformer_lm(vocab_size=tok.vocab_size)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_FILE, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks, verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"Model -> {MODEL_FILE}")

    hist_path = os.path.join(OUTPUT_DIR, "history.json")
    hist_dict = {
        "epochs_completed": len(history.history["loss"]),
        "training_time_seconds": round(elapsed, 2),
        "history": {
            "loss":         [float(v) for v in history.history["loss"]],
            "accuracy":     [float(v) for v in history.history["accuracy"]],
            "val_loss":     [float(v) for v in history.history["val_loss"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
    }
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=4)

    _plot_training_curves(hist_dict["history"])

    print("\n-- Sample generation after training --")
    seed_text = text[:40]
    generated = generate(seed_text, length=200, temperature=0.8, tok=tok)
    print(f"Seed : {repr(seed_text)}")
    print(f"Output:\n{generated}\n")


def generate(
    seed_text: str,
    length: int = 200,
    temperature: float = 0.8,
    tok: Optional[CharTokenizer] = None,
    model: Optional[Model] = None,
) -> str:
    """
    Auto-regressively generate `length` new characters given a seed string.

    How temperature works
    ---------------------
    logits -> divide by T -> softmax -> sample
    • T < 1 : sharper distribution -> more predictable / repetitive
    • T = 1 : raw model distribution
    • T > 1 : flatter distribution -> more creative / random

    Parameters
    ----------
    seed_text   : prompt string (must be at least 1 char in vocab)
    length      : number of characters to generate
    temperature : sampling temperature  (0 < T, typically 0.5-1.5)
    tok         : CharTokenizer (loaded from Output/ if None)
    model       : Keras model (loaded from MODEL_FILE if None)
    """
    tok_path   = os.path.join(OUTPUT_DIR, "tokenizer.json")
    if tok is None:
        if not os.path.isfile(tok_path):
            raise FileNotFoundError(f"Tokenizer not found at '{tok_path}'. Train first.")
        tok = CharTokenizer.load(tok_path)
    if model is None:
        if not os.path.isfile(MODEL_FILE):
            raise FileNotFoundError(f"Model not found at '{MODEL_FILE}'. Train first.")
        model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects={
                "PositionalEncoding":            PositionalEncoding,
                "CausalMultiHeadSelfAttention":  CausalMultiHeadSelfAttention,
                "TransformerBlock":              TransformerBlock,
            },
        )

    ids      = tok.encode(seed_text)
    generated = list(ids)

    for _ in range(length):
        context = generated[-SEQ_LEN:]
        pad_len = SEQ_LEN - len(context)
        padded  = [0] * pad_len + context          # left-pad with 0
        inp     = np.array([padded], dtype=np.int32)

        logits  = model.predict(inp, verbose=0)[0]  # (SEQ_LEN, vocab_size)
        last_logits = logits[len(context) - 1]       # prediction after last token

        scaled  = last_logits / max(temperature, 1e-8)
        probs   = tf.nn.softmax(scaled).numpy()
        next_id = np.random.choice(len(probs), p=probs)
        generated.append(next_id)

    return tok.decode(generated[len(ids):])


def _plot_training_curves(h: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    epochs = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].plot(epochs, h["loss"],         "o-",  color="steelblue", label="Train Loss",   linewidth=2)
    axes[0].plot(epochs, h["val_loss"],     "s--", color="coral",     label="Val Loss",     linewidth=2)
    axes[0].set_title("Loss over Epochs",   fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, h["accuracy"],     "o-",  color="steelblue", label="Train Acc",    linewidth=2)
    axes[1].plot(epochs, h["val_accuracy"], "s--", color="coral",     label="Val Acc",      linewidth=2)
    axes[1].set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Token Accuracy")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Transformer LM -- Training Metrics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def visualize_attention(
    seed_text: str = "The sun rose over",
    tok: Optional[CharTokenizer] = None,
    model: Optional[Model] = None,
):
    """
    Extract and plot the attention weight matrix from the first Transformer block
    for a given seed string.  Saved to Output/attention_map.png.
    """
    tok_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    if tok is None:
        if not os.path.isfile(tok_path):
            print(f"Tokenizer not found at '{tok_path}'. Skipping attention viz.")
            return
        tok = CharTokenizer.load(tok_path)
    if model is None:
        if not os.path.isfile(MODEL_FILE):
            print(f"Model not found at '{MODEL_FILE}'. Skipping attention viz.")
            return
        model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects={
                "PositionalEncoding":           PositionalEncoding,
                "CausalMultiHeadSelfAttention": CausalMultiHeadSelfAttention,
                "TransformerBlock":             TransformerBlock,
            },
        )

    ids     = tok.encode(seed_text[:SEQ_LEN])
    pad_len = SEQ_LEN - len(ids)
    padded  = np.array([[0] * pad_len + ids], dtype=np.int32)
    L       = len(ids)
    labels  = list(seed_text[:L])

    try:
        attn_layer = model.get_layer("transformer_block_0").attn
        emb_layer   = model.get_layer("token_embedding")
        pos_layer   = model.get_layer("pos_encoding")
        norm1_layer = model.get_layer("transformer_block_0").norm1

        x   = emb_layer(padded)
        x   = pos_layer(x)
        x_n = norm1_layer(x)
        _, weights = attn_layer(x_n, training=False)   # (1, H, L, L)

        weights_np = weights.numpy()[0]                 # (H, L, L)
        n_heads    = weights_np.shape[0]

        fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5), dpi=120)
        if n_heads == 1:
            axes = [axes]

        for h_idx, ax in enumerate(axes):
            w = weights_np[h_idx, pad_len:, pad_len:][:L, :L]
            sns.heatmap(w, ax=ax, cmap="Blues", xticklabels=labels,
                        yticklabels=labels, cbar=(h_idx == n_heads - 1),
                        vmin=0, vmax=w.max())
            ax.set_title(f"Head {h_idx + 1}", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.tick_params(axis="y", rotation=0,  labelsize=8)

        plt.suptitle(f'Attention Weights -- Block 0\nSeed: "{seed_text[:L]}"',
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "attention_map.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Attention visualisation skipped: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transformer Next-Token Prediction  --  train | generate | visualize"
    )
    sub = parser.add_subparsers(dest="cmd")

    tp = sub.add_parser("train", help="Train the Transformer language model")
    tp.add_argument("--text-file",   type=str,   default=None,
                    help="Path to a .txt corpus file (default: built-in text)")
    tp.add_argument("--epochs",      type=int,   default=EPOCHS)
    tp.add_argument("--seq-len",     type=int,   default=SEQ_LEN)
    tp.add_argument("--embed-dim",   type=int,   default=EMBED_DIM)
    tp.add_argument("--num-heads",   type=int,   default=NUM_HEADS)
    tp.add_argument("--ff-dim",      type=int,   default=FF_DIM)
    tp.add_argument("--num-layers",  type=int,   default=NUM_LAYERS)
    tp.add_argument("--lr",          type=float, default=LEARNING_RATE)
    tp.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)

    gp = sub.add_parser("generate", help="Generate text from a seed prompt")
    gp.add_argument("--seed",        type=str,   default="The sun rose over the",
                    help="Seed text to continue from")
    gp.add_argument("--length",      type=int,   default=300,
                    help="Number of characters to generate")
    gp.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature (lower=focused, higher=creative)")

    vp = sub.add_parser("visualize", help="Plot attention maps and training curves")
    vp.add_argument("--seed",  type=str, default="The sun rose over",
                    help="Short text to visualise attention for")

    args = parser.parse_args()

    if args.cmd == "train":
        EPOCHS        = args.epochs
        SEQ_LEN       = args.seq_len
        EMBED_DIM     = args.embed_dim
        NUM_HEADS     = args.num_heads
        FF_DIM        = args.ff_dim
        NUM_LAYERS    = args.num_layers
        LEARNING_RATE = args.lr
        BATCH_SIZE    = args.batch_size
        train(text_file=args.text_file)

    elif args.cmd == "generate":
        text = generate(
            seed_text   = args.seed,
            length      = args.length,
            temperature = args.temperature,
        )
        print(f"\nSeed   : {repr(args.seed)}")
        print(f"Output :\n{text}\n")

    elif args.cmd == "visualize":
        hist_path = os.path.join(OUTPUT_DIR, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                _plot_training_curves(json.load(f)["history"])
        visualize_attention(seed_text=args.seed)

    else:
        parser.print_help()

