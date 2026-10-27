"""
Dog vs Cat Image Classification - Streamlit Web Application
===========================================================
Interactive, premium web application for dog and cat image classification
using a 3-Convolutional-Layer CNN trained from scratch on Kaggle Dogs vs Cats.

Launch command:
    streamlit run app.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

from src.data_loader import IMAGE_SIZE
from src.predict import preprocess_image
from src.model import explain_layers


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dog vs Cat CNN Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern typography and cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }
    
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 8px 0;
    }
    
    .main-header p {
        color: #E0E7FF !important;
        font-size: 1.05rem;
        margin: 0;
    }
    
    .result-card {
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-top: 16px;
        background: #FFFFFF;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    .badge-cat {
        background-color: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    
    .badge-dog {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Loader with Resource Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classification_model(model_path: str = "models/dog_cat_cnn.keras"):
    """
    Loads and caches the compiled Keras model.
    Guarantees the model is loaded once and never retrained on app launch.
    """
    if not os.path.isfile(model_path):
        return None
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return tf.keras.models.load_model(model_path)


# Load model
MODEL_PATH = "models/dog_cat_cnn.keras"
model = load_classification_model(MODEL_PATH)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🐶🐱 Project Details")
    st.markdown("""
    **Architecture:** Sequential CNN  
    **Convolutional Layers:** Exactly 3 Conv2D  
    **Input Shape:** 128 × 128 × 3 RGB  
    **Loss:** Binary Cross-Entropy  
    **Optimizer:** Adam (`lr=0.001`)  
    **Decision Threshold:** 0.50  
    """)
    st.divider()

    # Load evaluated metrics if available
    metrics_path = "results/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        acc_pct = metrics_data.get("overall_metrics", {}).get("accuracy_percent", "N/A")
        val_samples = metrics_data.get("validation_samples", "N/A")
        st.markdown(f"### 📊 Benchmark Performance")
        st.metric(label="Validation Accuracy", value=acc_pct)
        st.metric(label="Validation Samples", value=val_samples)
        st.divider()

    st.markdown("### 🧪 Quick Samples")
    st.write("Test with sample dataset images:")
    sample_cat_path = "data/cats/cat_000001.jpg"
    sample_dog_path = "data/dogs/dog_000001.jpg"

    use_sample_cat = False
    use_sample_dog = False

    if os.path.exists(sample_cat_path):
        if st.button("🐱 Load Sample Cat Image", use_container_width=True):
            use_sample_cat = True
    if os.path.exists(sample_dog_path):
        if st.button("🐶 Load Sample Dog Image", use_container_width=True):
            use_sample_dog = True


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🐾 Dog vs. Cat Image Classifier</h1>
    <p>Academic Deep Learning Project featuring a 3-Convolutional-Layer CNN with reproducible training & evaluation.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        f"⚠️ Model checkpoint `{MODEL_PATH}` not found! "
        "Please execute `python -m src.train` in terminal before launching the web app."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab_inference, tab_architecture, tab_evaluation = st.tabs([
    "🔍 Live Inference",
    "🧠 CNN Architecture",
    "📈 Evaluation Metrics & Visualizations"
])


# -----------------------------------------------------------------------------
# Tab 1: Live Inference
# -----------------------------------------------------------------------------
with tab_inference:
    col_upload, col_preview = st.columns([1, 1], gap="large")

    selected_image = None
    image_source_label = ""

    with col_upload:
        st.markdown("### 1. Upload or Select an Image")
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            help="Upload a cat or dog image to classify"
        )

        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_source_label = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"❌ Failed to read image: {e}")
        elif use_sample_cat and os.path.exists(sample_cat_path):
            selected_image = Image.open(sample_cat_path).convert("RGB")
            image_source_label = "Sample Image: Cat (cat_000001.jpg)"
        elif use_sample_dog and os.path.exists(sample_dog_path):
            selected_image = Image.open(sample_dog_path).convert("RGB")
            image_source_label = "Sample Image: Dog (dog_000001.jpg)"

        predict_button = st.button(
            "🚀 Classify Image",
            type="primary",
            use_container_width=True,
            disabled=(selected_image is None)
        )

    with col_preview:
        st.markdown("### 2. Image Preview & Results")
        if selected_image is not None:
            st.image(selected_image, caption=image_source_label, use_container_width=True)

            if predict_button:
                with st.spinner("Analyzing image across 3 Conv2D layers..."):
                    # Preprocess image
                    tensor, _ = preprocess_image(selected_image)

                    # Model forward pass
                    sigmoid_val = float(model.predict(tensor, verbose=0)[0][0])

                    if sigmoid_val >= 0.5:
                        predicted_class = "Dog"
                        confidence = sigmoid_val * 100.0
                        badge_class = "badge-dog"
                        icon = "🐶"
                    else:
                        predicted_class = "Cat"
                        confidence = (1.0 - sigmoid_val) * 100.0
                        badge_class = "badge-cat"
                        icon = "🐱"

                    cat_prob = (1.0 - sigmoid_val) * 100.0
                    dog_prob = sigmoid_val * 100.0

                    st.markdown(f"""
                    <div class="result-card">
                        <span class="metric-badge {badge_class}">{icon} PREDICTION: {predicted_class.upper()}</span>
                        <h2 style="margin: 12px 0 6px 0; color: #1E293B;">Confidence: {confidence:.2f}%</h2>
                        <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 16px;">
                            Sigmoid Output: <code>{sigmoid_val:.4f}</code> (Threshold: 0.50)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Class Probabilities:")
                    st.write(f"🐱 **Cat Probability:** {cat_prob:.2f}%")
                    st.progress(cat_prob / 100.0)

                    st.write(f"🐶 **Dog Probability:** {dog_prob:.2f}%")
                    st.progress(dog_prob / 100.0)
        else:
            st.info("👆 Please upload an image or choose a sample from the sidebar to begin.")


# -----------------------------------------------------------------------------
# Tab 2: CNN Architecture
# -----------------------------------------------------------------------------
with tab_architecture:
    st.markdown("### 🏛️ Exact 3-Convolutional-Layer CNN Architecture")
    st.write(
        "Built strictly with exactly three `Conv2D` layers, followed by `MaxPooling2D`, "
        "`Flatten`, `Dense(128)`, `Dropout(0.5)`, and `Dense(1, Sigmoid)`."
    )

    layers_info = explain_layers()
    for layer_name, description in layers_info.items():
        with st.expander(f"🔹 {layer_name}", expanded=False):
            st.write(description)

    st.markdown("#### Layer Architecture Summary Table")
    summary_data = [
        {"Stage": "Input", "Layer Type": "InputLayer", "Output Shape": "(None, 128, 128, 3)", "Parameters": 0},
        {"Stage": "Conv 1", "Layer Type": "Conv2D (32 filters, 3x3, ReLU)", "Output Shape": "(None, 128, 128, 32)", "Parameters": 896},
        {"Stage": "Pool 1", "Layer Type": "MaxPooling2D (2x2)", "Output Shape": "(None, 64, 64, 32)", "Parameters": 0},
        {"Stage": "Conv 2", "Layer Type": "Conv2D (64 filters, 3x3, ReLU)", "Output Shape": "(None, 64, 64, 64)", "Parameters": 18496},
        {"Stage": "Pool 2", "Layer Type": "MaxPooling2D (2x2)", "Output Shape": "(None, 32, 32, 64)", "Parameters": 0},
        {"Stage": "Conv 3", "Layer Type": "Conv2D (128 filters, 3x3, ReLU)", "Output Shape": "(None, 32, 32, 128)", "Parameters": 73856},
        {"Stage": "Pool 3", "Layer Type": "MaxPooling2D (2x2)", "Output Shape": "(None, 16, 16, 128)", "Parameters": 0},
        {"Stage": "Flatten", "Layer Type": "Flatten", "Output Shape": "(None, 32768)", "Parameters": 0},
        {"Stage": "Dense", "Layer Type": "Dense (128 units, ReLU)", "Output Shape": "(None, 128)", "Parameters": 4194432},
        {"Stage": "Dropout", "Layer Type": "Dropout (rate=0.5)", "Output Shape": "(None, 128)", "Parameters": 0},
        {"Stage": "Output", "Layer Type": "Dense (1 unit, Sigmoid)", "Output Shape": "(None, 1)", "Parameters": 129},
    ]
    st.table(summary_data)
    st.info("💡 **Total Trainable Parameters:** 4,287,809 (100% trainable, zero frozen/pretrained weights).")


# -----------------------------------------------------------------------------
# Tab 3: Evaluation Metrics & Visualizations
# -----------------------------------------------------------------------------
with tab_evaluation:
    st.markdown("### 📊 Evaluation Metrics & Visualizations")

    # Display saved charts
    col_acc, col_loss = st.columns(2)
    acc_img = "results/accuracy.png"
    loss_img = "results/loss.png"

    with col_acc:
        if os.path.exists(acc_img):
            st.image(acc_img, caption="Training vs. Validation Accuracy", use_container_width=True)
    with col_loss:
        if os.path.exists(loss_img):
            st.image(loss_img, caption="Training vs. Validation Loss", use_container_width=True)

    col_cm, col_samples = st.columns([1, 1.2])
    cm_img = "results/confusion_matrix.png"
    sample_img = "results/sample_predictions.png"

    with col_cm:
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix Heatmap", use_container_width=True)
    with col_samples:
        if os.path.exists(sample_img):
            st.image(sample_img, caption="Sample Validation Predictions", use_container_width=True)

    # Detailed metrics table
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        st.markdown("#### Quantitative Benchmark Summary")
        cm_data = m["confusion_matrix"]
        st.write(f"- **Validation Samples:** {m['validation_samples']}")
        st.write(f"- **Validation Accuracy:** **{m['overall_metrics']['accuracy_percent']}**")
        st.write(f"- **Precision (Dog):** {m['overall_metrics']['precision_dog']}")
        st.write(f"- **Recall (Dog):** {m['overall_metrics']['recall_dog']}")
        st.write(f"- **F1-Score (Dog):** {m['overall_metrics']['f1_score_dog']}")
        st.write(f"- **True Negatives (Cats Correct):** {cm_data['true_negatives_cats_correct']}")
        st.write(f"- **True Positives (Dogs Correct):** {cm_data['true_positives_dogs_correct']}")
        st.write(f"- **False Positives (Cats Misclassified as Dogs):** {cm_data['false_positives_cats_misclassified']}")
        st.write(f"- **False Negatives (Dogs Misclassified as Cats):** {cm_data['false_negatives_dogs_misclassified']}")
