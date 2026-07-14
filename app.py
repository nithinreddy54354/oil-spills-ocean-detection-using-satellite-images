"""
app.py — Oil Spill Detection · Streamlit Web UI
Run with:  streamlit run app.py
"""

import io
import time
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms, models

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Spill Detector",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — dark industrial theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}
.stApp {
    background: #0a0f1a;
    color: #c8d8e8;
}
h1, h2, h3, h4 {
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 0.05em;
}
.metric-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #112233 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    margin: 4px;
}
.metric-label {
    font-size: 12px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a8fbf;
}
.metric-value {
    font-size: 36px;
    font-weight: 700;
    font-family: 'Share Tech Mono', monospace;
    margin: 4px 0;
}
.spill-detected {
    color: #ff4444;
    text-shadow: 0 0 20px rgba(255,68,68,0.5);
}
.spill-clear {
    color: #44ff88;
    text-shadow: 0 0 20px rgba(68,255,136,0.5);
}
.result-banner {
    padding: 20px 30px;
    border-radius: 8px;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-align: center;
    margin: 12px 0;
}
.banner-spill {
    background: linear-gradient(90deg, rgba(255,44,44,0.15), rgba(200,0,0,0.25));
    border: 1px solid rgba(255,68,68,0.6);
    color: #ff6666;
}
.banner-clear {
    background: linear-gradient(90deg, rgba(44,255,120,0.10), rgba(0,180,80,0.20));
    border: 1px solid rgba(68,255,136,0.5);
    color: #55ff99;
}
.info-box {
    background: #0d1b2a;
    border-left: 3px solid #1e6fbf;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
    margin: 8px 0;
    color: #8ab4cc;
}
.stButton>button {
    background: linear-gradient(135deg, #1a3a5c, #0e2a45);
    border: 1px solid #2a6499;
    color: #7dc4ff;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 0.08em;
    font-size: 16px;
    padding: 10px 28px;
    border-radius: 4px;
    transition: all 0.2s;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #1e4a7a, #122e55);
    border-color: #3a84c9;
    box-shadow: 0 0 15px rgba(58,132,201,0.3);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODEL_PATH = "oil_spill_model.pth"
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# MODEL LOADER (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes    = checkpoint.get("classes", ["no_oil_spill", "oil_spill"])

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model, classes


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict(model, classes, image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    return pred_label, confidence, probs, classes


def make_confidence_chart(probs, classes):
    colors = []
    for c in classes:
        colors.append("#ff4444" if "oil_spill" in c and "no_" not in c else "#44ff88")

    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    bars = ax.barh(classes, probs * 100, color=colors, height=0.45,
                   edgecolor="none")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=11,
                color="#c8d8e8", fontweight="bold")

    ax.set_xlim(0, 115)
    ax.spines[:].set_visible(False)
    ax.tick_params(colors="#7a9abb", labelsize=11)
    ax.xaxis.set_visible(False)
    ax.set_title("Confidence Scores", color="#5a8fbf",
                 fontsize=12, pad=10, loc="left")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## By Nithin Reddy")
    st.markdown("## 🛢️ Oil Spill Detector - upload only ocean or water contained images .")
    st.markdown("---")
    st.markdown("""
<div class='info-box'>
Model: <strong>EfficientNet-B0</strong><br>
Task: Binary Classification<br>
Input: Satellite / Aerial Images
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Device")
    device_str = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔵 CPU"
    st.markdown(f"`{device_str}`")

    st.markdown("#### About")
    st.markdown("""
Upload a satellite or aerial image.  
The model will classify it as:
- 🔴 **Oil Spill Detected**
- 🟢 **No Spill (Clear)**
    """)

    st.markdown("---")
    st.markdown("<small style='color:#3a5a7a'>Built with PyTorch + Streamlit</small>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown("# 🛰️ Oil Spill Detection System")
st.markdown("<p style='color:#5a8fbf;letter-spacing:0.08em'>Satellite Image Analysis · Deep Learning Classification</p>",
            unsafe_allow_html=True)
st.markdown("---")

# Load model
with st.spinner("Loading model..."):
    try:
        model, classes = load_model()
        st.success(f"✅ Model loaded  |  Accuracy: {torch.load(MODEL_PATH, map_location='cpu').get('accuracy', 0):.2f}%")
    except FileNotFoundError:
        st.error(f"❌ Model file `{MODEL_PATH}` not found. Run `train_model.py` first.")
        st.stop()

# Upload
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Choose a satellite or aerial image",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded.name}",
                 use_column_width=True)

        st.markdown(f"""
<div class='metric-card'>
<div class='metric-label'>Image Size</div>
<div class='metric-value' style='font-size:20px'>{image.size[0]} × {image.size[1]} px</div>
</div>
""", unsafe_allow_html=True)

        run_btn = st.button("🔍  Analyze Image")

with col_result:
    st.markdown("### 📊 Analysis Results")

    if uploaded and run_btn:
        with st.spinner("Analyzing..."):
            time.sleep(0.3)  # brief pause for UX
            label, confidence, probs, classes = predict(model, classes, image)

        is_spill = "oil_spill" in label and "no_" not in label

        # Banner
        banner_cls  = "banner-spill" if is_spill else "banner-clear"
        banner_icon = "🔴 OIL SPILL DETECTED" if is_spill else "🟢 NO OIL SPILL — CLEAR"
        st.markdown(f"<div class='result-banner {banner_cls}'>{banner_icon}</div>",
                    unsafe_allow_html=True)

        # Metrics
        m1, m2 = st.columns(2)
        color_cls = "spill-detected" if is_spill else "spill-clear"
        with m1:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-label'>Classification</div>
<div class='metric-value {color_cls}' style='font-size:18px'>
{'OIL SPILL' if is_spill else 'CLEAR'}</div>
</div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-label'>Confidence</div>
<div class='metric-value {color_cls}'>{confidence:.1f}%</div>
</div>""", unsafe_allow_html=True)

        # Chart
        fig = make_confidence_chart(np.array(probs), classes)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Advisory
        st.markdown("#### Advisory")
        if is_spill:
            if confidence >= 90:
                st.error("⚠️ **HIGH CONFIDENCE** — Immediate response recommended. Alert relevant environmental authorities.")
            elif confidence >= 70:
                st.warning("⚠️ **MODERATE CONFIDENCE** — Likely spill. Consider aerial confirmation and response preparation.")
            else:
                st.warning("⚠️ **LOW CONFIDENCE** — Possible spill. Further inspection advised.")
        else:
            if confidence >= 90:
                st.success("✅ **Area appears clean.** No spill signatures detected.")
            else:
                st.info("ℹ️ **Likely clean**, but confidence is moderate. Consider re-analysis with higher resolution imagery.")

    elif not uploaded:
        st.markdown("""
<div style='text-align:center; padding:80px 20px; color:#2a4a6a; border:1px dashed #1a3a5a; border-radius:8px;'>
<div style='font-size:48px;'>🛰️</div>
<div style='font-size:18px; letter-spacing:0.1em; margin-top:12px;'>
Awaiting satellite image upload
</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#2a4a6a;font-size:13px;letter-spacing:0.08em'>"
    "OIL SPILL DETECTION SYSTEM · EfficientNet-B0 · PyTorch · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)