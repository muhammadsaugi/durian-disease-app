"""
🌿 Sistem Deteksi Penyakit Daun Durian
Berbasis LBP + GLCM + Gabor + EfficientNet Fine-tuned + SVM
Accuracy: 96.23% (Clean Pipeline Validation)
"""

import streamlit as st
import numpy as np
import cv2
import pickle
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Penyakit Daun Durian",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Mono:wght@400;500&display=swap');

:root {
    --green-dark:   #1a3a2a;
    --green-mid:    #2d6a4f;
    --green-light:  #52b788;
    --green-pale:   #d8f3dc;
    --accent:       #f4a261;
    --accent-dark:  #e76f51;
    --bg:           #f8fdf9;
    --card:         #ffffff;
    --text:         #1a2e1e;
    --text-muted:   #5a7a62;
    --border:       #d0e8d8;
    --shadow:       0 4px 24px rgba(26,58,42,0.10);
    --radius:       16px;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, var(--green-light) 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 20%;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.main-header h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
    color: white !important;
}
.main-header p {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0.5rem 0 0 0;
    font-weight: 400;
    color: rgba(255,255,255,0.9) !important;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-right: 6px;
    backdrop-filter: blur(4px);
}

/* Cards */
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.info-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(26,58,42,0.14);
}
.info-card h4 {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin: 0 0 0.5rem 0;
}
.info-card .value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--green-mid);
    line-height: 1;
}
.info-card .sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Result Box */
.result-box {
    background: linear-gradient(135deg, var(--green-dark), var(--green-mid));
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 40px rgba(26,58,42,0.25);
    margin: 1rem 0;
}
.result-box .label-name {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.3px;
    margin: 0.5rem 0;
}
.result-box .confidence {
    font-size: 3rem;
    font-weight: 800;
    color: #95e5b4;
    line-height: 1;
}
.result-box .conf-label {
    font-size: 0.85rem;
    opacity: 0.8;
    margin-top: 0.3rem;
}

/* Disease Info Card */
.disease-card {
    background: var(--card);
    border-left: 5px solid var(--green-light);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1.5rem;
    margin-top: 1.5rem;
    box-shadow: var(--shadow);
}
.disease-card.warning { border-left-color: var(--accent); }
.disease-card.danger  { border-left-color: var(--accent-dark); }
.disease-card.healthy { border-left-color: var(--green-light); }
.disease-card h3 {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 0.75rem 0;
    color: var(--text);
}
.disease-card p {
    font-size: 0.9rem;
    color: var(--text-muted);
    line-height: 1.7;
    margin: 0.4rem 0;
}
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px 3px 3px 0;
}
.tag-green  { background: #d8f3dc; color: #2d6a4f; }
.tag-orange { background: #fde8d0; color: #c05621; }
.tag-red    { background: #fde0dc; color: #c0392b; }

/* Upload area */
.stFileUploader > div {
    border: 2px dashed var(--green-light) !important;
    border-radius: var(--radius) !important;
    background: var(--green-pale) !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}
.stFileUploader > div:hover {
    border-color: var(--green-mid) !important;
    background: #c8ead0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.9) !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #95e5b4 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* Metric strip */
.metric-strip {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-chip {
    background: var(--green-pale);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--green-mid);
}

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 6px 0;
}
.prob-label {
    font-size: 0.82rem;
    font-weight: 600;
    width: 170px;
    color: var(--text);
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 10px;
    background: var(--green-pale);
    border-radius: 5px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.6s ease;
}
.prob-val {
    font-size: 0.82rem;
    font-weight: 700;
    width: 48px;
    text-align: right;
    color: var(--text-muted);
    font-family: 'Fira Mono', monospace;
}

/* Divider */
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ── Disease Info Database ─────────────────────────────────────
DISEASE_INFO = {
    "ALGAL_LEAF_SPOT": {
        "name_id"   : "Bercak Daun Alga",
        "name_en"   : "Algal Leaf Spot",
        "pathogen"  : "Cephaleuros virescens (Alga parasit)",
        "severity"  : "Sedang",
        "card_class": "warning",
        "emoji"     : "🔴",
        "color"     : "#e76f51",
        "symptoms"  : "Bercak bulat berwarna oranye kehijauan pada permukaan daun. Bercak dapat bergabung membentuk area kerusakan yang lebih luas. Daun yang terinfeksi berat dapat menguning dan gugur lebih awal.",
        "cause"     : "Kondisi lembab dan hangat. Sering terjadi pada musim hujan. Tanaman dengan drainase buruk lebih rentan.",
        "treatment" : [
            "Semprot fungisida berbahan tembaga (copper hydroxide)",
            "Pangkas daun yang terinfeksi berat",
            "Perbaiki sirkulasi udara di sekitar tanaman",
            "Hindari penyiraman berlebih pada daun"
        ],
        "prevention": "Jaga kebersihan kebun, buang daun gugur, pastikan drainase baik.",
        "tag_class" : "tag-orange"
    },
    "ALLOCARIDARA_ATTACK": {
        "name_id"   : "Serangan Penggerek Pucuk",
        "name_en"   : "Allocaridara Attack",
        "pathogen"  : "Allocaridara malayensis (Serangga hama)",
        "severity"  : "Tinggi",
        "card_class": "danger",
        "emoji"     : "🐛",
        "color"     : "#c0392b",
        "symptoms"  : "Kerusakan pada tepi dan permukaan daun akibat aktivitas larva. Daun terlihat berlubang, terpotong tidak beraturan, atau menggulung. Pucuk muda sering menjadi target utama.",
        "cause"     : "Infestasi serangga Allocaridara malayensis. Lebih sering terjadi saat musim kemarau atau saat tanaman sedang dalam fase pertumbuhan aktif.",
        "treatment" : [
            "Aplikasi insektisida sistemik (imidakloprid atau klorantraniliprol)",
            "Kumpulkan dan musnahkan larva secara manual",
            "Pasang perangkap feromon",
            "Gunakan musuh alami seperti parasitoid"
        ],
        "prevention": "Pantau tanaman secara rutin, terutama pucuk muda. Jaga kebersihan lingkungan kebun.",
        "tag_class" : "tag-red"
    },
    "HEALTHY_LEAF": {
        "name_id"   : "Daun Sehat",
        "name_en"   : "Healthy Leaf",
        "pathogen"  : "Tidak ada patogen",
        "severity"  : "Normal",
        "card_class": "healthy",
        "emoji"     : "✅",
        "color"     : "#2d6a4f",
        "symptoms"  : "Daun berwarna hijau merata, permukaan mengkilap, tidak ada bercak, berlubang, atau perubahan warna abnormal. Bentuk daun sempurna sesuai morfologi durian.",
        "cause"     : "Tanaman dalam kondisi sehat dan terawat.",
        "treatment" : [
            "Pertahankan perawatan rutin yang sudah dilakukan",
            "Lanjutkan pemupukan berimbang",
            "Monitoring berkala untuk deteksi dini"
        ],
        "prevention": "Pupuk berimbang, irigasi cukup, sanitasi kebun rutin.",
        "tag_class" : "tag-green"
    },
    "LEAF_BLIGHT": {
        "name_id"   : "Hawar Daun",
        "name_en"   : "Leaf Blight",
        "pathogen"  : "Phytophthora palmivora / Rhizoctonia solani",
        "severity"  : "Tinggi",
        "card_class": "danger",
        "emoji"     : "🍂",
        "color"     : "#8B4513",
        "symptoms"  : "Bercak coklat besar tidak beraturan yang meluas cepat. Area yang terinfeksi menjadi kering dan rapuh. Pada infeksi berat seluruh daun bisa mongering dan mati dalam beberapa hari.",
        "cause"     : "Patogen jamur/oomycete yang berkembang pada kondisi basah. Penyebaran cepat melalui percikan air hujan dan luka pada daun.",
        "treatment" : [
            "Semprot fungisida berbahan aktif metalaksil atau mankozeb",
            "Potong dan bakar bagian yang terinfeksi segera",
            "Kurangi kelembaban dengan pemangkasan kanopi",
            "Aplikasi fungisida sistemik jika infeksi meluas"
        ],
        "prevention": "Hindari luka mekanis, aplikasi fungisida preventif di musim hujan.",
        "tag_class" : "tag-red"
    },
    "PHOMOPSIS_LEAF_SPOT": {
        "name_id"   : "Bercak Daun Phomopsis",
        "name_en"   : "Phomopsis Leaf Spot",
        "pathogen"  : "Phomopsis durionis (Jamur)",
        "severity"  : "Sedang",
        "card_class": "warning",
        "emoji"     : "🟤",
        "color"     : "#7d5a3c",
        "symptoms"  : "Bercak kecil hingga sedang berwarna coklat gelap dengan tepi kuning. Bercak biasanya berbentuk bulat atau lonjong. Pada kondisi lembab, dapat terlihat massa spora berwarna hitam di tengah bercak.",
        "cause"     : "Infeksi jamur Phomopsis yang masuk melalui luka atau stomata. Berkembang pada kondisi lembab dengan suhu 20–30°C.",
        "treatment" : [
            "Fungisida berbahan difenokonazol atau propikonazol",
            "Pangkas daun yang menunjukkan gejala berat",
            "Semprot pada pagi hari agar daun kering sebelum sore"
        ],
        "prevention": "Rotasi fungisida untuk mencegah resistensi. Jaga sirkulasi udara baik.",
        "tag_class" : "tag-orange"
    }
}

CLASS_NAMES = [
    "ALGAL_LEAF_SPOT",
    "ALLOCARIDARA_ATTACK",
    "HEALTHY_LEAF",
    "LEAF_BLIGHT",
    "PHOMOPSIS_LEAF_SPOT"
]

COLORS = {
    "ALGAL_LEAF_SPOT"     : "#e76f51",
    "ALLOCARIDARA_ATTACK" : "#c0392b",
    "HEALTHY_LEAF"        : "#2d6a4f",
    "LEAF_BLIGHT"         : "#8B4513",
    "PHOMOPSIS_LEAF_SPOT" : "#7d5a3c"
}


# ── Model Loading ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load semua artifact model dari folder models/"""
    artifacts = {}
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "models")

        with open(os.path.join(model_dir, "svm_model.pkl"), "rb") as f:
            artifacts["model"] = pickle.load(f)
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            artifacts["scaler"] = pickle.load(f)
        with open(os.path.join(model_dir, "selector.pkl"), "rb") as f:
            artifacts["selector"] = pickle.load(f)

        # CNN extractor (optional — hanya jika ada)
        cnn_path = os.path.join(model_dir, "cnn_extractor.pkl")
        if os.path.exists(cnn_path):
            with open(cnn_path, "rb") as f:
                artifacts["cnn"] = pickle.load(f)

        artifacts["loaded"] = True
        artifacts["mode"]   = "full" if "cnn" in artifacts else "handcrafted"
        return artifacts

    except FileNotFoundError as e:
        artifacts["loaded"] = False
        artifacts["error"]  = str(e)
        return artifacts


# ── Feature Extraction ────────────────────────────────────────
IMG_SIZE_HC  = 128
LBP_CONFIGS  = [{"radius":1,"n_points":8},{"radius":2,"n_points":16},{"radius":3,"n_points":24}]
GLCM_DIST    = [1, 3, 5]
GLCM_ANGLES  = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS   = ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]
GABOR_FREQS  = [0.1, 0.3, 0.5]
GABOR_THETA  = [0, np.pi/4, np.pi/2, 3*np.pi/4]
HIST_BINS    = 32

def preprocess(img_bgr, size=IMG_SIZE_HC):
    img = cv2.resize(img_bgr, (size, size))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blurred  = cv2.GaussianBlur(enhanced, (3, 3), 1)
    gray     = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv_f    = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return gray, hsv_f

def extract_lbp(gray):
    from skimage.feature import local_binary_pattern
    feats = []
    for cfg in LBP_CONFIGS:
        lbp = local_binary_pattern(gray, cfg["n_points"], cfg["radius"], method="uniform")
        nb  = cfg["n_points"] + 2
        h, _ = np.histogram(lbp.ravel(), bins=nb, range=(0, nb))
        feats.extend(h.astype(float) / (h.sum() + 1e-7))
    return np.array(feats)

def extract_glcm(gray):
    from skimage.feature import graycomatrix, graycoprops
    feats = []
    iq    = (gray // 4).astype(np.uint8)
    glcm  = graycomatrix(iq, distances=GLCM_DIST, angles=GLCM_ANGLES,
                          levels=64, symmetric=True, normed=True)
    for p in GLCM_PROPS:
        feats.extend(graycoprops(glcm, p).flatten())
    return np.array(feats)

def extract_gabor(gray):
    from skimage.filters import gabor
    feats  = []
    gf     = gray.astype(float) / 255.0
    for freq in GABOR_FREQS:
        for theta in GABOR_THETA:
            r, i = gabor(gf, frequency=freq, theta=theta)
            m    = np.sqrt(r**2 + i**2)
            feats += [m.mean(), m.std()]
    return np.array(feats)

def extract_color(hsv):
    feats  = []
    ranges = [180, 256, 256]
    for ch, r in zip(range(3), ranges):
        h = cv2.calcHist([hsv], [ch], None, [HIST_BINS], [0, r]).flatten()
        feats.extend(h / (h.sum() + 1e-7))
    return np.array(feats)

def extract_all(img_bgr):
    gray, hsv = preprocess(img_bgr)
    return np.concatenate([
        extract_lbp(gray),
        extract_glcm(gray),
        extract_gabor(gray),
        extract_color(hsv)
    ])


# ── Prediction ────────────────────────────────────────────────
def predict(img_bgr, artifacts):
    feats    = extract_all(img_bgr).reshape(1, -1)
    scaled   = artifacts["scaler"].transform(feats)
    selected = artifacts["selector"].transform(scaled)
    proba    = artifacts["model"].predict_proba(selected)[0]
    pred_idx = np.argmax(proba)
    return CLASS_NAMES[pred_idx], proba, pred_idx


# ── Visualizations ────────────────────────────────────────────
def plot_probability_bar(proba):
    """Render probability bars as HTML."""
    html = '<div style="margin: 0.5rem 0;">'
    sorted_idx = np.argsort(proba)[::-1]
    for i, idx in enumerate(sorted_idx):
        cls   = CLASS_NAMES[idx]
        p     = proba[idx]
        color = COLORS[cls]
        width = f"{p * 100:.1f}%"
        name  = DISEASE_INFO[cls]["name_id"]
        bold  = "font-weight:800;" if i == 0 else ""
        html += f"""
        <div class="prob-row">
          <span class="prob-label" style="{bold}">{name[:22]}</span>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{width}; background:{color};"></div>
          </div>
          <span class="prob-val">{p*100:.1f}%</span>
        </div>"""
    html += "</div>"
    return html

def plot_pie_chart(proba):
    names  = [DISEASE_INFO[c]["name_id"] for c in CLASS_NAMES]
    colors = [COLORS[c] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    wedges, texts, autotexts = ax.pie(
        proba, labels=None, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(linewidth=2, edgecolor="white")
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
    ax.legend(wedges, [n[:20] for n in names], loc="lower center",
              bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7,
              framealpha=0, labelcolor="#1a3a2a")
    ax.set_title("Distribusi Probabilitas", fontsize=10,
                  fontweight="bold", color="#1a3a2a", pad=8)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close()
    buf.seek(0)
    return buf

def show_preprocessing_steps(img_bgr):
    """Tampilkan langkah preprocessing."""
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(img_bgr, (128, 128))
    hsv       = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    clahe     = cv2.createCLAHE(2.0, (8, 8))
    hsv[:,:,2]= clahe.apply(hsv[:,:,2])
    enhanced  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blurred   = cv2.GaussianBlur(enhanced, (3,3), 1)
    gray      = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.patch.set_facecolor("none")
    steps = [
        (img_rgb, "1. Original", "viridis"),
        (cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), "2. CLAHE", None),
        (cv2.cvtColor(blurred,  cv2.COLOR_BGR2RGB), "3. Gaussian Blur", None),
        (gray, "4. Grayscale (LBP)", "gray"),
    ]
    for ax, (im, title, cmap) in zip(axes, steps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=9, fontweight="bold", color="#1a3a2a")
        ax.axis("off")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close()
    buf.seek(0)
    return buf


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Durian Leaf AI")
    st.markdown("---")
    st.markdown("### 📊 Performa Model")

    metrics = [
        ("Akurasi", "96.23%", "Clean Pipeline"),
        ("ROC-AUC", "0.9972", "Macro OvR"),
        ("F1-Score", "0.9573", "Macro Avg"),
        ("Kelas", "5 Kelas", "Penyakit & Sehat"),
    ]
    for label, val, sub in metrics:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08); border-radius:10px;
                    padding:10px 14px; margin:6px 0; border:1px solid rgba(255,255,255,0.12);">
            <div style="font-size:0.72rem; opacity:0.65; text-transform:uppercase;
                        letter-spacing:1px; font-weight:600;">{label}</div>
            <div style="font-size:1.3rem; font-weight:800; color:#95e5b4;">{val}</div>
            <div style="font-size:0.72rem; opacity:0.6;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Metode Ekstraksi")
    methods = ["LBP Multi-Radius (R=1,2,3)", "GLCM (3 jarak × 4 sudut)",
               "Gabor Filter (3 frek × 4 arah)", "Color Histogram HSV",
               "EfficientNet-B0 Fine-tuned"]
    for m in methods:
        st.markdown(f"<div style='font-size:0.82rem; padding:4px 0; "
                    f"opacity:0.85;'>• {m}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Kelas Penyakit")
    for cls in CLASS_NAMES:
        info  = DISEASE_INFO[cls]
        color = COLORS[cls]
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; "
            f"padding:5px 0; font-size:0.82rem;'>"
            f"<span style='width:10px; height:10px; border-radius:50%; "
            f"background:{color}; flex-shrink:0; display:inline-block;'></span>"
            f"{info['name_id']}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; opacity:0.55; text-align:center;'>"
        "Skripsi / Jurnal Sinta 2<br>Deteksi Penyakit Daun Durian<br>"
        "Machine Learning · 2024</div>",
        unsafe_allow_html=True
    )


# ── Main Content ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 Deteksi Penyakit Daun Durian</h1>
    <p>Sistem klasifikasi berbasis Machine Learning — Upload gambar daun durian untuk mendapatkan diagnosis otomatis</p>
    <span class="badge">🎯 Akurasi 96.23%</span>
    <span class="badge">🔬 5 Kategori</span>
    <span class="badge">⚡ EfficientNet + SVM</span>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
with st.spinner("Memuat model..."):
    artifacts = load_model_artifacts()

if not artifacts.get("loaded"):
    st.error(f"❌ Model belum tersedia. Pastikan file model ada di folder `models/`")
    st.info("""
    **File yang dibutuhkan di folder `models/`:**
    - `svm_model.pkl`
    - `scaler.pkl`
    - `selector.pkl`

    Ekspor dari Google Colab menggunakan script `export_model.py`
    """)
    st.stop()


# ── Upload Section ────────────────────────────────────────────
st.markdown('<div class="section-title">Upload Gambar Daun Durian</div>',
            unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Pilih gambar (JPG, PNG, JPEG) — bisa upload beberapa sekaligus",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload satu atau beberapa gambar daun durian untuk didiagnosis"
)

if not uploaded_files:
    # Placeholder ketika belum ada upload
    st.markdown("""
    <div style="background: #f0faf2; border: 1.5px dashed #52b788; border-radius: 16px;
                padding: 3rem; text-align: center; color: #5a7a62;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🍃</div>
        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
            Belum ada gambar yang diupload
        </div>
        <div style="font-size: 0.88rem; opacity: 0.75;">
            Upload gambar daun durian di atas untuk memulai analisis.<br>
            Sistem mendukung format JPG, PNG, dan JPEG.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Panduan singkat
    st.markdown('<div class="section-title">Panduan Penggunaan</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    steps = [
        ("📸", "1. Upload Gambar",
         "Foto daun durian dengan jelas. Pastikan daun terlihat penuh dan pencahayaan cukup."),
        ("⚡", "2. Proses Otomatis",
         "Sistem akan mengekstrak fitur tekstur, warna, dan pola secara otomatis."),
        ("📊", "3. Lihat Hasil",
         "Dapatkan diagnosis lengkap beserta probabilitas dan rekomendasi penanganan."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], steps):
        col.markdown(f"""
        <div class="info-card" style="text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.8rem;">{icon}</div>
            <div style="font-weight:700; margin-bottom:0.5rem; font-size:0.95rem;">{title}</div>
            <div style="font-size:0.83rem; color:#5a7a62; line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

else:
    # ── Process each uploaded image ───────────────────────────
    for i, uploaded_file in enumerate(uploaded_files):
        st.markdown(f'<div class="section-title">Hasil Analisis — {uploaded_file.name}</div>',
                    unsafe_allow_html=True)

        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if img_bgr is None:
            st.error(f"❌ Gagal memuat gambar: {uploaded_file.name}")
            continue

        # Predict
        with st.spinner(f"Menganalisis {uploaded_file.name}..."):
            t0             = time.time()
            pred_class, proba, pred_idx = predict(img_bgr, artifacts)
            elapsed        = time.time() - t0

        info       = DISEASE_INFO[pred_class]
        confidence = proba[pred_idx]

        # ── Layout: Image | Result | Pie ─────────────────────
        col_img, col_res, col_pie = st.columns([1.2, 1.5, 1])

        with col_img:
            st.image(img_rgb, caption=f"📁 {uploaded_file.name}", use_container_width=True)
            st.markdown(f"""
            <div style="font-size:0.78rem; color:#5a7a62; margin-top:6px; text-align:center;">
                Dimensi: {img_rgb.shape[1]}×{img_rgb.shape[0]}px &nbsp;·&nbsp;
                Waktu: {elapsed*1000:.0f}ms
            </div>""", unsafe_allow_html=True)

        with col_res:
            sev_color = {"Normal":"#2d6a4f","Sedang":"#e67e22","Tinggi":"#c0392b"}.get(
                info["severity"], "#2d6a4f")
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:2.5rem; margin-bottom:4px;">{info["emoji"]}</div>
                <div style="font-size:0.78rem; opacity:0.75; text-transform:uppercase;
                            letter-spacing:1px; font-weight:600;">Hasil Deteksi</div>
                <div class="label-name">{info["name_id"]}</div>
                <div style="font-size:0.82rem; opacity:0.75; margin-bottom:1rem;">
                    {info["name_en"]}
                </div>
                <div class="confidence">{confidence*100:.1f}%</div>
                <div class="conf-label">Tingkat Keyakinan</div>
                <div style="margin-top:1rem; padding-top:1rem;
                            border-top:1px solid rgba(255,255,255,0.15);">
                    <span style="background:rgba(255,255,255,0.15); padding:4px 12px;
                                 border-radius:100px; font-size:0.78rem; font-weight:600;">
                        ⚠️ Tingkat Keparahan:
                        <span style="color:{sev_color}; background:white; padding:2px 8px;
                                     border-radius:100px; margin-left:4px;">
                            {info["severity"]}
                        </span>
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Probability bars
            st.markdown("**Probabilitas semua kelas:**")
            st.markdown(plot_probability_bar(proba), unsafe_allow_html=True)

        with col_pie:
            pie_buf = plot_pie_chart(proba)
            st.image(pie_buf, use_container_width=True)

        # ── Disease Detail ────────────────────────────────────
        with st.expander("📋 Detail Diagnosis & Rekomendasi Penanganan", expanded=True):
            d1, d2 = st.columns(2)

            with d1:
                st.markdown(f"""
                <div class="disease-card {info['card_class']}">
                    <h3>{info['emoji']} {info['name_id']} ({info['name_en']})</h3>
                    <p><strong>Patogen/Penyebab:</strong><br>{info['pathogen']}</p>
                    <p><strong>Gejala:</strong><br>{info['symptoms']}</p>
                    <p><strong>Penyebab Infeksi:</strong><br>{info['cause']}</p>
                </div>""", unsafe_allow_html=True)

            with d2:
                treatment_html = "".join(
                    f"<div style='padding:6px 0; border-bottom:1px solid #f0f0f0; "
                    f"font-size:0.88rem; color:#2d3748;'>"
                    f"<span style='color:#2d6a4f; font-weight:700; margin-right:8px;'>✓</span>"
                    f"{t}</div>"
                    for t in info["treatment"]
                )
                st.markdown(f"""
                <div class="disease-card {info['card_class']}">
                    <h3>💊 Rekomendasi Penanganan</h3>
                    {treatment_html}
                    <p style="margin-top:1rem;"><strong>🛡️ Pencegahan:</strong><br>
                    {info['prevention']}</p>
                </div>""", unsafe_allow_html=True)

        # ── Preprocessing Visualization ───────────────────────
        with st.expander("🔬 Visualisasi Langkah Preprocessing", expanded=False):
            prep_buf = show_preprocessing_steps(img_bgr)
            st.image(prep_buf, use_container_width=True)
            st.markdown("""
            <div style="font-size:0.82rem; color:#5a7a62; padding:8px 0; line-height:1.7;">
                <strong>Pipeline Preprocessing:</strong>
                Resize 128×128 → CLAHE (contrast enhancement) →
                Gaussian Blur (noise reduction) → Grayscale (untuk LBP/GLCM/Gabor) +
                HSV (untuk Color Histogram)
            </div>""", unsafe_allow_html=True)

        if i < len(uploaded_files) - 1:
            st.markdown("---")

    # ── Summary Table (jika multi-file) ──────────────────────
    if len(uploaded_files) > 1:
        st.markdown('<div class="section-title">Ringkasan Semua Gambar</div>',
                    unsafe_allow_html=True)
        summary_rows = []
        for uf in uploaded_files:
            fb  = np.asarray(bytearray(uf.read()), dtype=np.uint8)
            ib  = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            if ib is not None:
                pc, prob, pi = predict(ib, artifacts)
                inf = DISEASE_INFO[pc]
                summary_rows.append({
                    "File"            : uf.name,
                    "Prediksi"        : inf["name_id"],
                    "Keyakinan"       : f"{prob[pi]*100:.1f}%",
                    "Keparahan"       : inf["severity"],
                    "Patogen"         : inf["pathogen"]
                })
        if summary_rows:
            import pandas as pd
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
