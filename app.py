"""
🌿 Durian Leaf Disease Detection System
Accuracy: 96.23% | Model: RandomForest/SVM + EfficientNet-B0 Fine-tuned + SelectKBest
"""

import streamlit as st
import numpy as np
import cv2
import pickle
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DurianAI — Durian Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --forest:  #0d2b1a; --emerald: #1a5c38; --jade: #2d8653;
    --pale:    #e8f7ee; --cream:   #fafdf7;
    --gold:    #d4a017; --coral:   #e05c3a;
    --text:    #0d1f14; --muted:   #4a6b56;
    --border:  #c5e0cc; --shadow:  0 4px 32px rgba(13,43,26,.12);
    --r:       14px;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--cream) !important;
    color: var(--text) !important;
}
.hero {
    background: linear-gradient(135deg, var(--forest) 0%, var(--emerald) 55%, var(--jade) 100%);
    border-radius: 20px; padding: 2.6rem 3rem 2.2rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
    content: '🌿'; position: absolute; font-size: 10rem;
    right: -1rem; top: -1.5rem; opacity: .07; line-height: 1;
}
.hero-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important; color: #fff !important;
    margin: 0 0 .35rem 0 !important; line-height: 1.15 !important; letter-spacing: -.5px;
}
.hero-sub { color: rgba(255,255,255,.78); font-size: .92rem; margin: 0 0 1.1rem 0; }
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.22);
    color: #fff; padding: 4px 13px; border-radius: 100px;
    font-size: .73rem; font-weight: 600; letter-spacing: .3px;
}
.metrics-row { display: flex; gap: 10px; margin-bottom: 1.4rem; flex-wrap: wrap; }
.metric-card {
    background: #fff; border: 1px solid var(--border);
    border-radius: var(--r); padding: .9rem 1.3rem;
    flex: 1; min-width: 115px; box-shadow: var(--shadow);
}
.metric-card .m-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem; color: var(--emerald); line-height: 1; margin-bottom: 2px;
}
.metric-card .m-lbl {
    font-size: .68rem; text-transform: uppercase;
    letter-spacing: 1px; color: var(--muted); font-weight: 700;
}
.metric-card .m-sub { font-size: .72rem; color: var(--muted); margin-top: 2px; }
.prob-wrap { margin: .6rem 0; }
.prob-row { display: flex; align-items: center; gap: 9px; margin: 5px 0; }
.prob-lbl { font-size: .78rem; font-weight: 600; width: 168px; flex-shrink: 0; color: var(--text); }
.prob-bg { flex: 1; height: 8px; background: var(--pale); border-radius: 4px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 4px; }
.prob-pct {
    font-size: .76rem; font-weight: 700; width: 42px; text-align: right;
    color: var(--muted); font-family: 'JetBrains Mono', monospace;
}
.disease-panel {
    background: #fff; border-radius: var(--r); padding: 1.4rem;
    box-shadow: var(--shadow); border-left: 5px solid var(--jade);
}
.disease-panel.moderate { border-left-color: var(--gold); }
.disease-panel.high     { border-left-color: var(--coral); }
.disease-panel h4 {
    font-family: 'DM Serif Display', serif; font-size: 1.05rem; margin: 0 0 .75rem 0;
}
.disease-panel p { font-size: .86rem; color: var(--muted); line-height: 1.75; margin: .35rem 0; }
.disease-panel strong { color: var(--text); }
.treat-item {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 5px 0; border-bottom: 1px solid #f0f4f1;
    font-size: .85rem; color: var(--text); line-height: 1.5;
}
.treat-ok { color: var(--jade); font-weight: 800; flex-shrink: 0; }
.sec-title {
    font-size: .68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 2px; color: var(--muted);
    margin: 1.8rem 0 .8rem 0; display: flex; align-items: center; gap: 10px;
}
.sec-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.guide-card {
    background: #fff; border: 1px solid var(--border);
    border-radius: var(--r); padding: 1.3rem;
    box-shadow: var(--shadow); text-align: center;
}
.guide-icon { font-size: 2rem; margin-bottom: .6rem; }
.guide-title { font-weight: 700; font-size: .92rem; margin-bottom: .35rem; color: var(--text); }
.guide-desc { font-size: .8rem; color: var(--muted); line-height: 1.6; }
[data-testid="stFileUploadDropzone"] {
    background: var(--pale) !important;
    border: 2px dashed var(--jade) !important;
    border-radius: var(--r) !important;
}
[data-testid="stSidebar"] { background: var(--forest) !important; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,.88) !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #7dffa8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12) !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
.stAlert { border-radius: var(--r) !important; }
.stDataFrame { border-radius: var(--r) !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Disease Database (English) ────────────────────────────────
DISEASE_DB = {
    "ALGAL_LEAF_SPOT": {
        "name"      : "Algal Leaf Spot",
        "pathogen"  : "Cephaleuros virescens",
        "severity"  : "Moderate", "sev_class": "moderate",
        "emoji"     : "🔴", "color": "#e07a3a",
        "symptoms"  : "Circular orange-greenish spots on the leaf surface. Spots may merge into larger damaged areas. Heavily infected leaves turn yellow and drop prematurely.",
        "cause"     : "Warm and humid conditions, common during rainy seasons. Plants with poor drainage are more susceptible.",
        "treatment" : [
            "Apply copper-based fungicide (copper hydroxide)",
            "Prune and destroy heavily infected leaves",
            "Improve air circulation around the plant",
            "Avoid over-irrigation and water splashing onto leaves",
        ],
        "prevention": "Maintain orchard sanitation, remove fallen leaves, ensure good drainage.",
    },
    "ALLOCARIDARA_ATTACK": {
        "name"      : "Allocaridara Attack",
        "pathogen"  : "Allocaridara malayensis",
        "severity"  : "High", "sev_class": "high",
        "emoji"     : "🐛", "color": "#c03a2a",
        "symptoms"  : "Damage on leaf edges and surfaces caused by larvae. Leaves appear holed, irregularly cut, or rolled. Young shoots are the primary target.",
        "cause"     : "Infestation by Allocaridara malayensis insects. More frequent during dry seasons or active growth phases.",
        "treatment" : [
            "Apply systemic insecticide (imidacloprid or chlorantraniliprole)",
            "Manually collect and destroy larvae",
            "Install pheromone traps around the orchard",
            "Utilize natural enemies such as parasitoids",
        ],
        "prevention": "Regularly monitor young shoots. Maintain clean orchard environment.",
    },
    "HEALTHY_LEAF": {
        "name"      : "Healthy Leaf",
        "pathogen"  : "No pathogen detected",
        "severity"  : "Normal", "sev_class": "normal",
        "emoji"     : "✅", "color": "#2d8653",
        "symptoms"  : "Uniform green color, glossy surface, no spots or abnormal discoloration. Leaf shape is perfect and consistent with durian morphology.",
        "cause"     : "Plant is in healthy and well-maintained condition.",
        "treatment" : [
            "Maintain current care routine",
            "Continue balanced fertilization (N-P-K)",
            "Conduct regular monitoring for early disease detection",
        ],
        "prevention": "Balanced fertilization, adequate irrigation, routine orchard sanitation.",
    },
    "LEAF_BLIGHT": {
        "name"      : "Leaf Blight",
        "pathogen"  : "Phytophthora palmivora / Rhizoctonia solani",
        "severity"  : "High", "sev_class": "high",
        "emoji"     : "🍂", "color": "#7a3010",
        "symptoms"  : "Large irregular brown spots that spread rapidly. Infected areas become dry and brittle. Severe infection can kill the entire leaf within a few days.",
        "cause"     : "Fungal/oomycete pathogens thriving in wet conditions. Spreads through rain splashing and leaf wounds.",
        "treatment" : [
            "Apply metalaxyl or mancozeb fungicide immediately",
            "Cut and burn infected parts as quickly as possible",
            "Reduce humidity by thinning the canopy",
            "Use systemic fungicide if infection has spread widely",
        ],
        "prevention": "Avoid mechanical damage. Apply preventive fungicide during rainy season.",
    },
    "PHOMOPSIS_LEAF_SPOT": {
        "name"      : "Phomopsis Leaf Spot",
        "pathogen"  : "Phomopsis durionis",
        "severity"  : "Moderate", "sev_class": "moderate",
        "emoji"     : "🟤", "color": "#7d5a3c",
        "symptoms"  : "Small to medium dark brown spots with yellow margins. Spots are round or oval. In humid conditions, black spore masses may appear at the center of spots.",
        "cause"     : "Phomopsis fungus entering through wounds or stomata. Develops in humid conditions at 20–30°C.",
        "treatment" : [
            "Apply difenoconazole or propiconazole fungicide",
            "Prune leaves showing severe symptoms",
            "Spray in the morning to allow leaves to dry before evening",
        ],
        "prevention": "Rotate fungicides to prevent resistance. Maintain good air circulation.",
    },
}

CLASS_NAMES = [
    "ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK", "HEALTHY_LEAF",
    "LEAF_BLIGHT", "PHOMOPSIS_LEAF_SPOT"
]
SEV_LABEL = {"normal": "Normal", "moderate": "Moderate", "high": "High"}

# ── Feature Extraction ────────────────────────────────────────
IMG_SIZE_HC = 128
LBP_CONFIGS = [{"radius":1,"n_points":8},{"radius":2,"n_points":16},{"radius":3,"n_points":24}]
GLCM_DIST   = [1, 3, 5]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS  = ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]
GABOR_FREQS = [0.1, 0.3, 0.5]
GABOR_THETA = [0, np.pi/4, np.pi/2, 3*np.pi/4]
HIST_BINS   = 32

def preprocess(img_bgr, size=IMG_SIZE_HC):
    img = cv2.resize(img_bgr, (size, size))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cl  = cv2.createCLAHE(2.0, (8, 8)); hsv[:,:,2] = cl.apply(hsv[:,:,2])
    enh = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blr = cv2.GaussianBlur(enh, (3,3), 1)
    return cv2.cvtColor(blr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

def extract_lbp(g):
    from skimage.feature import local_binary_pattern
    f = []
    for c in LBP_CONFIGS:
        lbp = local_binary_pattern(g, c["n_points"], c["radius"], method="uniform")
        nb  = c["n_points"] + 2
        h, _ = np.histogram(lbp.ravel(), bins=nb, range=(0,nb))
        f.extend(h.astype(float)/(h.sum()+1e-7))
    return np.array(f)

def extract_glcm(g):
    from skimage.feature import graycomatrix, graycoprops
    f  = []; iq = (g//4).astype(np.uint8)
    gl = graycomatrix(iq, distances=GLCM_DIST, angles=GLCM_ANGLES, levels=64, symmetric=True, normed=True)
    for p in GLCM_PROPS: f.extend(graycoprops(gl, p).flatten())
    return np.array(f)

def extract_gabor(g):
    from skimage.filters import gabor
    f = []; gf = g.astype(float)/255.0
    for freq in GABOR_FREQS:
        for theta in GABOR_THETA:
            r, i = gabor(gf, frequency=freq, theta=theta)
            m    = np.sqrt(r**2+i**2); f += [m.mean(), m.std()]
    return np.array(f)

def extract_color(h):
    f = []
    for ch, r in zip(range(3), [180,256,256]):
        hist = cv2.calcHist([h],[ch],None,[HIST_BINS],[0,r]).flatten()
        f.extend(hist/(hist.sum()+1e-7))
    return np.array(f)

def extract_handcrafted(img_bgr):
    g, h = preprocess(img_bgr)
    return np.concatenate([extract_lbp(g), extract_glcm(g), extract_gabor(g), extract_color(h)])

# ── Model Loading ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    arts = {"ok": False, "cnn_error": None}
    try:
        with open(os.path.join(model_dir, "svm_model.pkl"),  "rb") as f: arts["model"]    = pickle.load(f)
        with open(os.path.join(model_dir, "scaler.pkl"),     "rb") as f: arts["scaler"]   = pickle.load(f)
        with open(os.path.join(model_dir, "selector.pkl"),   "rb") as f: arts["selector"] = pickle.load(f)
        arts["model_type"] = type(arts["model"]).__name__
        arts["has_proba"]  = hasattr(arts["model"], "predict_proba")
        if hasattr(arts["scaler"], "n_features_in_"):
            arts["expected_features"] = int(arts["scaler"].n_features_in_)
        else:
            try:    arts["expected_features"] = int(arts["scaler"].center_.shape[0])
            except: arts["expected_features"] = 1526
        arts["n_hc"] = 246; arts["n_cnn"] = arts["expected_features"] - 246
        arts["ok"] = True
        pth = os.path.join(model_dir, "efficientnet_state_dict.pth")
        if arts["n_cnn"] > 0 and os.path.exists(pth):
            _load_cnn(arts, pth)
        elif arts["n_cnn"] > 0:
            arts["cnn_error"] = (
                f"❌ `efficientnet_state_dict.pth` not found in `models/` folder. "
                f"Scaler expects {arts['expected_features']} features "
                f"({arts['n_hc']} handcrafted + {arts['n_cnn']} CNN). "
                f"Please upload the CNN extractor file to GitHub.")
        else:
            arts["use_cnn"] = False
    except Exception as e:
        arts["error"] = str(e)
    return arts

def _load_cnn(arts, pth_path):
    try:
        import torch, torch.nn as nn
        import torchvision.models as models, torchvision.transforms as T
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
        first_key  = list(state_dict.keys())[0]
        base = models.efficientnet_b0(weights=None)
        if first_key[0].isdigit():
            seq = nn.Sequential(base.features, base.avgpool, nn.Flatten())
            seq.load_state_dict(state_dict, strict=True); arts["cnn"] = seq.eval()
        else:
            in_f = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(.4), nn.Linear(in_f,512), nn.ReLU(),
                nn.Dropout(.3), nn.Linear(512, len(CLASS_NAMES)))
            base.load_state_dict(state_dict, strict=True)
            arts["cnn"] = nn.Sequential(base.features, base.avgpool, nn.Flatten()).eval()
        arts["cnn_transform"] = T.Compose([
            T.Resize((224,224)), T.ToTensor(),
            T.Normalize([.485,.456,.406],[.229,.224,.225])])
        arts["use_cnn"] = True; arts["cnn_error"] = None
    except Exception as e:
        arts["use_cnn"] = False
        arts["cnn_error"] = f"⚠️ CNN failed to load: {str(e)[:200]}"

def predict(img_bgr, arts):
    hc    = extract_handcrafted(img_bgr).reshape(1,-1)
    n_exp = arts.get("expected_features", 1526)
    if arts.get("use_cnn") and arts.get("cnn") is not None:
        import torch
        pil  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        inp  = arts["cnn_transform"](pil).unsqueeze(0)
        with torch.no_grad(): cnn_f = arts["cnn"](inp).numpy()
        feats = np.concatenate([hc, cnn_f], axis=1)
    else:
        feats = hc
    actual = feats.shape[1]
    if actual != n_exp:
        if actual < n_exp: feats = np.concatenate([feats, np.zeros((1,n_exp-actual))], axis=1)
        else:              feats = feats[:,:n_exp]
    scaled   = arts["scaler"].transform(feats)
    selected = arts["selector"].transform(scaled)
    if arts["has_proba"]:
        proba = arts["model"].predict_proba(selected)[0]
    else:
        pred  = int(arts["model"].predict(selected)[0])
        proba = np.zeros(len(CLASS_NAMES)); proba[pred] = 1.0
    idx = int(np.argmax(proba))
    return CLASS_NAMES[idx], proba, idx

# ── Visualizations ────────────────────────────────────────────
def render_prob_bars(proba):
    s_idx = np.argsort(proba)[::-1]
    html  = '<div class="prob-wrap">'
    for rank, i in enumerate(s_idx):
        name  = DISEASE_DB[CLASS_NAMES[i]]["name"]
        color = DISEASE_DB[CLASS_NAMES[i]]["color"]
        pct   = proba[i]*100
        bold  = "font-weight:800;" if rank==0 else ""
        html += (f'<div class="prob-row">'
                 f'<span class="prob-lbl" style="{bold}">{name[:24]}</span>'
                 f'<div class="prob-bg"><div class="prob-fill" '
                 f'style="width:{pct:.1f}%;background:{color};"></div></div>'
                 f'<span class="prob-pct">{pct:.1f}%</span></div>')
    return html + "</div>"

def make_result_figure(img_bgr, proba, pred_cls, conf, elapsed_ms):
    """
    Single publication-ready figure:
    Col 0-1 : Input image (tall)
    Col 2-3 : Horizontal bar chart
    Col 4   : Preprocessing steps (stacked 2×2)
    All in ONE frame → clean single screenshot for journal.
    """
    names  = [DISEASE_DB[c]["name"]  for c in CLASS_NAMES]
    colors = [DISEASE_DB[c]["color"] for c in CLASS_NAMES]
    info   = DISEASE_DB[pred_cls]

    fig = plt.figure(figsize=(18, 5.5), facecolor="#fafdf7")
    gs  = fig.add_gridspec(2, 9, hspace=0.5, wspace=0.4,
                            left=0.03, right=0.97, top=0.88, bottom=0.10)

    # ── A: Input image ────────────────────────────────────────
    ax_img = fig.add_subplot(gs[:, 0:2])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax_img.imshow(img_rgb)
    ax_img.set_title(f"Input Image\n({img_rgb.shape[1]}×{img_rgb.shape[0]}px)",
                     fontsize=9, fontweight="bold", color="#0d1f14", pad=5)
    ax_img.axis("off")
    # small caption below image
    ax_img.text(0.5, -0.04,
                f"Prediction: {info['name']}  |  Confidence: {conf*100:.1f}%  |  {elapsed_ms:.0f} ms",
                transform=ax_img.transAxes, ha="center", va="top",
                fontsize=7.5, color="#2d8653", fontstyle="italic")

    # ── B: Horizontal bar chart ───────────────────────────────
    ax_bar = fig.add_subplot(gs[:, 2:5])
    s_idx = np.argsort(proba)
    bnames = [names[i]  for i in s_idx]
    bcolor = [colors[i] for i in s_idx]
    bvals  = [proba[i]*100 for i in s_idx]
    bars   = ax_bar.barh(bnames, bvals, color=bcolor,
                          edgecolor="white", linewidth=1.2, height=0.52)
    for bar, val in zip(bars, bvals):
        ax_bar.text(val + 0.6, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=9,
                    fontweight="bold", color="#0d1f14")
    ax_bar.set_xlim(0, 112)
    ax_bar.set_xlabel("Confidence (%)", fontsize=9, color="#4a6b56")
    ax_bar.set_title("Class Probability Distribution",
                     fontsize=10, fontweight="bold", color="#0d1f14", pad=6)
    ax_bar.spines[["top","right","left"]].set_visible(False)
    ax_bar.tick_params(axis="y", labelsize=9)
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.set_facecolor("#fafdf7")
    ax_bar.grid(axis="x", alpha=0.22, linestyle="--")

    # ── C: Preprocessing pipeline (2×2 grid) ─────────────────
    rsz  = cv2.resize(img_bgr, (128,128))
    hsv  = cv2.cvtColor(rsz, cv2.COLOR_BGR2HSV)
    cl   = cv2.createCLAHE(2.0,(8,8)); hsv[:,:,2]=cl.apply(hsv[:,:,2])
    enh  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blr  = cv2.GaussianBlur(enh,(3,3),1)
    gray = cv2.cvtColor(blr, cv2.COLOR_BGR2GRAY)

    prep = [
        (cv2.cvtColor(rsz, cv2.COLOR_BGR2RGB), "① Resize\n128×128",   None,   (0, 5)),
        (cv2.cvtColor(enh, cv2.COLOR_BGR2RGB), "② CLAHE\nEnhancement",None,   (0, 6)),
        (cv2.cvtColor(blr, cv2.COLOR_BGR2RGB), "③ Gaussian\nBlur",    None,   (1, 5)),
        (gray,                                  "④ Grayscale\n(LBP/GLCM)","gray",(1, 6)),
    ]
    # Also add HSV Color Hist step
    hsv_vis = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

    for im, title, cmap, (row, col) in prep:
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=7.5, fontweight="bold",
                     color="#0d1f14", pad=3, linespacing=1.3)
        ax.axis("off")

    # Feature bar (right-most)
    ax_feat = fig.add_subplot(gs[:, 7:9])
    feat_names  = ["LBP\nMulti-R", "GLCM\n3×4", "Gabor\n3×4", "Color\nHSV", "EfficientNet\nCNN"]
    feat_vals   = [54, 72, 24, 96, 1280]
    feat_colors = ["#2d8653","#52b788","#95d5b2","#d8f3dc","#1a5c38"]
    bar2 = ax_feat.barh(feat_names, feat_vals, color=feat_colors,
                         edgecolor="white", linewidth=1, height=0.55)
    for b, v in zip(bar2, feat_vals):
        ax_feat.text(v+5, b.get_y()+b.get_height()/2, f"{v}",
                     va="center", fontsize=8.5, fontweight="bold", color="#0d1f14")
    ax_feat.set_xlim(0, 1500)
    ax_feat.set_xlabel("No. of Features", fontsize=8, color="#4a6b56")
    ax_feat.set_title("Feature\nExtraction",
                      fontsize=9.5, fontweight="bold", color="#0d1f14", pad=4)
    ax_feat.spines[["top","right","left"]].set_visible(False)
    ax_feat.tick_params(axis="y", labelsize=8.5)
    ax_feat.tick_params(axis="x", labelsize=7.5)
    ax_feat.set_facecolor("#fafdf7")
    ax_feat.grid(axis="x", alpha=0.22, linestyle="--")

    # ── Super title ───────────────────────────────────────────
    fig.suptitle(
        f"DurianAI  ·  Durian Leaf Disease Detection  |  "
        f"Detected: {info['name']}  (Confidence: {conf*100:.1f}%)  |  "
        f"Total Features: 1,526  ·  Test Accuracy: 96.23%",
        fontsize=9.5, color="#1a5c38", fontweight="bold", y=0.97
    )

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="#fafdf7")
    plt.close(); buf.seek(0)
    return buf

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 DurianAI")
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    for val, lbl, sub in [
        ("96.23%",   "Accuracy",   "Clean Pipeline Validation"),
        ("0.9972",   "ROC-AUC",    "Macro One-vs-Rest"),
        ("0.9573",   "F1-Score",   "Macro Average"),
        ("5 Classes","Dataset",    "Disease & Healthy"),
    ]:
        st.markdown(
            f"""<div style="background:rgba(255,255,255,.08);border:1px solid
            rgba(255,255,255,.12);border-radius:10px;padding:10px 14px;margin:5px 0;">
            <div style="font-size:.68rem;opacity:.6;text-transform:uppercase;
            letter-spacing:1px;font-weight:600;">{lbl}</div>
            <div style="font-size:1.32rem;font-weight:800;color:#7dffa8;">{val}</div>
            <div style="font-size:.68rem;opacity:.55;">{sub}</div></div>""",
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔬 Feature Extraction")
    for m in ["LBP Multi-Radius (R=1,2,3) — 54",
              "GLCM (3 dist × 4 angles) — 72",
              "Gabor Filter (3×4) — 24",
              "Color Histogram HSV — 96",
              "EfficientNet-B0 Fine-tuned — 1280",
              "──────────────────",
              "Total: 1,526 → SelectKBest 500"]:
        st.markdown(f"<div style='font-size:.77rem;padding:2px 0;opacity:.82;'>• {m}</div>",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🌿 Disease Classes")
    for cls in CLASS_NAMES:
        d = DISEASE_DB[cls]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0;font-size:.79rem;'>"
            f"<span style='width:9px;height:9px;border-radius:50%;background:{d['color']};"
            f"flex-shrink:0;display:inline-block;'></span>{d['name']}</div>",
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:.68rem;opacity:.45;text-align:center;'>"
        "Durian Leaf Disease Detection<br>Research · 2024–2025</div>",
        unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🌿 DurianAI</div>
  <p class="hero-sub">Durian Leaf Disease Detection System — Upload a leaf image for automated ML-based diagnosis</p>
  <div class="hero-badges">
    <span class="badge">🎯 Accuracy 96.23%</span>
    <span class="badge">⚡ EfficientNet-B0 + SVM/RF</span>
    <span class="badge">🔬 5 Disease Categories</span>
    <span class="badge">📊 LBP + GLCM + Gabor + CNN</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── Metric Strip ──────────────────────────────────────────────
st.markdown('<div class="metrics-row">' + "".join([
    f'<div class="metric-card"><div class="m-val">{v}</div>'
    f'<div class="m-lbl">{l}</div><div class="m-sub">{s}</div></div>'
    for v,l,s in [
        ("96.23%", "Test Accuracy",  "Clean Pipeline Validation"),
        ("0.9972",  "ROC-AUC",        "Macro One-vs-Rest"),
        ("0.9573",  "Macro F1",       "Average all classes"),
        ("4/5",     "Bias Test",      "Diagnostic passed"),
        ("1,526",   "Total Features", "HC + CNN combined"),
    ]
]) + '</div>', unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
with st.spinner("⏳ Loading model..."):
    arts = load_artifacts()

if not arts["ok"]:
    st.error(f"❌ Failed to load model: `{arts.get('error','unknown')}`")
    st.info("Required files in `models/`: `svm_model.pkl`, `scaler.pkl`, `selector.pkl`, `efficientnet_state_dict.pth`")
    st.stop()

if arts.get("cnn_error"):
    st.error(arts["cnn_error"])
    st.info(f"""**How to fix:**
1. Ensure `efficientnet_state_dict.pth` exists in the `models/` folder
2. Download from Google Drive: `dataset_duren/streamlit_models/efficientnet_state_dict.pth`
3. Upload to GitHub repo under `models/`  (file size: ~15.6 MB)

> Scaler was trained with **{arts.get('expected_features',1526)} features** \
({arts.get('n_hc',246)} handcrafted + {arts.get('n_cnn',1280)} CNN).""")
    st.stop()

m_type = arts.get("model_type","Model")
cnn_ok = arts.get("use_cnn", False)
st.success(
    f"✅ Model ready: **{m_type}** + "
    f"{'EfficientNet-B0 Fine-tuned' if cnn_ok else 'Handcrafted Only'} "
    f"| {arts.get('expected_features',1526)} features total"
)

# ── Upload ────────────────────────────────────────────────────
st.markdown('<div class="sec-title">Upload Durian Leaf Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Select one or more images (JPG / PNG / JPEG)",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    help="Ensure the leaf is clearly visible with adequate lighting"
)

if not uploaded:
    st.markdown("""
    <div style="background:#e8f7ee;border:1.5px dashed #2d8653;border-radius:16px;
         padding:3rem;text-align:center;color:#2d6a4f;margin-top:1rem;">
      <div style="font-size:3.5rem;margin-bottom:.8rem;">🍃</div>
      <div style="font-size:1.1rem;font-weight:700;margin-bottom:.4rem;">No image uploaded yet</div>
      <div style="font-size:.88rem;opacity:.75;">
        Upload a durian leaf image above to start the analysis<br>
        Formats: JPG, PNG, JPEG — multiple files supported
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">How to Use</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📸", "1. Upload Image",
         "Take a clear photo of the durian leaf. Single or multiple images at once."),
        (c2, "⚡", "2. Automatic Analysis",
         "The system extracts 1,526 features (texture, color, CNN patterns) automatically."),
        (c3, "📊", "3. View Results",
         "Get full diagnosis with confidence score, probabilities, and treatment recommendations."),
    ]:
        col.markdown(f"""<div class="guide-card"><div class="guide-icon">{icon}</div>
            <div class="guide-title">{title}</div>
            <div class="guide-desc">{desc}</div></div>""", unsafe_allow_html=True)

else:
    results_summary = []

    for i, uf in enumerate(uploaded):
        st.markdown(f'<div class="sec-title">Analysis — {uf.name}</div>',
                    unsafe_allow_html=True)

        raw   = np.frombuffer(uf.read(), dtype=np.uint8)
        img_b = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_b is None:
            st.error(f"❌ Failed to read image: {uf.name}"); continue

        with st.spinner(f"Analyzing {uf.name}..."):
            t0 = time.time()
            pred_cls, proba, pred_idx = predict(img_b, arts)
            elapsed = time.time() - t0

        info      = DISEASE_DB[pred_cls]
        conf      = proba[pred_idx]
        sev_label = SEV_LABEL.get(info["sev_class"], "Normal")
        sev_color = {"Normal":"#2d8653","Moderate":"#b06000","High":"#b02020"}.get(sev_label,"#2d8653")

        # ── ROW 1: image | result card | prob bars ─────────────
        col_img, col_card, col_bars = st.columns([1.1, 1.25, 1.4])

        with col_img:
            img_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=uf.name, use_container_width=True)
            st.markdown(
                f"<div style='font-size:.72rem;color:#4a6b56;text-align:center;margin-top:3px;'>"
                f"{img_rgb.shape[1]}×{img_rgb.shape[0]}px · {elapsed*1000:.0f} ms</div>",
                unsafe_allow_html=True)

        with col_card:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0d2b1a,#2d8653);
                 border-radius:16px;padding:1.5rem 1.5rem;color:#fff;text-align:center;
                 box-shadow:0 6px 32px rgba(13,43,26,.25);">
              <div style="font-size:2.2rem;margin-bottom:.3rem;">{info["emoji"]}</div>
              <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:1.5px;
                   opacity:.7;font-weight:600;margin-bottom:.2rem;">Detection Result</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;
                   line-height:1.2;margin-bottom:.12rem;">{info["name"]}</div>
              <div style="font-size:.73rem;opacity:.62;margin-bottom:.9rem;font-style:italic;">
                {info["pathogen"]}</div>
              <div style="font-family:'DM Serif Display',serif;font-size:2.7rem;
                   color:#7dffa8;line-height:1;">{conf*100:.1f}%</div>
              <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:1px;
                   opacity:.7;margin-top:.15rem;">Confidence Score</div>
              <div style="margin-top:.75rem;">
                <span style="background:rgba(255,255,255,.14);border:1px solid
                     rgba(255,255,255,.24);padding:4px 13px;border-radius:100px;
                     font-size:.72rem;font-weight:700;">
                  Severity:
                  <span style="color:{sev_color};background:#fff;padding:1px 9px;
                       border-radius:100px;margin-left:4px;font-weight:800;">{sev_label}</span>
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

        with col_bars:
            st.markdown("**Class Probabilities:**")
            st.markdown(render_prob_bars(proba), unsafe_allow_html=True)

        # ── ROW 2: SINGLE FRAME FIGURE (journal-ready) ─────────
        st.markdown(
            '<div class="sec-title">Combined Result Figure — Screenshot Ready for Journal</div>',
            unsafe_allow_html=True)
        fig_buf = make_result_figure(img_b, proba, pred_cls, conf, elapsed*1000)
        st.image(fig_buf, use_container_width=True)
        st.caption(
            "📸 Single-frame figure combining: input image · class probability chart · "
            "preprocessing pipeline · feature extraction summary. "
            "Designed for direct screenshot inclusion in research publications."
        )

        # ── EXPANDER: Disease Detail ────────────────────────────
        with st.expander("📋 Disease Details & Treatment Recommendations", expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"""<div class="disease-panel {info['sev_class']}">
                  <h4>{info["emoji"]} {info["name"]}</h4>
                  <p><strong>Pathogen:</strong> <em>{info["pathogen"]}</em></p>
                  <p><strong>Symptoms:</strong><br>{info["symptoms"]}</p>
                  <p><strong>Cause of Infection:</strong><br>{info["cause"]}</p>
                  <p><strong>🛡️ Prevention:</strong><br>{info["prevention"]}</p>
                </div>""", unsafe_allow_html=True)
            with d2:
                items = "".join(
                    f'<div class="treat-item"><span class="treat-ok">✓</span>{t}</div>'
                    for t in info["treatment"])
                st.markdown(f"""<div class="disease-panel {info['sev_class']}">
                  <h4>💊 Treatment Steps</h4>{items}</div>""", unsafe_allow_html=True)

        # ── EXPANDER: Preprocessing ────────────────────────────
        with st.expander("🔬 Preprocessing Pipeline Visualization", expanded=False):
            rsz = cv2.resize(img_b,(128,128))
            hsv = cv2.cvtColor(rsz, cv2.COLOR_BGR2HSV)
            cl  = cv2.createCLAHE(2.0,(8,8)); hsv[:,:,2]=cl.apply(hsv[:,:,2])
            enh = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            blr = cv2.GaussianBlur(enh,(3,3),1)
            gray= cv2.cvtColor(blr, cv2.COLOR_BGR2GRAY)
            p1,p2,p3,p4 = st.columns(4)
            p1.image(cv2.cvtColor(rsz,cv2.COLOR_BGR2RGB), caption="① Resize 128×128",    use_container_width=True)
            p2.image(cv2.cvtColor(enh,cv2.COLOR_BGR2RGB), caption="② CLAHE Enhancement", use_container_width=True)
            p3.image(cv2.cvtColor(blr,cv2.COLOR_BGR2RGB), caption="③ Gaussian Blur",      use_container_width=True)
            p4.image(gray, caption="④ Grayscale (LBP/GLCM)", use_container_width=True, clamp=True)
            st.markdown(
                "<div style='font-size:.82rem;color:#4a6b56;line-height:1.8;margin-top:.5rem;'>"
                "<strong>Pipeline:</strong> Resize 128×128 → CLAHE (local contrast enhancement) "
                "→ Gaussian Blur (noise reduction) → Grayscale (for LBP, GLCM, Gabor) "
                "+ HSV (for Color Histogram)</div>", unsafe_allow_html=True)

        results_summary.append({
            "File"       : uf.name,
            "Prediction" : info["name"],
            "Confidence" : f"{conf*100:.1f}%",
            "Severity"   : sev_label,
            "Pathogen"   : info["pathogen"],
            "Time (ms)"  : f"{elapsed*1000:.0f}",
        })

        if i < len(uploaded)-1:
            st.markdown("---")

    # ── Summary Table ─────────────────────────────────────────
    if len(uploaded) > 1:
        st.markdown('<div class="sec-title">Summary — All Analyzed Images</div>',
                    unsafe_allow_html=True)
        import pandas as pd
        st.dataframe(pd.DataFrame(results_summary), use_container_width=True, hide_index=True)