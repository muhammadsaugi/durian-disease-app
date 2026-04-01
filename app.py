"""
🌿 Sistem Deteksi Penyakit Daun Durian
Akurasi: 96.23% | Model: RandomForest/SVM + EfficientNet Fine-tuned + SelectKBest
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
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DurianAI — Deteksi Penyakit Daun Durian",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --forest:   #0d2b1a;
    --emerald:  #1a5c38;
    --jade:     #2d8653;
    --mint:     #52c47a;
    --pale:     #e8f7ee;
    --cream:    #fafdf7;
    --gold:     #d4a017;
    --coral:    #e05c3a;
    --rust:     #8b3a1f;
    --text:     #0d1f14;
    --muted:    #4a6b56;
    --border:   #c5e0cc;
    --shadow:   0 4px 32px rgba(13,43,26,.12);
    --r:        14px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--cream) !important;
    color: var(--text) !important;
}

/* ── HEADER ── */
.hero {
    background: linear-gradient(135deg, var(--forest) 0%, var(--emerald) 55%, var(--jade) 100%);
    border-radius: 20px;
    padding: 2.8rem 3.2rem 2.4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🌿';
    position: absolute;
    font-size: 11rem;
    right: -1rem; top: -2rem;
    opacity: .07;
    line-height: 1;
}
.hero::after {
    content: '';
    position: absolute;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: rgba(255,255,255,.04);
    bottom: -120px; left: -60px;
}
.hero-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.5rem !important;
    color: #fff !important;
    margin: 0 0 .4rem 0 !important;
    line-height: 1.15 !important;
    letter-spacing: -.5px;
}
.hero-sub {
    color: rgba(255,255,255,.75);
    font-size: .95rem;
    margin: 0 0 1.2rem 0;
    font-weight: 400;
}
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.22);
    color: #fff;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .3px;
    backdrop-filter: blur(4px);
}

/* ── METRIC CARDS ── */
.metrics-row { display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1rem 1.4rem;
    flex: 1; min-width: 120px;
    box-shadow: var(--shadow);
    transition: transform .18s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .m-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: var(--emerald);
    line-height: 1;
    margin-bottom: 2px;
}
.metric-card .m-lbl {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    font-weight: 600;
}
.metric-card .m-sub { font-size: .75rem; color: var(--muted); margin-top: 2px; }

/* ── RESULT BOX ── */
.result-card {
    background: linear-gradient(135deg, var(--forest), var(--emerald));
    border-radius: 18px;
    padding: 2rem 2.2rem;
    color: #fff;
    text-align: center;
    box-shadow: 0 8px 40px rgba(13,43,26,.28);
    margin-bottom: 1rem;
}
.result-icon { font-size: 3rem; margin-bottom: .5rem; }
.result-tag {
    font-size: .72rem; text-transform: uppercase; letter-spacing: 1.5px;
    opacity: .7; margin-bottom: .3rem; font-weight: 600;
}
.result-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.55rem; line-height: 1.2; margin-bottom: .2rem;
}
.result-en { font-size: .82rem; opacity: .65; margin-bottom: 1.2rem; }
.result-conf {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem; color: #7dffa8; line-height: 1;
}
.result-conf-lbl { font-size: .78rem; opacity: .7; margin-top: .2rem; }
.sev-chip {
    display: inline-block;
    margin-top: 1rem; padding: 5px 16px;
    border-radius: 100px;
    font-size: .75rem; font-weight: 700;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.2);
}

/* ── PROB BARS ── */
.prob-wrap { margin: .8rem 0; }
.prob-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.prob-lbl { font-size: .8rem; font-weight: 600; width: 160px; flex-shrink: 0; color: var(--text); }
.prob-bg { flex: 1; height: 9px; background: var(--pale); border-radius: 5px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 5px; transition: width .5s ease; }
.prob-pct { font-size: .78rem; font-weight: 700; width: 44px; text-align: right;
             color: var(--muted); font-family: 'JetBrains Mono', monospace; }

/* ── DISEASE CARD ── */
.disease-panel {
    background: #fff;
    border-radius: var(--r);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    border-left: 5px solid var(--jade);
    margin-bottom: 1rem;
}
.disease-panel.warn { border-left-color: var(--gold); }
.disease-panel.danger { border-left-color: var(--coral); }
.disease-panel h4 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; margin: 0 0 .8rem 0;
}
.disease-panel p { font-size: .875rem; color: var(--muted); line-height: 1.75; margin: .35rem 0; }
.disease-panel strong { color: var(--text); }
.treat-item {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 6px 0; border-bottom: 1px solid #f0f4f1;
    font-size: .875rem; color: var(--text); line-height: 1.5;
}
.treat-icon { color: var(--jade); font-weight: 800; flex-shrink: 0; margin-top: 1px; }

/* ── SECTION TITLE ── */
.sec-title {
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 2px; color: var(--muted); margin: 2rem 0 .8rem 0;
    display: flex; align-items: center; gap: 10px;
}
.sec-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploadDropzone"] {
    background: var(--pale) !important;
    border: 2px dashed var(--jade) !important;
    border-radius: var(--r) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] { background: var(--forest) !important; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,.88) !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #7dffa8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12) !important; }

/* ── GUIDE CARDS ── */
.guide-card {
    background: #fff; border: 1px solid var(--border);
    border-radius: var(--r); padding: 1.4rem;
    box-shadow: var(--shadow); text-align: center;
    transition: transform .18s, box-shadow .18s;
}
.guide-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(13,43,26,.16); }
.guide-icon { font-size: 2.2rem; margin-bottom: .7rem; }
.guide-title { font-weight: 700; font-size: .95rem; margin-bottom: .4rem; color: var(--text); }
.guide-desc { font-size: .82rem; color: var(--muted); line-height: 1.6; }

/* ── TABLE ── */
.stDataFrame { border-radius: var(--r) !important; overflow: hidden; }

/* ── MISC ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
.stAlert { border-radius: var(--r) !important; }
</style>
""", unsafe_allow_html=True)

# ── Disease Database ──────────────────────────────────────────
DISEASE_DB = {
    "ALGAL_LEAF_SPOT": {
        "name_id"   : "Bercak Daun Alga",
        "name_en"   : "Algal Leaf Spot",
        "pathogen"  : "Cephaleuros virescens",
        "severity"  : "Sedang", "sev_class": "warn",
        "emoji"     : "🔴", "color": "#e07a3a",
        "symptoms"  : "Bercak bulat oranye-kehijauan pada permukaan daun. Bercak dapat bergabung membentuk area kerusakan luas. Daun yang terinfeksi berat menguning dan gugur lebih awal.",
        "cause"     : "Kondisi lembab dan hangat, sering terjadi di musim hujan. Tanaman dengan drainase buruk lebih rentan.",
        "treatment" : [
            "Semprot fungisida berbahan tembaga (copper hydroxide)",
            "Pangkas daun yang terinfeksi berat dan musnahkan",
            "Perbaiki sirkulasi udara di sekitar tanaman",
            "Hindari penyiraman berlebih dan percikan air ke daun",
        ],
        "prevention": "Jaga kebersihan kebun, buang daun gugur, pastikan drainase baik.",
    },
    "ALLOCARIDARA_ATTACK": {
        "name_id"   : "Serangan Penggerek Pucuk",
        "name_en"   : "Allocaridara Attack",
        "pathogen"  : "Allocaridara malayensis",
        "severity"  : "Tinggi", "sev_class": "danger",
        "emoji"     : "🐛", "color": "#c03a2a",
        "symptoms"  : "Kerusakan tepi dan permukaan daun akibat larva. Daun berlubang, terpotong tidak beraturan, atau menggulung. Pucuk muda menjadi target utama.",
        "cause"     : "Infestasi serangga Allocaridara malayensis. Lebih sering di musim kemarau atau saat tanaman dalam fase pertumbuhan aktif.",
        "treatment" : [
            "Insektisida sistemik (imidakloprid atau klorantraniliprol)",
            "Kumpulkan dan musnahkan larva secara manual",
            "Pasang perangkap feromon di sekitar kebun",
            "Manfaatkan musuh alami seperti parasitoid",
        ],
        "prevention": "Pantau pucuk muda secara rutin. Jaga kebersihan lingkungan kebun.",
    },
    "HEALTHY_LEAF": {
        "name_id"   : "Daun Sehat",
        "name_en"   : "Healthy Leaf",
        "pathogen"  : "Tidak ada patogen",
        "severity"  : "Normal", "sev_class": "",
        "emoji"     : "✅", "color": "#2d8653",
        "symptoms"  : "Daun hijau merata, permukaan mengkilap, tidak ada bercak atau perubahan warna abnormal. Bentuk daun sempurna sesuai morfologi tanaman durian.",
        "cause"     : "Tanaman dalam kondisi sehat dan terawat dengan baik.",
        "treatment" : [
            "Pertahankan rutinitas perawatan yang sudah dilakukan",
            "Lanjutkan pemupukan berimbang (N-P-K)",
            "Monitoring berkala untuk deteksi dini penyakit",
        ],
        "prevention": "Pupuk berimbang, irigasi cukup, sanitasi kebun rutin.",
    },
    "LEAF_BLIGHT": {
        "name_id"   : "Hawar Daun",
        "name_en"   : "Leaf Blight",
        "pathogen"  : "Phytophthora palmivora / Rhizoctonia solani",
        "severity"  : "Tinggi", "sev_class": "danger",
        "emoji"     : "🍂", "color": "#7a3010",
        "symptoms"  : "Bercak coklat besar tidak beraturan yang meluas cepat. Area terinfeksi menjadi kering dan rapuh. Infeksi berat dapat mematikan seluruh daun dalam beberapa hari.",
        "cause"     : "Jamur/oomycete yang berkembang pada kondisi basah. Menyebar melalui percikan air hujan dan luka pada daun.",
        "treatment" : [
            "Fungisida metalaksil atau mankozeb segera",
            "Potong dan bakar bagian terinfeksi secepatnya",
            "Kurangi kelembaban dengan pemangkasan kanopi",
            "Fungisida sistemik jika infeksi sudah meluas",
        ],
        "prevention": "Hindari luka mekanis. Aplikasi fungisida preventif di musim hujan.",
    },
    "PHOMOPSIS_LEAF_SPOT": {
        "name_id"   : "Bercak Daun Phomopsis",
        "name_en"   : "Phomopsis Leaf Spot",
        "pathogen"  : "Phomopsis durionis",
        "severity"  : "Sedang", "sev_class": "warn",
        "emoji"     : "🟤", "color": "#7d5a3c",
        "symptoms"  : "Bercak kecil-sedang berwarna coklat gelap dengan tepi kuning. Berbentuk bulat atau lonjong. Pada kondisi lembab terlihat massa spora hitam di tengah bercak.",
        "cause"     : "Jamur Phomopsis yang masuk melalui luka atau stomata. Berkembang pada kondisi lembab suhu 20–30°C.",
        "treatment" : [
            "Fungisida difenokonazol atau propikonazol",
            "Pangkas daun bergejala berat",
            "Semprot pagi hari agar daun kering sebelum sore",
        ],
        "prevention": "Rotasi fungisida untuk mencegah resistensi. Sirkulasi udara yang baik.",
    },
}

CLASS_NAMES = [
    "ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK", "HEALTHY_LEAF",
    "LEAF_BLIGHT", "PHOMOPSIS_LEAF_SPOT"
]
COLORS = [DISEASE_DB[c]["color"] for c in CLASS_NAMES]

# ── Feature Extraction Constants ──────────────────────────────
IMG_SIZE_HC  = 128
LBP_CONFIGS  = [{"radius":1,"n_points":8},{"radius":2,"n_points":16},{"radius":3,"n_points":24}]
GLCM_DIST    = [1, 3, 5]
GLCM_ANGLES  = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS   = ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]
GABOR_FREQS  = [0.1, 0.3, 0.5]
GABOR_THETA  = [0, np.pi/4, np.pi/2, 3*np.pi/4]
HIST_BINS    = 32

# ── Feature Extraction ────────────────────────────────────────
def preprocess(img_bgr, size=IMG_SIZE_HC):
    img  = cv2.resize(img_bgr, (size, size))
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cl   = cv2.createCLAHE(2.0, (8, 8))
    hsv[:,:,2] = cl.apply(hsv[:,:,2])
    enh  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blr  = cv2.GaussianBlur(enh, (3,3), 1)
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
    f  = []
    iq = (g//4).astype(np.uint8)
    gl = graycomatrix(iq, distances=GLCM_DIST, angles=GLCM_ANGLES,
                       levels=64, symmetric=True, normed=True)
    for p in GLCM_PROPS:
        f.extend(graycoprops(gl, p).flatten())
    return np.array(f)

def extract_gabor(g):
    from skimage.filters import gabor
    f  = []
    gf = g.astype(float)/255.0
    for freq in GABOR_FREQS:
        for theta in GABOR_THETA:
            r, i = gabor(gf, frequency=freq, theta=theta)
            m    = np.sqrt(r**2+i**2)
            f   += [m.mean(), m.std()]
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

        # ── Deteksi berapa fitur yang diharapkan scaler ───────
        # n_features_in_ = jumlah fitur saat scaler.fit()
        if hasattr(arts["scaler"], "n_features_in_"):
            arts["expected_features"] = int(arts["scaler"].n_features_in_)
        else:
            # Fallback: coba dari shape center_
            try:
                arts["expected_features"] = int(arts["scaler"].center_.shape[0])
            except Exception:
                arts["expected_features"] = 1526  # default v3

        arts["n_hc"]  = 246   # LBP(54)+GLCM(72)+Gabor(24)+Color(96)
        arts["n_cnn"] = arts["expected_features"] - arts["n_hc"]  # biasanya 1280

        arts["ok"] = True

        # ── Load CNN (wajib jika expected_features > 246) ─────
        pth = os.path.join(model_dir, "efficientnet_state_dict.pth")
        if arts["n_cnn"] > 0 and os.path.exists(pth):
            _load_cnn(arts, pth)
        elif arts["n_cnn"] > 0 and not os.path.exists(pth):
            arts["cnn_error"] = (
                f"❌ File `efficientnet_state_dict.pth` tidak ditemukan di folder `models/`. "
                f"Scaler mengharapkan {arts['expected_features']} fitur "
                f"({arts['n_hc']} HC + {arts['n_cnn']} CNN), "
                f"tapi CNN extractor tidak bisa dimuat."
            )
        else:
            # Scaler dilatih hanya dengan HC (246 fitur) — tidak butuh CNN
            arts["use_cnn"] = False

    except Exception as e:
        arts["error"] = str(e)
    return arts


def _load_cnn(arts, pth_path):
    try:
        import torch, torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as T

        state_dict = torch.load(pth_path, map_location="cpu",
                                 weights_only=False)
        first_key  = list(state_dict.keys())[0]
        base = models.efficientnet_b0(weights=None)

        if first_key[0].isdigit():
            # State dict dari nn.Sequential extractor langsung
            seq = nn.Sequential(base.features, base.avgpool, nn.Flatten())
            seq.load_state_dict(state_dict, strict=True)
            arts["cnn"] = seq.eval()
        else:
            # State dict dari model EfficientNet penuh
            in_f = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(.4), nn.Linear(in_f, 512), nn.ReLU(),
                nn.Dropout(.3), nn.Linear(512, len(CLASS_NAMES))
            )
            base.load_state_dict(state_dict, strict=True)
            arts["cnn"] = nn.Sequential(
                base.features, base.avgpool, nn.Flatten()
            ).eval()

        arts["cnn_transform"] = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        arts["use_cnn"]   = True
        arts["cnn_error"] = None

    except Exception as e:
        arts["use_cnn"]   = False
        arts["cnn_error"] = f"⚠️ CNN gagal dimuat: {str(e)[:200]}"


def predict(img_bgr, arts):
    """
    Pipeline prediksi:
    1. Ekstrak fitur handcrafted (selalu)
    2. Ekstrak fitur CNN jika use_cnn=True
    3. Gabungkan → pastikan dimensi sesuai scaler
    4. Scale → Select → Predict
    """
    hc    = extract_handcrafted(img_bgr).reshape(1, -1)  # (1, 246)
    n_exp = arts.get("expected_features", 1526)

    if arts.get("use_cnn") and arts.get("cnn") is not None:
        import torch
        pil   = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        inp   = arts["cnn_transform"](pil).unsqueeze(0)
        with torch.no_grad():
            cnn_f = arts["cnn"](inp).numpy()          # (1, 1280)
        feats = np.concatenate([hc, cnn_f], axis=1)  # (1, 1526)
    else:
        feats = hc                                    # (1, 246)

    # ── Dimensi check — jika masih tidak cocok, pad/trim ─────
    actual = feats.shape[1]
    if actual != n_exp:
        if actual < n_exp:
            # Pad dengan nol (rare case)
            pad   = np.zeros((1, n_exp - actual))
            feats = np.concatenate([feats, pad], axis=1)
        else:
            # Trim (rare case)
            feats = feats[:, :n_exp]

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
        name  = DISEASE_DB[CLASS_NAMES[i]]["name_id"]
        color = DISEASE_DB[CLASS_NAMES[i]]["color"]
        pct   = proba[i]*100
        bold  = "font-weight:800;" if rank==0 else ""
        html += f"""
        <div class="prob-row">
          <span class="prob-lbl" style="{bold}">{name[:22]}</span>
          <div class="prob-bg">
            <div class="prob-fill" style="width:{pct:.1f}%;background:{color};"></div>
          </div>
          <span class="prob-pct">{pct:.1f}%</span>
        </div>"""
    return html + "</div>"

def make_pie(proba):
    names  = [DISEASE_DB[c]["name_id"] for c in CLASS_NAMES]
    colors = [DISEASE_DB[c]["color"]   for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    wedges, _, autos = ax.pie(
        proba, colors=colors, autopct=lambda p: f"{p:.1f}%" if p>3 else "",
        startangle=130, pctdistance=.72,
        wedgeprops=dict(linewidth=2, edgecolor="white")
    )
    for a in autos: a.set_fontsize(8); a.set_color("white"); a.set_fontweight("bold")
    ax.legend(wedges,[n[:18] for n in names], loc="lower center",
              bbox_to_anchor=(.5,-.28), ncol=2, fontsize=7,
              framealpha=0, labelcolor="#0d2b1a")
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf,format="png",dpi=150,bbox_inches="tight",
                                  facecolor="none",transparent=True); plt.close(); buf.seek(0)
    return buf

def make_prep_steps(img_bgr):
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rsz  = cv2.resize(img_bgr, (128,128))
    hsv  = cv2.cvtColor(rsz, cv2.COLOR_BGR2HSV)
    cl   = cv2.createCLAHE(2.0,(8,8)); hsv[:,:,2]=cl.apply(hsv[:,:,2])
    enh  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blr  = cv2.GaussianBlur(enh,(3,3),1)
    gray = cv2.cvtColor(blr, cv2.COLOR_BGR2GRAY)
    steps = [(rgb,"① Original",None),
             (cv2.cvtColor(enh,cv2.COLOR_BGR2RGB),"② CLAHE",None),
             (cv2.cvtColor(blr,cv2.COLOR_BGR2RGB),"③ Gaussian Blur",None),
             (gray,"④ Grayscale (LBP)","gray")]
    fig, axes = plt.subplots(1,4,figsize=(12,3)); fig.patch.set_facecolor("#fafdf7")
    for ax,(im,title,cmap) in zip(axes,steps):
        ax.imshow(im,cmap=cmap); ax.set_title(title,fontsize=9,fontweight="bold",color="#0d2b1a"); ax.axis("off")
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf,format="png",dpi=140,bbox_inches="tight",facecolor="#fafdf7"); plt.close(); buf.seek(0)
    return buf

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 DurianAI")
    st.markdown("---")
    st.markdown("### 📊 Performa Model")
    metrics_side = [("96.23%","Akurasi","Clean Pipeline"),
                    ("0.9972","ROC-AUC","Macro OvR"),
                    ("0.9573","F1-Score","Macro Avg"),
                    ("5 Kelas","Dataset","Penyakit & Sehat")]
    for val, lbl, sub in metrics_side:
        st.markdown(f"""<div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);
            border-radius:10px;padding:10px 14px;margin:5px 0;">
            <div style="font-size:.7rem;opacity:.6;text-transform:uppercase;letter-spacing:1px;font-weight:600;">{lbl}</div>
            <div style="font-size:1.35rem;font-weight:800;color:#7dffa8;">{val}</div>
            <div style="font-size:.7rem;opacity:.55;">{sub}</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔬 Metode")
    for m in ["LBP Multi-Radius (R=1,2,3)","GLCM (3×4 matriks)","Gabor Filter (3 frek×4 arah)",
              "Color Histogram HSV","EfficientNet-B0 Fine-tuned"]:
        st.markdown(f"<div style='font-size:.8rem;padding:3px 0;opacity:.82;'>• {m}</div>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🌿 5 Kelas Deteksi")
    for cls in CLASS_NAMES:
        d = DISEASE_DB[cls]
        st.markdown(f"""<div style='display:flex;align-items:center;gap:8px;padding:4px 0;font-size:.8rem;'>
            <span style='width:10px;height:10px;border-radius:50%;background:{d["color"]};
            flex-shrink:0;display:inline-block;'></span>{d["name_id"]}</div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:.7rem;opacity:.45;text-align:center;'>Penelitian Deteksi Penyakit<br>Daun Durian · 2024–2025</div>",unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🌿 DurianAI</div>
  <p class="hero-sub">Sistem Deteksi Penyakit Daun Durian — Upload foto daun untuk diagnosis otomatis berbasis Machine Learning</p>
  <div class="hero-badges">
    <span class="badge">🎯 Akurasi 96.23%</span>
    <span class="badge">⚡ EfficientNet + SVM/RF</span>
    <span class="badge">🔬 5 Kategori Penyakit</span>
    <span class="badge">📊 LBP + GLCM + Gabor</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── Metric Strip ──────────────────────────────────────────────
st.markdown('<div class="metrics-row">' + "".join([
    f'<div class="metric-card"><div class="m-val">{v}</div><div class="m-lbl">{l}</div><div class="m-sub">{s}</div></div>'
    for v,l,s in [("96.23%","Test Accuracy","Clean Pipeline Valid."),
                   ("0.9972","ROC-AUC","Macro One-vs-Rest"),
                   ("0.9573","Macro F1","Rata-rata semua kelas"),
                   ("4/5","Bias Test","Lulus diagnostic"),
                   ("1526","Total Fitur","HC + CNN")]
]) + '</div>', unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
with st.spinner("⏳ Memuat model..."):
    arts = load_artifacts()

if not arts["ok"]:
    st.error(f"❌ Gagal memuat model: `{arts.get('error','unknown')}`")
    st.info("""**File yang dibutuhkan di folder `models/`:**
- `svm_model.pkl`
- `scaler.pkl`
- `selector.pkl`
- `efficientnet_state_dict.pth`""")
    st.stop()

# ── Tampilkan error CNN jika ada (STOP agar tidak crash saat prediksi) ──
if arts.get("cnn_error"):
    st.error(arts["cnn_error"])
    st.info(f"""**Cara mengatasi:**
1. Pastikan file `efficientnet_state_dict.pth` ada di folder `models/`
2. File ini harus di-download dari Google Drive: `dataset_duren/streamlit_models/efficientnet_state_dict.pth`
3. Upload ke GitHub repo di folder `models/`
4. Ukuran file: ~15.6 MB

> Scaler dilatih dengan **{arts.get('expected_features', 1526)} fitur** ({arts.get('n_hc',246)} HC + {arts.get('n_cnn',1280)} CNN).
> Tanpa CNN extractor, prediksi tidak bisa dijalankan.""")
    st.stop()

mode_badge = "🤖 " + arts.get("model_type","Model")
if arts.get("use_cnn"):
    mode_badge += " + EfficientNet-B0 Fine-tuned"
    n_f = arts.get("expected_features", 1526)
    st.success(f"✅ Model siap: **{mode_badge}** | {n_f} fitur total")
else:
    mode_badge += " + Handcrafted Only"
    st.success(f"✅ Model siap: **{mode_badge}**")

# ── Upload ────────────────────────────────────────────────────
st.markdown('<div class="sec-title">Upload Gambar Daun Durian</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Pilih satu atau beberapa gambar (JPG / PNG / JPEG)",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    help="Pastikan foto daun jelas, pencahayaan cukup, dan daun terlihat penuh"
)

if not uploaded:
    st.markdown("""
    <div style="background:#e8f7ee;border:1.5px dashed #2d8653;border-radius:16px;
         padding:3rem;text-align:center;color:#2d6a4f;margin-top:1rem;">
      <div style="font-size:3.5rem;margin-bottom:.8rem;">🍃</div>
      <div style="font-size:1.1rem;font-weight:700;margin-bottom:.4rem;">Belum ada gambar</div>
      <div style="font-size:.88rem;opacity:.75;">
        Upload foto daun durian di atas untuk memulai analisis<br>
        Format: JPG, PNG, JPEG — bisa multiple files sekaligus
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Cara Penggunaan</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1,"📸","1. Upload Foto","Foto daun durian dengan jelas. Satu atau banyak gambar sekaligus."),
        (c2,"⚡","2. Analisis Otomatis","Sistem mengekstrak 1526 fitur tekstur, warna, dan pola CNN secara otomatis."),
        (c3,"📊","3. Lihat Hasil","Diagnosis lengkap + probabilitas + rekomendasi penanganan spesifik."),
    ]:
        col.markdown(f"""<div class="guide-card"><div class="guide-icon">{icon}</div>
            <div class="guide-title">{title}</div>
            <div class="guide-desc">{desc}</div></div>""", unsafe_allow_html=True)

else:
    results_summary = []

    for i, uf in enumerate(uploaded):
        st.markdown(f'<div class="sec-title">Analisis — {uf.name}</div>', unsafe_allow_html=True)

        raw   = np.frombuffer(uf.read(), dtype=np.uint8)
        img_b = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        img_r = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        if img_b is None:
            st.error(f"❌ Gagal membaca gambar: {uf.name}"); continue

        with st.spinner(f"Menganalisis {uf.name}..."):
            t0 = time.time()
            pred_cls, proba, pred_idx = predict(img_b, arts)
            elapsed = time.time() - t0

        info = DISEASE_DB[pred_cls]
        conf = proba[pred_idx]

        # ── Layout ────────────────────────────────────────────
        col_img, col_res, col_pie = st.columns([1.1, 1.6, 1.0])

        with col_img:
            st.image(img_r, caption=uf.name, use_container_width=True)
            st.markdown(f"<div style='font-size:.74rem;color:#4a6b56;text-align:center;margin-top:4px;'>"
                        f"{img_r.shape[1]}×{img_r.shape[0]}px · {elapsed*1000:.0f}ms</div>",
                        unsafe_allow_html=True)

        with col_res:
            sev_color = {"Normal":"#2d8653","Sedang":"#c07a10","Tinggi":"#c03020"}.get(info["severity"],"#2d8653")
            st.markdown(f"""
            <div class="result-card">
              <div class="result-icon">{info["emoji"]}</div>
              <div class="result-tag">Hasil Deteksi</div>
              <div class="result-name">{info["name_id"]}</div>
              <div class="result-en">{info["name_en"]}</div>
              <div class="result-conf">{conf*100:.1f}%</div>
              <div class="result-conf-lbl">Tingkat Keyakinan</div>
              <div class="sev-chip">⚠️ Keparahan:
                <span style="color:{sev_color};background:#fff;padding:2px 10px;
                border-radius:100px;margin-left:6px;font-weight:800;">{info["severity"]}</span>
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("**Probabilitas semua kelas:**")
            st.markdown(render_prob_bars(proba), unsafe_allow_html=True)

        with col_pie:
            st.image(make_pie(proba), use_container_width=True)

        # ── Detail & Penanganan ───────────────────────────────
        with st.expander("📋 Detail Penyakit & Rekomendasi Penanganan", expanded=True):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"""<div class="disease-panel {info['sev_class']}">
                  <h4>{info["emoji"]} {info["name_id"]}</h4>
                  <p><strong>Patogen:</strong> {info["pathogen"]}</p>
                  <p><strong>Gejala:</strong><br>{info["symptoms"]}</p>
                  <p><strong>Penyebab Infeksi:</strong><br>{info["cause"]}</p>
                  <p><strong>🛡️ Pencegahan:</strong><br>{info["prevention"]}</p>
                </div>""", unsafe_allow_html=True)
            with d2:
                items = "".join(f'<div class="treat-item"><span class="treat-icon">✓</span>{t}</div>'
                                for t in info["treatment"])
                st.markdown(f"""<div class="disease-panel {info['sev_class']}">
                  <h4>💊 Langkah Penanganan</h4>{items}</div>""", unsafe_allow_html=True)

        # ── Preprocessing Viz ─────────────────────────────────
        with st.expander("🔬 Visualisasi Langkah Preprocessing", expanded=False):
            st.image(make_prep_steps(img_b), use_container_width=True)
            st.markdown("<div style='font-size:.8rem;color:#4a6b56;line-height:1.8;'>"
                        "<strong>Pipeline:</strong> Resize 128×128 → CLAHE (contrast enhancement) "
                        "→ Gaussian Blur (noise reduction) → Grayscale (input LBP/GLCM/Gabor) "
                        "+ HSV (Color Histogram)</div>", unsafe_allow_html=True)

        results_summary.append({
            "File": uf.name,
            "Prediksi": info["name_id"],
            "Keyakinan": f"{conf*100:.1f}%",
            "Keparahan": info["severity"],
            "Patogen": info["pathogen"]
        })

        if i < len(uploaded)-1:
            st.markdown("---")

    # ── Summary Table ─────────────────────────────────────────
    if len(uploaded) > 1:
        st.markdown('<div class="sec-title">Ringkasan Semua Gambar</div>', unsafe_allow_html=True)
        import pandas as pd
        st.dataframe(pd.DataFrame(results_summary), use_container_width=True, hide_index=True)
