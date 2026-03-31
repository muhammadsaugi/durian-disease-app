# 🌿 Sistem Deteksi Penyakit Daun Durian

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Accuracy](https://img.shields.io/badge/accuracy-96.23%25-brightgreen)](/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Aplikasi web untuk mendeteksi penyakit daun durian secara otomatis menggunakan
teknik Machine Learning berbasis **LBP + GLCM + Gabor + EfficientNet-B0 Fine-tuned + SVM**.

---

## 🎯 Kelas yang Dapat Dideteksi

| Kelas | Nama Indonesia | Keparahan |
|-------|---------------|-----------|
| `ALGAL_LEAF_SPOT` | Bercak Daun Alga | Sedang |
| `ALLOCARIDARA_ATTACK` | Serangan Penggerek Pucuk | Tinggi |
| `HEALTHY_LEAF` | Daun Sehat | Normal |
| `LEAF_BLIGHT` | Hawar Daun | Tinggi |
| `PHOMOPSIS_LEAF_SPOT` | Bercak Daun Phomopsis | Sedang |

---

## 📊 Performa Model

| Metrik | Nilai |
|--------|-------|
| Test Accuracy (Clean Pipeline) | **96.23%** |
| Test Accuracy (Full Pipeline) | **96.67%** |
| ROC-AUC (Macro OvR) | **0.9972** |
| Macro F1-Score | **0.9573** |
| F1 Min per Kelas | 0.9456 (ALGAL_LEAF_SPOT) |

---

## 🔬 Metode

### Feature Extraction
- **LBP Multi-Radius** (R=1,2,3) → 54 fitur tekstur lokal
- **GLCM** (3 jarak × 4 sudut × 6 properti) → 72 fitur tekstur global
- **Gabor Filter** (3 frekuensi × 4 orientasi) → 24 fitur tekstur frekuensi
- **Color Histogram HSV** (32 bins × 3 channel) → 96 fitur warna
- **EfficientNet-B0 Fine-tuned** → 1280 fitur semantik CNN
- **Total: 1526 fitur** → SelectKBest 500 fitur terpilih

### Classifier
- **SVM (Support Vector Machine)** dengan kernel RBF/Linear
- RobustScaler untuk normalisasi
- SMOTE untuk class balancing

---

## 🚀 Cara Deploy

### 1. Persiapan Model (di Google Colab)
```python
# Setelah notebook v3 selesai, jalankan export_model.py di Colab
# Ini akan mengekspor semua artifact ke Google Drive
exec(open('export_model.py').read())
```

### 2. Clone Repository
```bash
git clone https://github.com/USERNAME/durian-disease-app.git
cd durian-disease-app
```

### 3. Letakkan File Model
```
durian-disease-app/
└── models/
    ├── svm_model.pkl          ← dari export_model.py
    ├── scaler.pkl             ← dari export_model.py
    ├── selector.pkl           ← dari export_model.py
    └── metadata.json          ← dari export_model.py
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Jalankan Lokal
```bash
streamlit run app.py
```

### 6. Deploy ke Streamlit Cloud
1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app**
4. Pilih repository ini
5. Set **Main file**: `app.py`
6. Klik **Deploy**

> ⚠️ **Catatan model besar**: File `.pkl` dan `.onnx` besar mungkin perlu
> menggunakan [Git LFS](https://git-lfs.github.com/) atau
> [Streamlit secrets](https://docs.streamlit.io/library/advanced-features/secrets-management)
> untuk menyimpan di cloud storage.

---

## 📁 Struktur Proyek

```
durian-disease-app/
├── app.py                    # Aplikasi Streamlit utama
├── export_model.py           # Script export model dari Colab
├── requirements.txt          # Dependencies Python
├── README.md                 # Dokumentasi ini
├── .gitignore
├── .streamlit/
│   └── config.toml           # Konfigurasi tema Streamlit
└── models/                   # Folder model (isi manual)
    ├── svm_model.pkl
    ├── scaler.pkl
    ├── selector.pkl
    └── metadata.json
```

---

## 💡 Cara Penggunaan

1. **Upload gambar** daun durian (JPG/PNG/JPEG)
2. Sistem otomatis mengekstrak fitur dan melakukan prediksi
3. Lihat **hasil diagnosis** beserta tingkat keyakinan
4. Baca **detail penyakit** dan **rekomendasi penanganan**
5. Untuk banyak gambar sekaligus, upload multiple files

---

## 📚 Referensi

Penelitian ini merupakan bagian dari skripsi/jurnal:
> *"Komparasi Algoritma Machine Learning untuk Deteksi Penyakit Daun Durian
> Menggunakan Fitur LBP Multi-Radius, GLCM, Gabor, dan CNN Fine-tuned"*

---

## 👨‍💻 Author

Dibuat untuk keperluan penelitian dan akademik.  
Dataset: 5 kelas penyakit daun durian | Total: ~4515 gambar

---

## 📄 License

MIT License — bebas digunakan untuk keperluan akademik dan penelitian.
