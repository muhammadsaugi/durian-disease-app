# Models Folder

Letakkan file model berikut di folder ini setelah menjalankan `export_model.py` di Google Colab:

| File | Ukuran Perkiraan | Keterangan |
|------|-----------------|------------|
| `svm_model.pkl` | ~50–200 MB | Model SVM terbaik |
| `scaler.pkl` | < 1 MB | RobustScaler |
| `selector.pkl` | < 1 MB | SelectKBest / PCA |
| `metadata.json` | < 1 KB | Info model |
| `efficientnet_extractor.onnx` | ~20 MB | CNN extractor (opsional) |

## Cara Mendapatkan File Model

1. Jalankan notebook `deteksi_penyakit_daun_durian_v3.ipynb` di Google Colab hingga selesai
2. Buat cell baru di Colab dan jalankan isi file `export_model.py`
3. Download folder `streamlit_models` dari Google Drive kamu
4. Salin semua file ke folder ini

## Catatan Deploy ke Streamlit Cloud

Jika file model terlalu besar untuk di-push ke GitHub (> 100 MB):
- Gunakan **Git LFS**: `git lfs track "*.pkl"`
- Atau upload ke **Google Drive** dan gunakan `gdown` untuk download otomatis saat startup
- Atau gunakan **Hugging Face Hub** untuk hosting model gratis
