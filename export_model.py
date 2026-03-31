"""
export_model.py
===============
Jalankan script ini di Google Colab SETELAH notebook v3 selesai.
Script ini akan mengekspor semua artifact model ke folder saved_models_v3/
yang sudah ada di Google Drive kamu.

Cara pakai:
    Di Colab, buat cell baru dan paste seluruh isi file ini, lalu Run.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

# ── Konfigurasi ───────────────────────────────────────────────
# Sesuaikan path ini dengan lokasi di Google Drive kamu
DATASET_PATH = '/content/drive/MyDrive/widuri/dataset_duren'
EXPORT_DIR   = os.path.join(DATASET_PATH, 'streamlit_models')
os.makedirs(EXPORT_DIR, exist_ok=True)

print('='*60)
print('EXPORT MODEL UNTUK STREAMLIT DEPLOYMENT')
print('='*60)

# ── 1. Export SVM Model ───────────────────────────────────────
# best_combo dan best_models berasal dari notebook v3
bsname, bmname = best_combo
best_svm = best_models[bsname][bmname]

with open(os.path.join(EXPORT_DIR, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(best_svm, f)
print(f'✅ SVM model tersimpan | params: {results_cv[bsname][bmname]["best_params"]}')

# ── 2. Export Scaler ──────────────────────────────────────────
with open(os.path.join(EXPORT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f'✅ RobustScaler tersimpan')

# ── 3. Export Feature Selector ────────────────────────────────
# Ambil selector dari skenario terbaik
if 'KBest500' in bsname or 'KBest800' in bsname or 'KBest' in bsname:
    # Ambil selector yang sesuai
    if '500' in bsname:
        selector_to_save = sel_500
    elif '800' in bsname:
        selector_to_save = sel_800
    else:
        selector_to_save = sel_500
elif 'PCA' in bsname:
    selector_to_save = pca
else:
    selector_to_save = None

if selector_to_save is not None:
    with open(os.path.join(EXPORT_DIR, 'selector.pkl'), 'wb') as f:
        pickle.dump(selector_to_save, f)
    print(f'✅ Feature selector tersimpan | type: {type(selector_to_save).__name__}')

# ── 4. Export EfficientNet sebagai ONNX (lebih portabel) ──────
print('\nMengekspor EfficientNet ke ONNX format...')
efficientnet.eval()

# Buat CNN extractor (tanpa classifier)
cnn_extractor_export = nn.Sequential(
    efficientnet.features,
    efficientnet.avgpool,
    nn.Flatten()
).cpu()
cnn_extractor_export.eval()

# Export ke ONNX
dummy_input = torch.zeros(1, 3, 224, 224)
onnx_path   = os.path.join(EXPORT_DIR, 'efficientnet_extractor.onnx')

try:
    torch.onnx.export(
        cnn_extractor_export,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    print(f'✅ EfficientNet ONNX tersimpan: {onnx_path}')
except Exception as e:
    print(f'⚠️ ONNX export gagal: {e}')
    # Fallback: simpan sebagai PyTorch state_dict
    torch.save(cnn_extractor_export.state_dict(),
               os.path.join(EXPORT_DIR, 'efficientnet_state_dict.pth'))
    print(f'✅ Fallback: state_dict tersimpan')

# ── 5. Export metadata ────────────────────────────────────────
metadata = {
    'model_name'    : bmname,
    'scenario'      : bsname,
    'test_accuracy' : float(best_acc_overall),
    'class_names'   : CLASS_NAMES,
    'best_params'   : {k: str(v) for k, v in
                       results_cv[bsname][bmname]['best_params'].items()},
    'n_features_in' : int(X_tv_sm.shape[1]),
    'n_features_sel': int(scenarios[bsname]['X_train'].shape[1]),
    'scaler_type'   : 'RobustScaler',
    'feature_types' : ['LBP_54', 'GLCM_72', 'Gabor_24', 'ColorHist_96',
                       'EfficientNet_1280'],
}

with open(os.path.join(EXPORT_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

import json
with open(os.path.join(EXPORT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'✅ Metadata tersimpan')

# ── 6. Verifikasi: Test load dan predict ──────────────────────
print('\n🔍 VERIFIKASI — Test load & predict...')

with open(os.path.join(EXPORT_DIR, 'svm_model.pkl'), 'rb') as f:
    test_model = pickle.load(f)
with open(os.path.join(EXPORT_DIR, 'scaler.pkl'), 'rb') as f:
    test_scaler = pickle.load(f)
with open(os.path.join(EXPORT_DIR, 'selector.pkl'), 'rb') as f:
    test_selector = pickle.load(f)

# Test dengan 5 sampel dari test set
X_test_hc_sample = X_test_hc[:5]
X_test_cnn_sample = X_test_cnn[:5]
X_test_combined   = np.concatenate([X_test_hc_sample, X_test_cnn_sample], axis=1)
X_test_scaled     = test_scaler.transform(X_test_combined)
X_test_selected   = test_selector.transform(X_test_scaled)
preds             = test_model.predict(X_test_selected)
probas            = test_model.predict_proba(X_test_selected)

print(f'Sample predictions: {[CLASS_NAMES[p] for p in preds]}')
print(f'Confidence range  : {probas.max(axis=1).min():.3f} – {probas.max(axis=1).max():.3f}')
print('✅ Verifikasi berhasil!')

# ── Summary ───────────────────────────────────────────────────
print('\n' + '='*60)
print('RINGKASAN EXPORT')
print('='*60)
print(f'Lokasi  : {EXPORT_DIR}')
for fname in os.listdir(EXPORT_DIR):
    fpath = os.path.join(EXPORT_DIR, fname)
    fsize = os.path.getsize(fpath) / 1024 / 1024
    print(f'  {fname:40s}: {fsize:.2f} MB')

print(f'\n📌 LANGKAH SELANJUTNYA:')
print(f'1. Download folder "{EXPORT_DIR}" dari Google Drive')
print(f'2. Rename folder menjadi "models"')
print(f'3. Letakkan di dalam folder proyek Streamlit')
print(f'4. Jalankan: streamlit run app.py')
