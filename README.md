# Aplikasi Klasifikasi Gaji

Aplikasi Streamlit untuk memprediksi kategori gaji pekerja (`<=7jt` / `>7jt`) menggunakan model **Random Forest dengan Pendekatan Class Weighting**.

## Struktur Proyek

```
Aplikasi-Klasifikasi-Gaji-PM/
├── Dataset/
│   ├── DatasetPekerja.csv
│   └── DatasetPekerja_Processed.csv
├── ipynb/
│   └── Notebook.ipynb
├── model/
│   ├── rf_baseline.pkl
│   ├── rf_weighted.pkl
│   ├── rf_tuned_main.pkl
│   ├── rf_balanced_model.pkl
│   ├── logreg_pembanding.pkl
│   ├── xgb_pembanding.pkl
│   ├── preprocessor.pkl
│   ├── metrics.pkl
│   ├── shap_artifact.pkl
│   └── shap_summary.png
├── reports/
│   ├── final_report.pdf
│   └── presentation.pptx
├── frontend/
│   └── app.py
├── api/
│   ├── main.py
│   ├── inference.py
│   └── schemas.py
├── app.py                (Main Streamlit dashboard)
├── train_model.py        (Training script for 5 models)
├── requirements.txt
└── README.md
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

Jalankan perintah berikut dari folder root proyek.

### 1. Training model (opsional, jika model belum ada)

```bash
python train_model.py
```

Perintah ini akan membuat:
- 5 model: `rf_baseline.pkl`, `rf_weighted.pkl`, `rf_tuned_main.pkl`, `logreg_pembanding.pkl`, `xgb_pembanding.pkl`
- `preprocessor.pkl` (preprocessing pipeline)
- `metrics.pkl` (evaluation metrics for all models)
- `shap_artifact.pkl` and `shap_summary.png` (SHAP interpretation)

### 2. Menjalankan Streamlit dashboard utama

```bash
streamlit run app.py
```

Aplikasi akan tersedia di `http://localhost:8501`.

### 3. Menjalankan Streamlit frontend (opsional)

```bash
streamlit run frontend/app.py
```

### 4. Menjalankan FastAPI backend (opsional)

```bash
uvicorn api.main:app --reload
```

## Fitur yang Tersedia di Dashboard Utama (`app.py`)

### 📊 Dashboard EDA
- Statistik deskriptif fitur numerik
- Tabel missing values
- Distribusi kelas gaji (bar chart + pie chart)
- Analisis relasi fitur vs gaji (boxplot untuk numerik, stacked bar untuk kategorik)

### 🔮 Model Demo
- Input form untuk memasukkan data pekerja
- Prediksi kategori gaji (`<=7jt` / `>7jt`)
- Probabilitas prediksi
- Progress bar probabilitas

### 🎯 Evaluasi Model
- Perbandingan 5 model (RF Baseline, RF Weighted, RF Tuned, Logistic Regression, XGBoost) dalam tabel dan bar chart
- Confusion Matrix untuk semua 5 model
- Cross Validation F1-score per Fold (Model Utama: RF Tuned)
- Learning Curve (Model Utama: RF Tuned)

### 💡 Interpretasi Hasil
- Feature Importance (Mean Decrease in Impurity)
- Permutation Importance
- SHAP Summary Plot (gambar statis)
- Insight bisnis dari interpretasi

### 📖 Dokumentasi
- Penjelasan dataset
- Ringkasan metodologi (preprocessing, model, imbalance handling, interpretasi)
- Metrik kesuksesan proyek (Recall >7jt > 0.75, ROC-AUC > 0.85)
- Panduan penggunaan aplikasi

## Sidebar Dashboard
- Logo dan portal info
- **⭐ Ujian Akhir Semester ⭐**
- **Pembelajaran Mesin**
- **👩‍💻 Developer**
  - Nama : Kayla Verda Fawnia Heses
  - NIM  : A11.2024.15689
  - Kelas: A11.4404
  - Teknik Informatika – Universitas Dian Nuswantoro

## Footer Dashboard
- **© KAYLA VERDA - A11.2024.15689**

## Link Aplikasi (Deployed)
https://klasifikasigajipm.streamlit.app/
