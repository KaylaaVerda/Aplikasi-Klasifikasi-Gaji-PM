# Aplikasi Klasifikasi Gaji

Aplikasi Streamlit sederhana untuk memprediksi kategori gaji pekerja (`<=7jt` / `>7jt`) menggunakan model Random Forest.

## Struktur Proyek

```
app.py
train_model.py
Dataset/
  train.csv
model/
  rf_balanced_model.pkl      (dibuat setelah training)
  preprocessor.pkl           (dibuat setelah training)
  metrics.pkl                (dibuat setelah training)
requirements.txt
README.md
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

Jalankan perintah berikut dari folder root proyek.

### 1. Training model

```bash
python train_model.py
```

Perintah ini akan membuat:
- `model/rf_balanced_model.pkl`
- `model/preprocessor.pkl`
- `model/metrics.pkl`

### 2. Menjalankan Streamlit app

```bash
streamlit run app.py
```

Aplikasi akan tersedia di `http://localhost:8501`.

## Fitur yang tersedia

- Dashboard EDA untuk melihat distribusi data dan relasi fitur
- Demo prediksi kelas gaji berdasarkan input pengguna
- Evaluasi model dengan confusion matrix, cross-validation, dan learning curve
- Interpretasi fitur penting untuk memahami pengaruh variabel terhadap prediksi
