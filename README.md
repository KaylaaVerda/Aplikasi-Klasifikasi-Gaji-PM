# Aplikasi Klasifikasi Gaji

Deployment sederhana untuk model Random Forest klasifikasi kategori gaji pekerja (`<=7jt` / `>7jt`).

## Struktur Proyek

```
model/
  train_model.py
  rf_balanced_model.pkl      (dibuat setelah training)
  preprocessor.pkl           (dibuat setelah training)
api/
  main.py
  schemas.py
  inference.py
frontend/
  app.py
Dataset/
  train.csv
requirements.txt
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

Jalankan perintah berikut **secara berurutan** dari folder root proyek.

### 1. Training model

```bash
python model/train_model.py
```

Perintah ini akan membuat:
- `model/rf_balanced_model.pkl`
- `model/preprocessor.pkl`

### 2. Menjalankan API (FastAPI)

```bash
uvicorn api.main:app --reload
```

API tersedia di `http://127.0.0.1:8000`

Dokumentasi interaktif: `http://127.0.0.1:8000/docs`

Contoh request ke endpoint `POST /predict`:

```json
{
  "umur": 44,
  "kelas_pekerja": "Wiraswasta",
  "pendidikan": "Sarjana",
  "jmlh_tahun_pendidikan": 13,
  "status_perkawinan": "Menikah",
  "pekerjaan": "Ekesekutif Managerial",
  "jenis_kelamin": "Laki2",
  "keuntungan_kapital": 61404000.0,
  "kerugian_capital": 0.0,
  "jam_per_minggu": 45.0
}
```

### 3. Menjalankan frontend (Streamlit)

Buka terminal baru, lalu:

```bash
streamlit run frontend/app.py
```

Frontend tersedia di `http://localhost:8501`

Pastikan API FastAPI masih berjalan saat menggunakan form Streamlit.
