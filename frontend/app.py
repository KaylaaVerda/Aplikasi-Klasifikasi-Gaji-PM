from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "Dataset" / "train.csv"
API_URL = "http://127.0.0.1:8000/predict"


def _unique_values(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).unique()
    return sorted(v for v in values if v != "?")


@st.cache_data
def load_categories() -> dict[str, list[str]]:
    df = pd.read_csv(TRAIN_CSV, sep=";")
    return {
        "kelas_pekerja": _unique_values(df["Kelas Pekerja"]),
        "pendidikan": _unique_values(df["Pendidikan"]),
        "status_perkawinan": _unique_values(df["Status Perkawinan"]),
        "pekerjaan": _unique_values(df["Pekerjaan"]),
        "jenis_kelamin": _unique_values(df["Jenis Kelamin"]),
    }


def main() -> None:
    st.set_page_config(page_title="Prediksi Kategori Gaji", page_icon="💰", layout="centered")
    st.title("Prediksi Kategori Gaji")
    st.caption("Aplikasi klasifikasi gaji berbasis Random Forest")

    categories = load_categories()

    with st.form("prediction_form"):
        st.subheader("Data Pekerja")

        col1, col2 = st.columns(2)

        with col1:
            umur = st.number_input("Umur", min_value=17, max_value=100, value=30)
            kelas_pekerja = st.selectbox("Kelas Pekerja", categories["kelas_pekerja"])
            pendidikan = st.selectbox("Pendidikan", categories["pendidikan"])
            jmlh_tahun_pendidikan = st.number_input(
                "Jumlah Tahun Pendidikan", min_value=0, max_value=20, value=12
            )
            status_perkawinan = st.selectbox("Status Perkawinan", categories["status_perkawinan"])

        with col2:
            pekerjaan = st.selectbox("Pekerjaan", categories["pekerjaan"])
            jenis_kelamin = st.selectbox("Jenis Kelamin", categories["jenis_kelamin"])
            keuntungan_kapital = st.number_input(
                "Keuntungan Kapital", min_value=0.0, value=0.0, step=1000.0, format="%.0f"
            )
            kerugian_capital = st.number_input(
                "Kerugian Capital", min_value=0.0, value=0.0, step=1000.0, format="%.0f"
            )
            jam_per_minggu = st.number_input(
                "Jam per Minggu", min_value=1.0, max_value=100.0, value=40.0, step=1.0
            )

        submitted = st.form_submit_button("Prediksi Kategori Gaji", type="primary")

    if submitted:
        payload = {
            "umur": int(umur),
            "kelas_pekerja": kelas_pekerja,
            "pendidikan": pendidikan,
            "jmlh_tahun_pendidikan": int(jmlh_tahun_pendidikan),
            "status_perkawinan": status_perkawinan,
            "pekerjaan": pekerjaan,
            "jenis_kelamin": jenis_kelamin,
            "keuntungan_kapital": float(keuntungan_kapital),
            "kerugian_capital": float(kerugian_capital),
            "jam_per_minggu": float(jam_per_minggu),
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Tidak dapat terhubung ke API. Pastikan FastAPI berjalan di http://127.0.0.1:8000")
            return
        except requests.exceptions.HTTPError:
            st.error(f"API error: {response.text}")
            return

        st.divider()
        st.subheader("Hasil Prediksi")

        predicted = result["predicted_kategori_gaji"]
        if predicted == ">7jt":
            st.success(f"Kategori Gaji Prediksi: **{predicted}**")
        else:
            st.info(f"Kategori Gaji Prediksi: **{predicted}**")

        prob_0 = result["probabilitas_kelas_0"]
        prob_1 = result["probabilitas_kelas_1"]

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Probabilitas <=7jt", f"{prob_0:.2%}")
        metric_col2.metric("Probabilitas >7jt", f"{prob_1:.2%}")

        st.write("Distribusi Probabilitas")
        st.progress(prob_0, text=f"<=7jt: {prob_0:.2%}")
        st.progress(prob_1, text=f">7jt: {prob_1:.2%}")

        st.caption(f"Model: {result['model_used']}")


if __name__ == "__main__":
    main()
