import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_ROOT / "Dataset" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "model"

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Salary Classifier Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Aesthetic - Dark Mode Theme & Harmonious Color Palette)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Styling with Gradient */
    .title-gradient {
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        color: #A5F3FC;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Styling */
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4D96FF;
    }
    
    .metric-card-title {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .metric-card-value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Load dataset helper with caching
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Load raw train.csv and do basic cleaning of columns for EDA."""
    if not TRAIN_CSV.exists():
        st.error(f"Dataset tidak ditemukan di path: {TRAIN_CSV}. Pastikan dataset diletakkan dengan benar.")
        return pd.DataFrame()
    df = pd.read_csv(TRAIN_CSV, sep=";")
    return df


def main() -> None:
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=80)
    st.sidebar.markdown("### Salary Predictor & EDA Portal")
    st.sidebar.info("Gunakan navigasi tab di sebelah kanan untuk menjelajahi aplikasi.")
    
    # App Header
    st.markdown('<div class="title-gradient">💰 Salary Classifier Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Prediksi dan Analisis Kategori Gaji Pekerja berbasis Machine Learning (Random Forest)</div>', unsafe_allow_html=True)

    # 5 Tabs
    tabs = st.tabs([
        "📊 Dashboard EDA",
        "🔮 Model Demo",
        "🎯 Evaluasi Model",
        "💡 Interpretasi Hasil",
        "📖 Dokumentasi"
    ])
    
    # Load Data
    df_raw = load_raw_data()
    if df_raw.empty:
        return

    # ==================== TAB 1: DASHBOARD EDA ====================
    with tabs[0]:
        st.header("Analisis Data Eksploratif (EDA)")
        st.markdown("Tab ini menyajikan wawasan mendalam tentang struktur data dan pengaruh fitur-fitur pekerja terhadap kategori gaji.")
        
        # 1. Summary Statistics & Metrics Cards
        st.subheader("Ringkasan Statistik & Dataset")
        total_rows = len(df_raw)
        missing_cells = df_raw.isin(["?"]).sum().sum()
        duplicated_rows = df_raw.drop(columns=["id"]).duplicated().sum()
        
        # Metric Grid
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Jumlah Baris Data</div>
                <div class="metric-card-value">{total_rows:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Missing Values ('?')</div>
                <div class="metric-card-value">{missing_cells:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Baris Duplikat (Tanpa ID)</div>
                <div class="metric-card-value">{duplicated_rows:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        
        # Descriptive Stats Table & Missing Values Table
        stat_col1, stat_col2 = st.columns([2, 1])
        with stat_col1:
            st.markdown("##### Statistik Deskriptif Fitur Numerik")
            st.dataframe(df_raw.describe().round(2), use_container_width=True)
        with stat_col2:
            st.markdown("##### Nilai Hilang per Kolom")
            # Replace "?" to nan to count missing values
            df_missing = df_raw.replace("?", np.nan)
            missing_series = df_missing.isnull().sum()
            missing_df = pd.DataFrame({
                "Kolom": missing_series.index,
                "Missing Values": missing_series.values,
                "Persentase": (missing_series.values / len(df_raw) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df["Missing Values"] > 0].reset_index(drop=True), use_container_width=True)
            
        st.divider()
        
        # 2. Target Variable Distribution (Plotly Bar & Pie)
        st.subheader("Distribusi Kelas Target (Gaji)")
        dist_col1, dist_col2 = st.columns(2)
        
        target_counts = df_raw["Gaji"].value_counts().reset_index()
        target_counts.columns = ["Gaji", "Jumlah"]
        
        with dist_col1:
            fig_bar = px.bar(
                target_counts, 
                x="Gaji", 
                y="Jumlah", 
                color="Gaji",
                color_discrete_map={"<=7jt": "#4D96FF", ">7jt": "#FF6B6B"},
                text_auto=True,
                title="Jumlah Pekerja per Kategori Gaji"
            )
            fig_bar.update_layout(showlegend=False, template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with dist_col2:
            fig_pie = px.pie(
                target_counts, 
                names="Gaji", 
                values="Jumlah",
                color="Gaji",
                color_discrete_map={"<=7jt": "#4D96FF", ">7jt": "#FF6B6B"},
                hole=0.4,
                title="Proporsi Kategori Gaji"
            )
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.divider()
        
        # 3. Dynamic Feature Distribution vs Gaji (Interactive Boxplot / Bar Chart)
        st.subheader("Analisis Relasi Fitur terhadap Kategori Gaji")
        features_to_analyze = [
            "Umur", 
            "Jam per Minggu", 
            "Jmlh Tahun Pendidikan", 
            "Status Perkawinan", 
            "Jenis Kelamin", 
            "Pendidikan"
        ]
        
        selected_feature = st.selectbox(
            "Pilih fitur yang ingin dianalisis hubungannya dengan Gaji:",
            features_to_analyze
        )
        
        # Determine if numeric or categorical
        is_numeric = df_raw[selected_feature].dtype in [np.int64, np.float64]
        
        if is_numeric:
            fig_relation = px.box(
                df_raw,
                x="Gaji",
                y=selected_feature,
                color="Gaji",
                color_discrete_map={"<=7jt": "#4D96FF", ">7jt": "#FF6B6B"},
                points="outliers",
                title=f"Distribusi {selected_feature} berdasarkan Kategori Gaji"
            )
            fig_relation.update_layout(template="plotly_dark")
            st.plotly_chart(fig_relation, use_container_width=True)
        else:
            # Categorical - Plot a stacked/grouped proportion bar chart
            # First, group by selected feature and Gaji
            df_group = df_raw[df_raw[selected_feature] != "?"].groupby([selected_feature, "Gaji"]).size().reset_index(name="Count")
            
            # Calculate proportion within each category of the selected feature
            df_totals = df_group.groupby(selected_feature)["Count"].transform("sum")
            df_group["Proporsi"] = df_group["Count"] / df_totals
            
            fig_relation = px.bar(
                df_group,
                x=selected_feature,
                y="Proporsi",
                color="Gaji",
                color_discrete_map={"<=7jt": "#4D96FF", ">7jt": "#FF6B6B"},
                barmode="group",
                title=f"Proporsi Kategori Gaji berdasarkan {selected_feature}",
                labels={"Proporsi": "Proporsi Pekerja"}
            )
            fig_relation.update_layout(yaxis_tickformat=".0%", template="plotly_dark")
            st.plotly_chart(fig_relation, use_container_width=True)

    # ==================== TAB 2: MODEL DEMO ====================
    with tabs[1]:
        st.header("Demo Prediksi Kategori Gaji")
        st.warning("Bagian ini akan diimplementasikan secara penuh pada Bagian 3.")

    # ==================== TAB 3: EVALUASI MODEL ====================
    with tabs[2]:
        st.header("Evaluasi Kinerja Model")
        st.warning("Bagian ini akan diimplementasikan secara penuh pada Bagian 3.")

    # ==================== TAB 4: INTERPRETASI HASIL ====================
    with tabs[3]:
        st.header("Interpretasi & Insight Model")
        st.warning("Bagian ini akan diimplementasikan secara penuh pada Bagian 4.")

    # ==================== TAB 5: DOKUMENTASI ====================
    with tabs[4]:
        st.header("Dokumentasi Proyek")
        
        st.subheader("1. Penjelasan Dataset")
        st.markdown("""
        Dataset yang digunakan dalam proyek ini diadaptasi dari **UCI Adult Census Income Dataset** (Kohavi, 1996) 
        yang memuat profil sosio-demografis dan pekerjaan individu untuk memprediksi tingkat pendapatan. 
        Dataset mentah (`train.csv`) memiliki total baris **36.017 baris** dan **12 kolom** (fitur).
        """)
        
        # Column description table
        doc_cols = pd.DataFrame([
            {"Nama Fitur": "id", "Tipe": "Numerik", "Deskripsi": "Indeks / Nomor Urut data (Dihapus dari Pemodelan)"},
            {"Nama Fitur": "Umur", "Tipe": "Numerik", "Deskripsi": "Usia Pekerja (di-clipping dalam batas IQR [1 - 79] tahun)"},
            {"Nama Fitur": "Kelas Pekerja", "Tipe": "Kategorik", "Deskripsi": "Sektor ketenagakerjaan (Wiraswasta, Pekerja Negara, dll.)"},
            {"Nama Fitur": "Pendidikan", "Tipe": "Kategorik (Ordinal)", "Deskripsi": "Tingkat pendidikan tertinggi (SD hingga Doktor)"},
            {"Nama Fitur": "Jmlh Tahun Pendidikan", "Tipe": "Numerik", "Deskripsi": "Lama durasi menempuh pendidikan dalam tahun"},
            {"Nama Fitur": "Status Perkawinan", "Tipe": "Kategorik", "Deskripsi": "Status perkawinan pekerja (Menikah, Belum Menikah, Cerai, dll.)"},
            {"Nama Fitur": "Pekerjaan", "Tipe": "Kategorik", "Deskripsi": "Sektor pekerjaan spesifik pekerja (Manajerial, Spesialis, dll.)"},
            {"Nama Fituk": "Jenis Kelamin", "Tipe": "Kategorik", "Deskripsi": "Jenis kelamin pekerja (Laki2, Perempuan)"},
            {"Nama Fitur": "Keuntungan Kapital", "Tipe": "Numerik", "Deskripsi": "Keuntungan dari investasi keuangan (di-clipping IQR ke 0)"},
            {"Nama Fitur": "Kerugian Capital", "Tipe": "Numerik", "Deskripsi": "Kerugian dari investasi keuangan (di-clipping IQR ke 0)"},
            {"Nama Fitur": "Jam per Minggu", "Tipe": "Numerik", "Deskripsi": "Jumlah jam kerja per minggu (di-clipping IQR ke [26 - 58] jam)"},
            {"Nama Fitur": "Gaji (Target)", "Tipe": "Kategorik", "Deskripsi": "Kategori Gaji Pekerja (<=7jt atau >7jt)"}
        ])
        st.table(doc_cols)
        
        st.subheader("2. Ringkasan Metodologi")
        st.markdown("""
        - **Preprocessing Pipeline**:
          - **Pembersihan**: Menghapus kolom `id` dan `Berat Akhir` karena tidak relevan secara prediktif. Mengganti nilai string '?' menjadi NaN pada kolom `Kelas Pekerja` dan `Pekerjaan`. Menghapus duplikasi baris.
          - **Handling Missing Values**: Nilai hilang diimputasi menggunakan metode *Mode* (nilai tersering/`most_frequent`) menggunakan `SimpleImputer` di dalam pipeline.
          - **Handling Outlier**: Digunakan metode *IQR Clipping* dengan batas $[Q1 - 1.5 \\times IQR, Q3 + 1.5 \\times IQR]$ pada kolom numerik (`Umur`, `Keuntungan Kapital`, `Kerugian Capital`, `Jam per Minggu`) untuk membatasi rentang nilai ekstrim tanpa membuang baris data.
          - **Encoding**: 
            - *OrdinalEncoder* diterapkan pada kolom `Pendidikan` berdasarkan urutan logis (SD -> SMA -> D3 -> Sarjana -> Master -> Doktor).
            - *OneHotEncoder* dengan parameter `handle_unknown='ignore'` diterapkan pada variabel nominal.
          - **Scaling**: Standardisasi data menggunakan `StandardScaler` untuk menyamakan skala fitur-fitur numerik.
        - **Algoritma Permodelan**:
          - Menggunakan **Random Forest Classifier** yang merupakan algoritma *ensemble* (berbasis bagging) yang sangat andal menangani fitur campuran (numerik & kategorikal) serta hubungan non-linear.
        - **Mengatasi Imbalance Class**:
          - Karena proporsi kelas target tidak seimbang (hanya ~24% berpenghasilan >7jt), model utama dikonfigurasi dengan parameter `class_weight='balanced'` guna memberikan penalti kesalahan lebih tinggi pada kelas minoritas demi meningkatkan Recall dan skor F1.
        """)
        
        st.subheader("3. Panduan Penggunaan Aplikasi")
        st.markdown("""
        - **Dashboard EDA**: Membantu pengguna memvisualisasikan data mentah, sebaran kategori gaji, dan hubungan antar variabel melalui chart interaktif.
        - **Model Demo**: Memungkinkan pengguna memasukkan profil pekerja (umur, pendidikan, pekerjaan, dll.) untuk melihat prediksi kategori gajinya secara instan.
        - **Evaluasi Model**: Berisi perbandingan kinerja 3 model Random Forest serta kurva pembelajaran (learning curve) untuk memantau bias dan variansi model.
        - **Interpretasi Hasil**: Membantu analis/pengguna memahami fitur mana yang paling memengaruhi keputusan prediksi model utama.
        """)


if __name__ == "__main__":
    main()
