from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

FEATURE_COLUMNS = [
    "Umur",
    "Kelas Pekerja",
    "Pendidikan",
    "Jmlh Tahun Pendidikan",
    "Status Perkawinan",
    "Pekerjaan",
    "Jenis Kelamin",
    "Keuntungan Kapital",
    "Kerugian Capital",
    "Jam per Minggu",
]

_model = None
_preprocessor = None
_iqr_bounds = None


def _load_artifacts() -> None:
    global _model, _preprocessor, _iqr_bounds

    if _model is not None:
        return

    model_path = MODEL_DIR / "rf_balanced_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            "Model belum tersedia. Jalankan `python model/train_model.py` terlebih dahulu."
        )

    _model = joblib.load(model_path)
    artifacts = joblib.load(preprocessor_path)
    _preprocessor = artifacts["preprocessor"]
    _iqr_bounds = artifacts["iqr_bounds"]


def _request_to_dataframe(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Umur": data["umur"],
                "Kelas Pekerja": data["kelas_pekerja"],
                "Pendidikan": data["pendidikan"],
                "Jmlh Tahun Pendidikan": data["jmlh_tahun_pendidikan"],
                "Status Perkawinan": data["status_perkawinan"],
                "Pekerjaan": data["pekerjaan"],
                "Jenis Kelamin": data["jenis_kelamin"],
                "Keuntungan Kapital": data["keuntungan_kapital"],
                "Kerugian Capital": data["kerugian_capital"],
                "Jam per Minggu": data["jam_per_minggu"],
            }
        ]
    )[FEATURE_COLUMNS]


def _apply_iqr_clipping(df: pd.DataFrame) -> pd.DataFrame:
    clipped = df.copy()
    for col, (lower, upper) in _iqr_bounds.items():
        clipped[col] = clipped[col].clip(lower, upper)
    return clipped


def predict(data: dict) -> dict:
    _load_artifacts()

    df = _request_to_dataframe(data)
    df = _apply_iqr_clipping(df)
    features = _preprocessor.transform(df)
    probabilities = _model.predict_proba(features)[0]

    prob_kelas_0 = float(probabilities[0])
    prob_kelas_1 = float(probabilities[1])
    predicted_label = ">7jt" if prob_kelas_1 >= prob_kelas_0 else "<=7jt"

    return {
        "predicted_kategori_gaji": predicted_label,
        "probabilitas_kelas_1": prob_kelas_1,
        "probabilitas_kelas_0": prob_kelas_0,
        "model_used": "Random Forest (class_weight=balanced)",
    }
