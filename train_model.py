"""Train Random Forest models and save artifacts for Streamlit deployment."""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Dataset" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "model"

# Feature definitions
URUTAN_PENDIDIKAN = [
    "SD",
    "1st-4th",
    "5th-6th",
    "7th-8th",
    "9th",
    "10th",
    "11th",
    "12th",
    "SMA",
    "Pendidikan Tinggi",
    "D3",
    "D4",
    "Sarjana",
    "Master",
    "Sekolah Professional",
    "Doktor",
]

FITUR_NUMERIK = [
    "Umur",
    "Jmlh Tahun Pendidikan",
    "Keuntungan Kapital",
    "Kerugian Capital",
    "Jam per Minggu",
]
FITUR_ORDINAL = ["Pendidikan"]
FITUR_NOMINAL = ["Kelas Pekerja", "Status Perkawinan", "Pekerjaan", "Jenis Kelamin"]
KOLOM_CLIP = ["Umur", "Keuntungan Kapital", "Kerugian Capital", "Jam per Minggu"]


def load_and_clean_data() -> pd.DataFrame:
    """Load train.csv, clean missing values, drop unused columns, and drop duplicates."""
    df = pd.read_csv(DATA_PATH, sep=";")
    
    # 1. Drop kolom id dan Berat Akhir
    df = df.drop(columns=["id", "Berat Akhir"])
    
    # 2. Ganti nilai '?' pada Kelas Pekerja dan Pekerjaan menjadi NaN
    for col in ["Kelas Pekerja", "Pekerjaan"]:
        df[col] = df[col].replace("?", np.nan)
        
    # Konversi target ke biner untuk modeling
    df["target"] = (df["Gaji"] == ">7jt").astype(int)
    df = df.drop(columns=["Gaji"])
    
    # 3. Hapus baris duplikat
    df = df.drop_duplicates()
    
    return df


def apply_iqr_clipping(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Calculate IQR bounds and clip outliers on specified numerical columns."""
    df = df.copy()
    iqr_bounds = {}
    
    for col in KOLOM_CLIP:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_bounds[col] = (float(lower), float(upper))
        df[col] = df[col].clip(lower, upper)
        
    return df, iqr_bounds


def build_preprocessor() -> ColumnTransformer:
    """Construct ColumnTransformer for numerical, ordinal, and nominal features."""
    return ColumnTransformer(
        transformers=[
            ("numerik", StandardScaler(), FITUR_NUMERIK),
            (
                "ordinal",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(categories=[URUTAN_PENDIDIKAN])),
                    ]
                ),
                FITUR_ORDINAL,
            ),
            (
                "nominal",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                FITUR_NOMINAL,
            ),
        ]
    )


def main() -> None:
    print("Membaca dan memproses data awal...")
    df = load_and_clean_data()
    df, iqr_bounds = apply_iqr_clipping(df)
    
    # Split features and target
    feature_columns = FITUR_NUMERIK + FITUR_ORDINAL + FITUR_NOMINAL
    X = df[feature_columns]
    y = df["target"]
    
    # 5. Split data 80:20 (stratify=y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Fit preprocessor on train set
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Reconstruct processed feature names for feature importance interpretation
    one_hot_encoder = preprocessor.named_transformers_["nominal"].named_steps["encoder"]
    nominal_feature_names = list(one_hot_encoder.get_feature_names_out(FITUR_NOMINAL))
    all_feature_names = FITUR_NUMERIK + FITUR_ORDINAL + nominal_feature_names
    
    # 7. Inisialisasi 3 model untuk dibandingkan
    models = {
        "baseline": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "weighted": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "tuned": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }
    
    metrics = {}
    
    # Train and evaluate each model on test set
    for name, model in models.items():
        print(f"Melatih model {name}...")
        model.fit(X_train_processed, y_train)
        
        preds = model.predict(X_test_processed)
        probs = model.predict_proba(X_test_processed)[:, 1]
        
        # Hitung metrik evaluasi
        report_dict = classification_report(y_test, preds, target_names=["<=7jt", ">7jt"], digits=4, output_dict=True)
        report_text = classification_report(y_test, preds, target_names=["<=7jt", ">7jt"], digits=4)
        cm = confusion_matrix(y_test, preds).tolist()
        roc_auc = float(roc_auc_score(y_test, probs))
        
        metrics[name] = {
            "classification_report_dict": report_dict,
            "classification_report_text": report_text,
            "confusion_matrix": cm,
            "roc_auc_score": roc_auc
        }
    
    # 8. Hitung K-Fold CV untuk model utama (tuned)
    print("Menghitung StratifiedKFold 5-fold CV untuk model utama...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = []
    cv_auc = []
    
    for train_idx, val_idx in cv.split(X_train_processed, y_train):
        # We need numpy arrays for indexing since X_train_processed is a sparse/dense numpy matrix
        X_tr, X_val = X_train_processed[train_idx], X_train_processed[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        fold_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        fold_model.fit(X_tr, y_tr)
        preds_val = fold_model.predict(X_val)
        probs_val = fold_model.predict_proba(X_val)[:, 1]
        
        cv_f1.append(f1_score(y_val, preds_val))
        cv_auc.append(roc_auc_score(y_val, probs_val))
        
    metrics["tuned"]["cv_scores_f1"] = cv_f1
    metrics["tuned"]["cv_scores_auc"] = cv_auc
    
    # Hitung learning curve untuk model utama (tuned)
    print("Menghitung learning curve untuk model utama...")
    train_sizes, train_scores, val_scores = learning_curve(
        models["tuned"],
        X_train_processed,
        y_train,
        cv=5,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42,
        n_jobs=-1
    )
    
    metrics["tuned"]["learning_curve"] = {
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist()
    }
    
    # Feature importance model utama
    metrics["tuned"]["feature_importances"] = {
        "importances": models["tuned"].feature_importances_.tolist(),
        "names": all_feature_names
    }
    
    # 9. Simpan model utama, preprocessor, dan metrics.pkl
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(models["tuned"], MODEL_DIR / "rf_balanced_model.pkl")
    joblib.dump(
        {"preprocessor": preprocessor, "iqr_bounds": iqr_bounds},
        MODEL_DIR / "preprocessor.pkl"
    )
    joblib.dump(metrics, MODEL_DIR / "metrics.pkl")
    
    print("\nProses selesai!")
    print(f"Model utama disimpan ke: {MODEL_DIR / 'rf_balanced_model.pkl'}")
    print(f"Preprocessor disimpan ke: {MODEL_DIR / 'preprocessor.pkl'}")
    print(f"Metrics evaluasi disimpan ke: {MODEL_DIR / 'metrics.pkl'}")


if __name__ == "__main__":
    main()
