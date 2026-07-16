"""Train five machine learning models and save evaluation metrics and artifacts for Streamlit deployment."""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Matplotlib Agg backend for headless systems
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, RandomizedSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.inspection import permutation_importance
import shap

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH_PEKERJA = PROJECT_ROOT / "Dataset" / "DatasetPekerja.csv"
DATA_PATH_TRAIN = PROJECT_ROOT / "Dataset" / "train.csv"
DATA_PATH = DATA_PATH_PEKERJA if DATA_PATH_PEKERJA.exists() else DATA_PATH_TRAIN
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
    df_raw = pd.read_csv(DATA_PATH, sep=";")
    df = df_raw.copy()
    
    # 1. Drop id and Berat Akhir columns
    df = df.drop(columns=["id", "Berat Akhir"])
    
    # 2. Replace '?' with NaN globally
    df = df.replace("?", np.nan)
    
    # 3. Convert target class to binary
    df["target"] = (df["Gaji"] == ">7jt").astype(int)
    df = df.drop(columns=["Gaji"])
    
    # 4. Drop duplicate rows
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
    
    # Split data 80:20 (stratify=y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = build_preprocessor()
    
    # 1. RF Baseline
    print("Melatih model RF Baseline...")
    rf_baseline = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf_baseline.fit(X_train, y_train)
    
    # 2. RF Weighted
    print("Melatih model RF Weighted...")
    rf_weighted = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    rf_weighted.fit(X_train, y_train)
    
    # 3. RF Tuned
    print("Melatih model RF Tuned (Tuning via RandomizedSearchCV)...")
    rf_tuning_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    param_dist = {
        'model__n_estimators': [100, 150, 200],
        'model__max_depth': [10, 15, 20, None],
        'model__min_samples_leaf': [1, 5, 10],
        'model__min_samples_split': [2, 10],
        'model__max_features': ['sqrt', 'log2']
    }
    skf_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        rf_tuning_pipe,
        param_distributions=param_dist,
        n_iter=8,
        scoring='f1',
        cv=skf_search,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    rf_tuned = random_search.best_estimator_
    print("Hyperparameter terbaik:", random_search.best_params_)
    
    # 4. Logistic Regression
    print("Melatih model Logistic Regression...")
    logreg = Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1))
    ])
    logreg.fit(X_train, y_train)
    
    # 5. XGBoost
    print("Melatih model XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight yang digunakan: {scale_pos_weight:.4f}")
    xgb_model = Pipeline([
        ('prep', preprocessor),
        ('model', XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric='logloss', n_jobs=-1
        ))
    ])
    xgb_model.fit(X_train, y_train)
    
    # Metrics collection
    models = {
        "baseline": rf_baseline,
        "weighted": rf_weighted,
        "tuned": rf_tuned,
        "logreg": logreg,
        "xgb": xgb_model
    }
    
    metrics = {}
    comparison_rows = []
    
    for name, pipeline in models.items():
        print(f"Mengevaluasi model {name}...")
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label=1)
        rec = recall_score(y_test, preds, pos_label=1)
        f1 = f1_score(y_test, preds, pos_label=1)
        roc_auc = roc_auc_score(y_test, probs)
        
        report_dict = classification_report(y_test, preds, target_names=["<=7jt", ">7jt"], digits=4, output_dict=True)
        report_text = classification_report(y_test, preds, target_names=["<=7jt", ">7jt"], digits=4)
        cm = confusion_matrix(y_test, preds).tolist()
        
        metrics[name] = {
            "classification_report_dict": report_dict,
            "classification_report_text": report_text,
            "confusion_matrix": cm,
            "roc_auc_score": float(roc_auc)
        }
        
        # Display name mapping for table
        display_name = {
            "baseline": "RF Baseline",
            "weighted": "RF Weighted",
            "tuned": "RF Tuned",
            "logreg": "Logistic Regression",
            "xgb": "XGBoost"
        }[name]
        
        comparison_rows.append({
            "Model": display_name,
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "ROC-AUC": float(roc_auc)
        })
        
    # Store comparison table as DataFrame in metrics
    comparison_df = pd.DataFrame(comparison_rows)
    metrics["comparison_table"] = comparison_rows
    
    print("\n=== PERBANDINGAN MODEL ===")
    print(comparison_df.to_string(index=False))
    
    # 5-fold CV for Tuned Model
    print("\nMenghitung StratifiedKFold 5-fold CV untuk model utama (tuned)...")
    skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        rf_tuned, X_train, y_train,
        cv=skf_cv, scoring=['f1', 'roc_auc'], n_jobs=-1
    )
    metrics["tuned"]["cv_scores_f1"] = cv_results['test_f1'].tolist()
    metrics["tuned"]["cv_scores_auc"] = cv_results['test_roc_auc'].tolist()
    
    # Learning Curve for Tuned Model
    print("Menghitung learning curve untuk model utama (tuned)...")
    skf_lc = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        rf_tuned, X_train, y_train,
        cv=skf_lc, scoring='f1', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    metrics["tuned"]["learning_curve"] = {
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist()
    }
    
    # Default Feature Importances for Tuned Model
    ohe_enc = rf_tuned.named_steps['prep'].named_transformers_['nominal'].named_steps['encoder']
    nama_fitur = FITUR_NUMERIK + FITUR_ORDINAL + list(ohe_enc.get_feature_names_out(FITUR_NOMINAL))
    importances = rf_tuned.named_steps['model'].feature_importances_
    metrics["tuned"]["feature_importances"] = {
        "importances": importances.tolist(),
        "names": nama_fitur
    }
    
    # Permutation Importance for Tuned Model
    print("Menghitung permutation importance untuk model utama (tuned)...")
    perm_result = permutation_importance(
        rf_tuned, X_test, y_test,
        n_repeats=10, random_state=42, scoring='f1', n_jobs=-1
    )
    metrics["tuned"]["permutation_importance"] = {
        "importances_mean": perm_result.importances_mean.tolist(),
        "importances_std": perm_result.importances_std.tolist(),
        "names": X_test.columns.tolist()
    }
    
    # SHAP Values on Subsample 300
    print("Menghitung SHAP values pada subsample 300 data uji...")
    X_test_transformed = rf_tuned.named_steps['prep'].transform(X_test)
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=nama_fitur)
    
    np.random.seed(42)
    sample_idx = np.random.choice(X_test_transformed.shape[0], size=300, replace=False)
    X_shap_sample = X_test_transformed.iloc[sample_idx]
    
    explainer = shap.TreeExplainer(rf_tuned.named_steps['model'])
    shap_values = explainer.shap_values(X_shap_sample)
    
    if isinstance(shap_values, list):
        shap_values_kelas1 = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values_kelas1 = shap_values[:, :, 1]
    else:
        shap_values_kelas1 = shap_values
        
    shap_artifact = {
        "shap_values": shap_values_kelas1,
        "nama_fitur": nama_fitur,
        "expected_value": explainer.expected_value,
        "X_shap_sample": X_shap_sample
    }
    
    # Save SHAP Summary Plot as static PNG
    print("Menyimpan SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_kelas1, X_shap_sample, show=False, max_display=15)
    plt.title('SHAP Summary Plot (Kontribusi Fitur terhadap Prediksi Kelas >7jt)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(MODEL_DIR / "shap_summary.png", dpi=150)
    plt.close()
    
    # Save artifacts to model/
    print("Menyimpan seluruh model dan artifacts...")
    joblib.dump(rf_baseline, MODEL_DIR / "rf_baseline.pkl")
    joblib.dump(rf_weighted, MODEL_DIR / "rf_weighted.pkl")
    joblib.dump(rf_tuned, MODEL_DIR / "rf_tuned_main.pkl")
    joblib.dump(logreg, MODEL_DIR / "logreg_pembanding.pkl")
    joblib.dump(xgb_model, MODEL_DIR / "xgb_pembanding.pkl")
    
    joblib.dump(
        {"preprocessor": rf_tuned.named_steps['prep'], "iqr_bounds": iqr_bounds},
        MODEL_DIR / "preprocessor.pkl"
    )
    joblib.dump(metrics, MODEL_DIR / "metrics.pkl")
    joblib.dump(shap_artifact, MODEL_DIR / "shap_artifact.pkl")
    
    print("\nProses selesai dengan sukses!")
    print(f"Artifacts disimpan ke folder: {MODEL_DIR.absolute()}")


if __name__ == "__main__":
    main()
