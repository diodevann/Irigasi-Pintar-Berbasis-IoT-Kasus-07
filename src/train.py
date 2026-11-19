from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.features import (
    AMBANG_AGRONOMI,
    FEATURE_COLUMNS,
    TANAH_DICT,
    TANAMAN_DICT,
    build_batch_features,
    clean_raw_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def find_raw_data_path() -> Path:
    for p in [PROJECT_ROOT / "data" / "raw.csv.gz", PROJECT_ROOT / "data" / "raw.csv"]:
        if p.exists():
            return p
    raise FileNotFoundError("Dataset tidak ditemukan di folder data/ (raw.csv.gz atau raw.csv).")


def train_and_save_artifacts() -> dict:
    raw_df = pd.read_csv(find_raw_data_path())
    df_clean, cleaning_stats = clean_raw_data(raw_df)
    df = build_batch_features(df_clean)

    df = df.sort_values(["waktu_dt", "petak"]).reset_index(drop=True)
    mask_train = (df["waktu_dt"] >= "2025-08-01 00:00:00") & (df["waktu_dt"] <= "2025-09-30 23:59:59")
    mask_val = (df["waktu_dt"] >= "2025-10-01 00:00:00") & (df["waktu_dt"] <= "2025-10-14 23:59:59")
    mask_test = (df["waktu_dt"] >= "2025-10-15 00:00:00") & (df["waktu_dt"] <= "2025-10-31 23:59:59")

    train_df = df.loc[mask_train].reset_index(drop=True)
    val_df = df.loc[mask_val].reset_index(drop=True)
    test_df = df.loc[mask_test].reset_index(drop=True)

    X_train = train_df[FEATURE_COLUMNS].astype(np.float32)
    y_train = train_df["perlu_siram"].astype(int)
    X_val = val_df[FEATURE_COLUMNS].astype(np.float32)
    y_val = val_df["perlu_siram"].astype(int)

    val_rf = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=2026,
        n_jobs=-1,
    ).fit(X_train, y_train)

    val_hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        l2_regularization=1.0,
        max_iter=250,
        min_samples_leaf=15,
        random_state=2026,
    ).fit(X_train, y_train)

    th_rf, th_hgb, th_ens = 0.61, 0.47, 0.54

    pred_val_base = (val_df["kelembaban_tanah_pct"] < 35.0).astype(int)
    prob_val_rf = val_rf.predict_proba(X_val)[:, 1]
    prob_val_hgb = val_hgb.predict_proba(X_val)[:, 1]
    prob_val_ens = 0.5 * prob_val_rf + 0.5 * prob_val_hgb

    metrics_val = {
        "f1_baseline": round(float(f1_score(y_val, pred_val_base)), 4),
        "f1_ml_rf": round(float(f1_score(y_val, (prob_val_rf >= th_rf).astype(int))), 4),
        "f1_ml_hgb": round(float(f1_score(y_val, (prob_val_hgb >= th_hgb).astype(int))), 4),
        "f1_ml_ensemble": round(float(f1_score(y_val, (prob_val_ens >= th_ens).astype(int))), 4),
    }

    X_trainval = pd.concat([train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS]], axis=0).astype(np.float32).reset_index(drop=True)
    y_trainval = pd.concat([train_df["perlu_siram"], val_df["perlu_siram"]], axis=0).astype(int).reset_index(drop=True)
    X_test = test_df[FEATURE_COLUMNS].astype(np.float32)
    y_test = test_df["perlu_siram"].astype(int)

    final_rf = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=2026,
        n_jobs=-1,
    ).fit(X_trainval, y_trainval)

    final_hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        l2_regularization=1.0,
        max_iter=250,
        min_samples_leaf=15,
        random_state=2026,
    ).fit(X_trainval, y_trainval)

    prob_test_rf = final_rf.predict_proba(X_test)[:, 1]
    prob_test_hgb = final_hgb.predict_proba(X_test)[:, 1]
    prob_test_ens = 0.5 * prob_test_rf + 0.5 * prob_test_hgb

    pred_base = (test_df["kelembaban_tanah_pct"] < 35.0).astype(int)
    pred_rf = (prob_test_rf >= th_rf).astype(int)
    pred_hgb = (prob_test_hgb >= th_hgb).astype(int)
    pred_ens = (prob_test_ens >= th_ens).astype(int)

    crop_metrics = []
    for crop_name in ["cabai", "tomat", "selada"]:
        idx_c = test_df["tanaman"] == crop_name
        crop_metrics.append(
            {
                "Tanaman": crop_name.capitalize(),
                "Sampel Uji": int(idx_c.sum()),
                "Positif (Perlu Siram)": int(y_test[idx_c].sum()),
                "F1 Baseline (<35%)": round(float(f1_score(y_test[idx_c], pred_base[idx_c], zero_division=0)), 4),
                "F1 Random Forest": round(float(f1_score(y_test[idx_c], pred_rf[idx_c], zero_division=0)), 4),
                "F1 Ensemble": round(float(f1_score(y_test[idx_c], pred_ens[idx_c], zero_division=0)), 4),
            }
        )

    imp_df = (
        pd.DataFrame({"Fitur": FEATURE_COLUMNS, "Importance": final_rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(12)
    )
    top_features = [{"Fitur": r["Fitur"], "Importance": float(r["Importance"])} for _, r in imp_df.iterrows()]

    metrics_test = {
        "f1_baseline": round(float(f1_score(y_test, pred_base)), 4),
        "prec_baseline": round(float(precision_score(y_test, pred_base, zero_division=0)), 4),
        "rec_baseline": round(float(recall_score(y_test, pred_base, zero_division=0)), 4),
        "acc_baseline": round(float(accuracy_score(y_test, pred_base)), 4),
        "f1_ml_rf": round(float(f1_score(y_test, pred_rf)), 4),
        "prec_ml_rf": round(float(precision_score(y_test, pred_rf, zero_division=0)), 4),
        "rec_ml_rf": round(float(recall_score(y_test, pred_rf, zero_division=0)), 4),
        "acc_ml_rf": round(float(accuracy_score(y_test, pred_rf)), 4),
        "f1_ml_hgb": round(float(f1_score(y_test, pred_hgb)), 4),
        "f1_ml_ensemble": round(float(f1_score(y_test, pred_ens)), 4),
        "prec_ml_ensemble": round(float(precision_score(y_test, pred_ens, zero_division=0)), 4),
        "rec_ml_ensemble": round(float(recall_score(y_test, pred_ens, zero_division=0)), 4),
        "acc_ml_ensemble": round(float(accuracy_score(y_test, pred_ens)), 4),
        "cm_baseline": confusion_matrix(y_test, pred_base).tolist(),
        "cm_rf": confusion_matrix(y_test, pred_rf).tolist(),
        "cm_ensemble": confusion_matrix(y_test, pred_ens).tolist(),
        "crop_metrics": crop_metrics,
        "top_features": top_features,
    }

    bundel = {
        "model_version": "1.0.0",
        "model_rf": final_rf,
        "model_hgb": final_hgb,
        "threshold_rf": th_rf,
        "threshold_hgb": th_hgb,
        "threshold_ensemble": th_ens,
        "features": FEATURE_COLUMNS,
        "tanaman_dict": TANAMAN_DICT,
        "tanah_dict": TANAH_DICT,
        "ambang_agronomi": AMBANG_AGRONOMI,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "cleaning_stats": cleaning_stats,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundel, MODEL_PATH, compress=3)

    metadata = {
        "project": "Kasus 07 - Irigasi Pintar Berbasis IoT",
        "model_version": "1.0.0",
        "primary_model": "RandomForestClassifier",
        "secondary_model": "Ensemble (RandomForestClassifier + HistGradientBoostingClassifier)",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sklearn_version": sklearn.__version__,
        "primary_metric": "F1-Score",
        "threshold": {
            "random_forest": th_rf,
            "hist_gradient_boosting": th_hgb,
            "ensemble": th_ens,
            "business_rationale": (
                "Threshold 0.61 dipilih dari validation set untuk menyeimbangkan "
                "precision tinggi (mencegah pemborosan air dan busuk akar) serta recall "
                "(mencegah tanaman stres kekeringan)."
            ),
        },
        "input_features": FEATURE_COLUMNS,
        "data_split": {
            "method": "time-based split",
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "data_cleaning": cleaning_stats,
        "metrics_validation": metrics_val,
        "metrics_test": {
            "f1_baseline": metrics_test["f1_baseline"],
            "f1_random_forest": metrics_test["f1_ml_rf"],
            "precision_random_forest": metrics_test["prec_ml_rf"],
            "recall_random_forest": metrics_test["rec_ml_rf"],
            "accuracy_random_forest": metrics_test["acc_ml_rf"],
            "f1_hist_gradient_boosting": metrics_test["f1_ml_hgb"],
            "f1_ensemble": metrics_test["f1_ml_ensemble"],
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundel


if __name__ == "__main__":
    result = train_and_save_artifacts()
    print("Artifacts saved: artifacts/model.joblib, artifacts/metadata.json")
    print("Validation F1:", result["metrics_val"])
    print("Test F1 RF:", result["metrics_test"]["f1_ml_rf"], "| Baseline:", result["metrics_test"]["f1_baseline"])
