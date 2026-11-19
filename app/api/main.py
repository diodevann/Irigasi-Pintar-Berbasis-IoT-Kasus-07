from __future__ import annotations

import json
from typing import Any, Dict
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.features import build_single_feature_row, validate_single_input
from src.train import METADATA_PATH, MODEL_PATH, train_and_save_artifacts

app = FastAPI(
    title="Smart Irrigation IoT API",
    version="1.0.0",
)

_MODEL_CACHE: Dict[str, Any] | None = None


def get_model_bundle() -> Dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if MODEL_PATH.exists():
        try:
            _MODEL_CACHE = joblib.load(MODEL_PATH)
            return _MODEL_CACHE
        except Exception:
            pass

    _MODEL_CACHE = train_and_save_artifacts()
    return _MODEL_CACHE


@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "Smart Irrigation IoT API", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, Any]:
    bundel = get_model_bundle()
    return {
        "status": "ok",
        "model_version": bundel.get("model_version", "1.0.0"),
        "features_count": len(bundel.get("features", [])),
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        get_model_bundle()
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": "Body request bukan JSON yang valid.",
                "errors": ["Format payload harus berupa JSON valid."],
            },
        )

    is_valid, errors, cleaned = validate_single_input(payload)
    if not is_valid:
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": "Masukan sensor tidak valid atau di luar rentang fisik.",
                "errors": errors,
            },
        )

    bundel = get_model_bundle()
    df_row, summary = build_single_feature_row(cleaned)

    prob_rf = float(bundel["model_rf"].predict_proba(df_row)[0, 1])
    prob_hgb = float(bundel["model_hgb"].predict_proba(df_row)[0, 1])

    if cleaned["model_type"] == "ensemble":
        prob_ml = 0.5 * prob_rf + 0.5 * prob_hgb
        threshold = float(bundel["threshold_ensemble"])
        model_used = "Ensemble RF + HistGB"
    else:
        prob_ml = prob_rf
        threshold = float(bundel["threshold_rf"])
        model_used = "Random Forest"

    keputusan_ml = int(prob_ml >= threshold)
    keputusan_baseline = int(float(cleaned["kelembaban_tanah_pct"]) < 35.0)

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "model_used": model_used,
            "model_version": bundel.get("model_version", "1.0.0"),
            "keputusan_ml": keputusan_ml,
            "label_ml": "SIRAM (1)" if keputusan_ml == 1 else "TIDAK PERLU DISIRAM (0)",
            "probabilitas_ml": round(prob_ml, 4),
            "threshold": threshold,
            "keputusan_baseline": keputusan_baseline,
            "label_baseline": "SIRAM (1)" if keputusan_baseline == 1 else "JANGAN SIRAM (0)",
            "ringkasan_agronomi": summary,
        },
    )
