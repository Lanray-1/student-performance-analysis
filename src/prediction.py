import json

import joblib
import pandas as pd

from src.config import BASE_DIR, MODELS_DIR, NON_FEATURE_COLS

from src.preprocessing import get_column_types, transform_features, engineer_features
PROCESSED_DIR = BASE_DIR / "data" / "processed"

_encoder = joblib.load(MODELS_DIR / "encoder.joblib")
_scaler = joblib.load(MODELS_DIR / "scaler.joblib")
_exam_score_model = joblib.load(MODELS_DIR / "exam_score_model.joblib")
_dropout_risk_model = joblib.load(MODELS_DIR / "dropout_risk_model.joblib")

with open(PROCESSED_DIR / "selected_features.json") as f:
    _selected_features = json.load(f)


def _prepare_input(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    df = engineer_features(df)
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns], errors="ignore")
    numeric_cols, categorical_cols = get_column_types(X)
    X_processed = transform_features(X, _encoder, _scaler, numeric_cols, categorical_cols)

    missing = [c for c in _selected_features if c not in X_processed.columns]
    if missing:
        raise ValueError(f"Input missing expected features after transform: {missing}")

    return X_processed[_selected_features]

def predict_exam_score(raw: dict) -> float:
    X = _prepare_input(raw)
    return float(_exam_score_model.predict(X[["previous_gpa"]])[0])


def predict_dropout_risk(raw: dict) -> dict:
    X = _prepare_input(raw)
    prob = float(_dropout_risk_model.predict_proba(X)[0][1])
    return {"dropout_probability": prob, "dropout_risk": "Yes" if prob >= 0.5 else "No"}