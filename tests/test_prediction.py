import pandas as pd
import pytest

from src.config import RAW_DATA_PATH, NON_FEATURE_COLS
from src.prediction import predict_exam_score, predict_dropout_risk


@pytest.fixture
def sample_rows():
    df = pd.read_csv(RAW_DATA_PATH)
    rows = df.iloc[:5].drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    return rows.to_dict(orient="records")


def test_predict_exam_score_returns_float_in_range(sample_rows):
    for row in sample_rows:
        score = predict_exam_score(row)
        assert isinstance(score, float)
        assert 0 <= score <= 100


def test_predict_dropout_risk_returns_valid_probability(sample_rows):
    for row in sample_rows:
        result = predict_dropout_risk(row)
        assert 0 <= result["dropout_probability"] <= 1
        assert result["dropout_risk"] in {"Yes", "No"}

def test_missing_required_column_raises(sample_rows):
    bad_row = sample_rows[0].copy()
    del bad_row["previous_gpa"]
    with pytest.raises(ValueError, match="previous_gpa"):
        predict_exam_score(bad_row)