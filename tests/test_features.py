import numpy as np
import pandas as pd

from src.features import compute_vif, get_rf_importances


class TestComputeVIF:
    def test_uncorrelated_features_have_low_vif(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
        })
        result = compute_vif(X)
        assert (result["VIF"] < 5).all()

    def test_perfectly_collinear_features_have_high_vif(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=200)
        X = pd.DataFrame({"a": a, "b": a * 2 + 1})
        result = compute_vif(X)
        assert (result["VIF"] > 100).all()

    def test_output_shape_and_columns(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = compute_vif(X)
        assert set(result.columns) == {"feature", "VIF"}
        assert len(result) == len(X.columns)

    def test_sorted_descending(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=200)
        X = pd.DataFrame({
            "independent": rng.normal(size=200),
            "collinear": a * 3,
            "base": a,
        })
        result = compute_vif(X)
        assert result["VIF"].is_monotonic_decreasing


class TestGetRFImportances:
    def test_regression_importances_sum_to_one(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "x1": rng.normal(size=300),
            "x2": rng.normal(size=300),
        })
        y = X["x1"] * 5 + rng.normal(scale=0.1, size=300)
        result = get_rf_importances(X, y, task="regression")
        assert np.isclose(result.sum(), 1.0, atol=1e-6)

    def test_dominant_feature_ranks_first(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "informative": rng.normal(size=300),
            "noise": rng.normal(size=300),
        })
        y = X["informative"] * 10 + rng.normal(scale=0.01, size=300)
        result = get_rf_importances(X, y, task="regression")
        assert result.index[0] == "informative"

    def test_classification_task(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "x1": rng.normal(size=300),
            "x2": rng.normal(size=300),
        })
        y = (X["x1"] > 0).astype(int)
        result = get_rf_importances(X, y, task="classification")
        assert result.index[0] == "x1"

    def test_returns_series_indexed_by_columns(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
        y = rng.normal(size=50)
        result = get_rf_importances(X, y)
        assert isinstance(result, pd.Series)
        assert set(result.index) == {"a", "b"}

    def test_deterministic_with_fixed_seed(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
        y = rng.normal(size=100)
        r1 = get_rf_importances(X, y, random_state=42)
        r2 = get_rf_importances(X, y, random_state=42)
        pd.testing.assert_series_equal(r1, r2)