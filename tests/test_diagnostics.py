import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.diagnostics import (
    error_by_bins,
    feature_ablation,
    get_learning_curve,
    get_residuals,
    run_cross_validation,
    test_interaction as run_interaction_test,
)

@pytest.fixture
def linear_data():
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })
    y = X["x1"] * 5 + rng.normal(scale=0.5, size=n)
    return X, y


@pytest.fixture
def fitted_linear_model(linear_data):
    X, y = linear_data
    model = LinearRegression().fit(X, y)
    return model, X, y


class TestGetResiduals:
    def test_residuals_shape_matches_input(self, fitted_linear_model):
        model, X, y = fitted_linear_model
        preds, residuals = get_residuals(model, X, y)
        assert len(preds) == len(X)
        assert len(residuals) == len(X)

    def test_residuals_equal_actual_minus_predicted(self, fitted_linear_model):
        model, X, y = fitted_linear_model
        preds, residuals = get_residuals(model, X, y)
        np.testing.assert_allclose(residuals, y.values - preds)

    def test_residuals_near_zero_for_good_fit(self, fitted_linear_model):
        model, X, y = fitted_linear_model
        _, residuals = get_residuals(model, X, y)
        assert abs(residuals.mean()) < 1.0


class TestErrorByBins:
    def test_returns_expected_columns(self, fitted_linear_model):
        model, X, y = fitted_linear_model
        preds, _ = get_residuals(model, X, y)
        result = error_by_bins(
            y, preds, X["x1"],
            bins=[-np.inf, -1, 0, 1, np.inf],
            labels=["low", "mid-low", "mid-high", "high"],
        )
        assert set(result.columns) == {"bucket", "n", "mae", "rmse"}

    def test_n_sums_to_total_rows(self, fitted_linear_model):
        model, X, y = fitted_linear_model
        preds, _ = get_residuals(model, X, y)
        result = error_by_bins(
            y, preds, X["x1"],
            bins=[-np.inf, -1, 0, 1, np.inf],
            labels=["low", "mid-low", "mid-high", "high"],
        )
        assert result["n"].sum() == len(X)


class TestInteraction:
    def test_returns_expected_keys(self, linear_data):
        X, y = linear_data
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        result = run_interaction_test(LinearRegression, X_train, X_test, y_train, y_test, "x1", "x2")
        assert set(result.keys()) == {
            "interaction", "base_r2", "base_rmse",
            "with_interaction_r2", "with_interaction_rmse", "r2_delta",
        }

    def test_interaction_name_format(self, linear_data):
        X, y = linear_data
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        result = run_interaction_test(LinearRegression, X_train, X_test, y_train, y_test, "x1", "x2")
        assert result["interaction"] == "x1_x_x2"

    def test_does_not_mutate_original_data(self, linear_data):
        X, y = linear_data
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        original_cols = list(X_train.columns)

        run_interaction_test(LinearRegression, X_train, X_test, y_train, y_test, "x1", "x2")
        assert list(X_train.columns) == original_cols


class TestLearningCurve:
    def test_returns_expected_keys(self, linear_data):
        X, y = linear_data
        result = get_learning_curve(LinearRegression, X, y, cv=3)
        assert set(result.keys()) == {"train_sizes", "train_rmse_mean", "val_rmse_mean", "val_rmse_std"}

    def test_arrays_same_length(self, linear_data):
        X, y = linear_data
        result = get_learning_curve(LinearRegression, X, y, cv=3)
        n_points = len(result["train_sizes"])
        assert len(result["train_rmse_mean"]) == n_points
        assert len(result["val_rmse_mean"]) == n_points
        assert len(result["val_rmse_std"]) == n_points


class TestCrossValidation:
    def test_returns_expected_keys(self, linear_data):
        X, y = linear_data
        result = run_cross_validation(LinearRegression, X, y, cv=5)
        assert set(result.keys()) == {"rmse_mean", "rmse_std", "r2_mean", "r2_std"}

    def test_rmse_is_positive(self, linear_data):
        X, y = linear_data
        result = run_cross_validation(LinearRegression, X, y, cv=5)
        assert result["rmse_mean"] > 0

    def test_r2_reasonable_for_strong_signal(self, linear_data):
        X, y = linear_data
        result = run_cross_validation(LinearRegression, X, y, cv=5)
        assert result["r2_mean"] > 0.8  # x1 dominates y by construction


class TestFeatureAblation:
    def test_baseline_row_present(self, linear_data):
        X, y = linear_data
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        result = feature_ablation(
            LinearRegression, X_train, y_train, X_test, y_test,
            feature_groups={"group_a": ["x1"], "group_b": ["x2", "x3"]},
        )
        assert "(none — full model)" in result["ablated_group"].values
        baseline_row = result[result["ablated_group"] == "(none — full model)"]
        assert baseline_row["rmse_delta"].iloc[0] == 0.0

    def test_removing_dominant_feature_increases_rmse_most(self, linear_data):
        X, y = linear_data
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        result = feature_ablation(
            LinearRegression, X_train, y_train, X_test, y_test,
            feature_groups={"dominant": ["x1"], "noise": ["x2", "x3"]},
        )
        dominant_delta = result[result["ablated_group"] == "dominant"]["rmse_delta"].iloc[0]
        noise_delta = result[result["ablated_group"] == "noise"]["rmse_delta"].iloc[0]
        assert dominant_delta > noise_delta

    def test_baseline_pinned_first(self, linear_data):
        X, y = linear_data
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        result = feature_ablation(
            LinearRegression, X_train, y_train, X_test, y_test,
            feature_groups={"a": ["x1"], "b": ["x2"], "c": ["x3"]},
        )
        # Baseline row must always be first, regardless of whether any ablated
        # group's delta is negative (i.e. removing it slightly helped).
        assert result.iloc[0]["ablated_group"] == "(none — full model)"

    def test_ablated_rows_sorted_descending_by_delta(self, linear_data):
        X, y = linear_data
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        result = feature_ablation(
            LinearRegression, X_train, y_train, X_test, y_test,
            feature_groups={"a": ["x1"], "b": ["x2"], "c": ["x3"]},
        )
        # Everything after the pinned baseline row should be sorted by impact.
        ablated_deltas = result.iloc[1:]["rmse_delta"]
        assert ablated_deltas.is_monotonic_decreasing