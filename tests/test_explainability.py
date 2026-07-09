import matplotlib
matplotlib.use("Agg")  # non-interactive backend, prevents plt.show() from blocking/erroring in CI

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.explainability import get_shap_explainer, get_shap_values, plot_bar, plot_summary, plot_waterfall


@pytest.fixture
def fitted_tree_model():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "x1": rng.normal(size=100),
        "x2": rng.normal(size=100),
    })
    y = X["x1"] * 3 + rng.normal(scale=0.1, size=100)
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model, X


def test_get_shap_explainer_returns_explainer(fitted_tree_model):
    model, X = fitted_tree_model
    explainer = get_shap_explainer(model, X)
    assert explainer is not None


def test_get_shap_values_returns_values_matching_input_shape(fitted_tree_model):
    model, X = fitted_tree_model
    explainer = get_shap_explainer(model, X)
    shap_values = get_shap_values(explainer, X)
    assert shap_values.values.shape[0] == len(X)
    assert shap_values.values.shape[1] == X.shape[1]


def test_plot_summary_runs_without_error(fitted_tree_model, tmp_path):
    model, X = fitted_tree_model
    explainer = get_shap_explainer(model, X)
    shap_values = get_shap_values(explainer, X)
    save_path = tmp_path / "summary.png"
    plot_summary(shap_values, X, save_path=str(save_path))
    assert save_path.exists()


def test_plot_bar_runs_without_error(fitted_tree_model, tmp_path):
    model, X = fitted_tree_model
    explainer = get_shap_explainer(model, X)
    shap_values = get_shap_values(explainer, X)
    save_path = tmp_path / "bar.png"
    plot_bar(shap_values, X, save_path=str(save_path))
    assert save_path.exists()


def test_plot_waterfall_runs_without_error(fitted_tree_model, tmp_path):
    model, X = fitted_tree_model
    explainer = get_shap_explainer(model, X)
    shap_values = get_shap_values(explainer, X)
    save_path = tmp_path / "waterfall.png"
    plot_waterfall(shap_values, idx=0, save_path=str(save_path))
    assert save_path.exists()