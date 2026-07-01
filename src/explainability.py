import shap
import pandas as pd
import matplotlib.pyplot as plt


def get_shap_explainer(model, X_train: pd.DataFrame):
    """
    Return a SHAP TreeExplainer.
    Intended for tree-based models (XGBoost) used in Notebook 07.
    """
    return shap.TreeExplainer(model)


def get_shap_values(explainer, X: pd.DataFrame):
    """Compute SHAP values for the provided dataset."""
    return explainer(X)


def plot_summary(shap_values, X: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_bar(shap_values, X: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_waterfall(shap_values, idx: int = 0, save_path: str = None):
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[idx], show=False)
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()
