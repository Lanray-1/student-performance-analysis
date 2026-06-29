import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_residuals(model, X, y):
    preds = model.predict(X)
    residuals = y.values - preds if hasattr(y, "values") else y - preds
    return preds, residuals


def error_by_bins(y_true: pd.Series, y_pred: np.ndarray, bin_values: pd.Series, bins: list, labels: list) -> pd.DataFrame:
    df = pd.DataFrame({
        "actual": y_true.values if hasattr(y_true, "values") else y_true,
        "predicted": y_pred,
        "bin_val": bin_values.values if hasattr(bin_values, "values") else bin_values,
    })
    df["bucket"] = pd.cut(df["bin_val"], bins=bins, labels=labels)

    rows = []
    for bucket, group in df.groupby("bucket", observed=True):
        rows.append({
            "bucket": bucket,
            "n": len(group),
            "mae": mean_absolute_error(group["actual"], group["predicted"]),
            "rmse": float(np.sqrt(mean_squared_error(group["actual"], group["predicted"]))),
        })
    return pd.DataFrame(rows)


def test_interaction(model_fn, X_train, X_test, y_train, y_test, feat_a: str, feat_b: str):
    base_model = model_fn()
    base_model.fit(X_train, y_train)
    base_r2 = r2_score(y_test, base_model.predict(X_test))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, base_model.predict(X_test))))

    X_train_int = X_train.copy()
    X_test_int = X_test.copy()
    interaction_name = f"{feat_a}_x_{feat_b}"
    X_train_int[interaction_name] = X_train_int[feat_a] * X_train_int[feat_b]
    X_test_int[interaction_name] = X_test_int[feat_a] * X_test_int[feat_b]

    int_model = model_fn()
    int_model.fit(X_train_int, y_train)
    int_r2 = r2_score(y_test, int_model.predict(X_test_int))
    int_rmse = float(np.sqrt(mean_squared_error(y_test, int_model.predict(X_test_int))))

    return {
        "interaction": interaction_name,
        "base_r2": base_r2,
        "base_rmse": base_rmse,
        "with_interaction_r2": int_r2,
        "with_interaction_rmse": int_rmse,
        "r2_delta": int_r2 - base_r2,
    }


def get_learning_curve(model_fn, X, y, train_sizes=np.linspace(0.1, 1.0, 8), cv=5):
    model = model_fn()
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=kfold,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    return {
        "train_sizes": sizes,
        "train_rmse_mean": -train_scores.mean(axis=1),
        "val_rmse_mean": -val_scores.mean(axis=1),
        "val_rmse_std": val_scores.std(axis=1),
    }


def run_cross_validation(model_fn, X, y, cv=5) -> dict:
    model = model_fn()
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmse_scores = -cross_val_score(model, X, y, cv=kfold, scoring="neg_root_mean_squared_error", n_jobs=-1)
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2", n_jobs=-1)
    return {
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(),
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
    }


def feature_ablation(model_fn, X_train, y_train, X_test, y_test, feature_groups: dict) -> pd.DataFrame:
    full_model = model_fn()
    full_model.fit(X_train, y_train)
    full_rmse = float(np.sqrt(mean_squared_error(y_test, full_model.predict(X_test))))

    rows = [{"ablated_group": "(none — full model)", "rmse": full_rmse, "rmse_delta": 0.0}]

    for group_name, cols in feature_groups.items():
        cols_present = [c for c in cols if c in X_train.columns]
        if not cols_present:
            continue
        X_train_ablated = X_train.drop(columns=cols_present)
        X_test_ablated = X_test.drop(columns=cols_present)

        model = model_fn()
        model.fit(X_train_ablated, y_train)
        rmse = float(np.sqrt(mean_squared_error(y_test, model.predict(X_test_ablated))))

        rows.append({"ablated_group": group_name, "rmse": rmse, "rmse_delta": rmse - full_rmse})

    return pd.DataFrame(rows).sort_values("rmse_delta", ascending=False).reset_index(drop=True)
