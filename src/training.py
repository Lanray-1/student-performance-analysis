import numpy as np
import mlflow
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBRegressor, XGBClassifier


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_xgb_regressor(X_train, y_train, **kwargs):
    model = XGBRegressor(n_estimators=200, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_xgb_classifier(X_train, y_train, **kwargs):
    model = XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", **kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_regression(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def evaluate_classification(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def log_run(run_name: str, params: dict, metrics: dict, model, model_type: str = "sklearn"):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(scalar_metrics)

        if "confusion_matrix" in metrics:
            mlflow.log_dict({"confusion_matrix": metrics["confusion_matrix"]}, "confusion_matrix.json")

        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
