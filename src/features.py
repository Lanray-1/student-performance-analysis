import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def compute_vif(X_numeric: pd.DataFrame) -> pd.DataFrame:
    X_const = add_constant(X_numeric)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_numeric.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_const.values, i + 1)
        for i in range(len(X_numeric.columns))
    ]
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def get_rf_importances(X_train: pd.DataFrame, y_train, task: str = "regression", random_state: int = 42) -> pd.Series:
    if task == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    else:
        model = RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1, class_weight="balanced"
        )
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False)
