import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NON_FEATURE_COLS, TARGET_REGRESSION, TARGET_CLASSIFICATION


def build_feature_matrix(df: pd.DataFrame):
    X = df.drop(columns=NON_FEATURE_COLS)
    y_reg = df[TARGET_REGRESSION]
    y_clf = (df[TARGET_CLASSIFICATION] == "Yes").astype(int)
    return X, y_reg, y_clf


def get_column_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    return numeric_cols, categorical_cols


def split_data(X, y_reg, y_clf, test_size=0.2, random_state=42):
    return train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        stratify=y_clf,
        random_state=random_state,
    )


def fit_transformers(X_train: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    return encoder, scaler


def transform_features(X: pd.DataFrame, encoder, scaler, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    encoded = encoder.transform(X[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index
    )

    scaled = scaler.transform(X[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols, index=X.index)

    return pd.concat([scaled_df, encoded_df], axis=1)


def save_transformers(encoder, scaler, out_dir):
    joblib.dump(encoder, f"{out_dir}/encoder.joblib")
    joblib.dump(scaler, f"{out_dir}/scaler.joblib")
