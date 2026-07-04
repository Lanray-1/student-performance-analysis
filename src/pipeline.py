# src/pipeline.py
import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_DIR  # confirm MODELS_DIR exists in config
from src.database import load_dataframe, run_query
from src.preprocessing import (
    build_feature_matrix,
    get_column_types,
    split_data,
    fit_transformers,
    transform_features,
    save_transformers,
)


def main():
    raw = pd.read_csv(RAW_DATA_PATH)
    load_dataframe(raw)

    df = run_query("SELECT * FROM students;")

    # TODO: cleaning step goes here — currently unclear where this logic lives.
    # If it's in 03_preprocessing.ipynb only, it needs to become a real function
    # (e.g. clean_data(df) -> df) before this pipeline is complete.
    cleaned = df  # placeholder until cleaning function confirmed
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)

    X, y_reg, y_clf = build_feature_matrix(cleaned)
    numeric_cols, categorical_cols = get_column_types(X)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(
        X, y_reg, y_clf
    )

    encoder, scaler = fit_transformers(X_train, numeric_cols, categorical_cols)
    save_transformers(encoder, scaler, MODELS_DIR)

    X_train_processed = transform_features(X_train, encoder, scaler, numeric_cols, categorical_cols)
    X_test_processed = transform_features(X_test, encoder, scaler, numeric_cols, categorical_cols)

    # feature selection (VIF/RF importance -> DROP_COLS) not yet extracted from
    # notebook 04 into src/features.py — currently a no-op (DROP_COLS = []).
    # Until that's resolved, this pipeline stops at processed features, not
    # selected_features.json.

    return X_train_processed, X_test_processed, y_reg_train, y_reg_test, y_clf_train, y_clf_test


if __name__ == "__main__":
    main()
    