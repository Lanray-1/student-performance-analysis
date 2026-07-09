import numpy as np
import pandas as pd

from src.config import NON_FEATURE_COLS
from src.preprocessing import (
    build_feature_matrix,
    engineer_features,
    fit_transformers,
    get_column_types,
    split_data,
    transform_features,
)


class TestBuildFeatureMatrix:
    def test_drops_non_feature_cols(self, sample_raw_df):
        X, y_reg, y_clf = build_feature_matrix(sample_raw_df)
        for col in NON_FEATURE_COLS:
            assert col not in X.columns

    def test_grade_tier_excluded(self, sample_raw_df):
        X, y_reg, y_clf = build_feature_matrix(sample_raw_df)
        assert "grade_tier" not in X.columns

    def test_y_clf_is_binary_encoded(self, sample_raw_df):
        _, _, y_clf = build_feature_matrix(sample_raw_df)
        assert set(y_clf.unique()).issubset({0, 1})
        assert y_clf.dtype in (int, "int64")

    def test_y_reg_matches_exam_score(self, sample_raw_df):
        _, y_reg, _ = build_feature_matrix(sample_raw_df)
        pd.testing.assert_series_equal(y_reg, sample_raw_df["exam_score"], check_names=False)


class TestGetColumnTypes:
    def test_splits_numeric_and_categorical(self, sample_raw_df):
        X, _, _ = build_feature_matrix(sample_raw_df)
        numeric_cols, categorical_cols = get_column_types(X)
        assert "previous_gpa" in numeric_cols
        assert "gender" in categorical_cols
        assert set(numeric_cols).isdisjoint(set(categorical_cols))

    def test_all_columns_accounted_for(self, sample_raw_df):
        X, _, _ = build_feature_matrix(sample_raw_df)
        numeric_cols, categorical_cols = get_column_types(X)
        assert set(numeric_cols) | set(categorical_cols) == set(X.columns)

class TestSplitData:
    def test_split_preserves_row_count(self, sample_raw_df):
        X, y_reg, y_clf = build_feature_matrix(sample_raw_df)
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(X, y_reg, y_clf)
        assert len(X_train) + len(X_test) == len(X)

    def test_split_preserves_total_positive_count(self, sample_raw_df):
        X, y_reg, y_clf = build_feature_matrix(sample_raw_df)
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(X, y_reg, y_clf)
        assert y_clf_train.sum() + y_clf_test.sum() == y_clf.sum()


class TestFitAndTransformFeatures:
    def test_transform_output_has_expected_columns(self, sample_raw_df):
        X, _, _ = build_feature_matrix(sample_raw_df)
        numeric_cols, categorical_cols = get_column_types(X)
        encoder, scaler = fit_transformers(X, numeric_cols, categorical_cols)
        X_processed = transform_features(X, encoder, scaler, numeric_cols, categorical_cols)

        assert set(numeric_cols).issubset(set(X_processed.columns))
        for col in categorical_cols:
            assert any(c.startswith(col + "_") for c in X_processed.columns)

    def test_scaled_numeric_cols_have_zero_mean_on_train(self, sample_raw_df):
        X, _, _ = build_feature_matrix(sample_raw_df)
        numeric_cols, categorical_cols = get_column_types(X)
        encoder, scaler = fit_transformers(X, numeric_cols, categorical_cols)
        X_processed = transform_features(X, encoder, scaler, numeric_cols, categorical_cols)
        assert np.allclose(X_processed[numeric_cols].mean(), 0, atol=1e-8)

    def test_unknown_category_handled_gracefully(self, sample_raw_df):
        X, _, _ = build_feature_matrix(sample_raw_df)
        numeric_cols, categorical_cols = get_column_types(X)
        encoder, scaler = fit_transformers(X, numeric_cols, categorical_cols)

        X_new = X.copy()
        X_new.loc[X_new.index[0], "gender"] = "Nonbinary"  # unseen category
        X_processed = transform_features(X_new, encoder, scaler, numeric_cols, categorical_cols)
        gender_cols = [c for c in X_processed.columns if c.startswith("gender_")]
        assert (X_processed.loc[X_new.index[0], gender_cols] == 0).all()


class TestEngineerFeatures:
    def test_adds_expected_columns(self, sample_raw_df):
        result = engineer_features(sample_raw_df)
        assert "wellness_score" in result.columns
        assert "distraction_hours" in result.columns
        assert "study_efficiency" in result.columns

    def test_distraction_hours_is_sum(self, sample_raw_df):
        result = engineer_features(sample_raw_df)
        expected = sample_raw_df["social_media_hours"] + sample_raw_df["netflix_hours"]
        pd.testing.assert_series_equal(result["distraction_hours"], expected, check_names=False)

    def test_wellness_score_in_expected_range(self, sample_raw_df):
        result = engineer_features(sample_raw_df)
        # each component is normalized to roughly 0-1 before averaging, so result should be bounded
        assert (result["wellness_score"] >= 0).all()
        assert (result["wellness_score"] <= 1).all()

    def test_does_not_mutate_input(self, sample_raw_df):
        original = sample_raw_df.copy()
        engineer_features(sample_raw_df)
        pd.testing.assert_frame_equal(sample_raw_df, original)