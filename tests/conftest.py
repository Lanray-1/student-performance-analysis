import numpy as np
import pandas as pd
import pytest

from src.config import NON_FEATURE_COLS


@pytest.fixture
def sample_raw_df():
    n = 20
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "student_id": range(1, n + 1),
        "age": rng.integers(18, 28, n),
        "gender": rng.choice(["Female", "Male", "Other"], n),
        "major": rng.choice(["Computer Science", "Biology", "Arts", "Business", "Engineering", "Psychology"], n),
        "study_hours_per_day": rng.uniform(0, 12, n).round(2),
        "social_media_hours": rng.uniform(0, 5, n).round(2),
        "netflix_hours": rng.uniform(0, 4, n).round(2),
        "part_time_job": rng.choice(["No", "Yes"], n),
        "attendance_percentage": rng.uniform(40, 100, n).round(1),
        "sleep_hours": rng.uniform(4, 12, n).round(1),
        "diet_quality": rng.choice(["Good", "Fair", "Poor"], n),
        "exercise_frequency": rng.integers(0, 8, n),
        "parental_education_level": rng.choice(["Bachelor", "Master", "High School", "PhD", "Some College"], n),
        "internet_quality": rng.choice(["High", "Medium", "Low"], n),
        "mental_health_rating": rng.uniform(1, 10, n).round(1),
        "extracurricular_participation": rng.choice(["Yes", "No"], n),
        "previous_gpa": rng.uniform(1.64, 4.0, n).round(2),
        "semester": rng.integers(1, 9, n),
        "stress_level": rng.uniform(1, 10, n).round(1),
        "dropout_risk": ["Yes", "Yes"] + ["No"] * (n - 2),
        "social_activity": rng.integers(0, 6, n),
        "screen_time": rng.uniform(0.3, 21, n).round(1),
        "study_environment": rng.choice(["Library", "Dorm", "Cafe", "Quiet Room", "Co-Learning Group"], n),
        "access_to_tutoring": rng.choice(["Yes", "No"], n),
        "family_income_range": rng.choice(["Medium", "High", "Low"], n),
        "parental_support_level": rng.integers(1, 11, n),
        "motivation_level": rng.integers(1, 11, n),
        "exam_anxiety_score": rng.integers(5, 11, n),
        "learning_style": rng.choice(["Visual", "Auditory", "Kinesthetic", "Reading"], n),
        "time_management_score": rng.uniform(1, 10, n).round(1),
        "exam_score": rng.integers(36, 101, n),
        "grade_tier": rng.choice(["A", "B", "C", "D"], n),
    })
    return df


@pytest.fixture
def non_feature_cols():
    return NON_FEATURE_COLS