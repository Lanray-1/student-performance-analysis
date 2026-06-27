from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "enhanced_student_habits_performance_dataset.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_student_data.csv"
DB_PATH = BASE_DIR / "data" / "processed" / "students.db"

TABLE_NAME = "students"

TARGET_REGRESSION = "exam_score"
TARGET_CLASSIFICATION = "dropout_risk"
ID_COL = "student_id"
LEAKAGE_COLS = ["grade_tier"]

NON_FEATURE_COLS = [ID_COL, TARGET_REGRESSION, TARGET_CLASSIFICATION] + LEAKAGE_COLS
