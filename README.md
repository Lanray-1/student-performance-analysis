# Student Success Analytics Platform

End-to-end ML system analyzing student performance data: exploratory analysis, SQL-backed storage, feature engineering, predictive modeling (regression + classification), explainability, and a served API + dashboard.

## Dataset

80,000 student records, 31 raw columns + 4 engineered features (35 total).

- **Regression target:** `exam_score` (continuous, 0–100, left-skewed with a ceiling at 100)
- **Classification target:** `dropout_risk` (binary, severely imbalanced — ~98% No / ~2% Yes)

Engineered features (built in `01_eda.ipynb`):
- `wellness_score` — composite of mental health, sleep, stress, exercise
- `distraction_hours` — social media + Netflix hours combined
- `study_efficiency` — study hours relative to distraction hours
- `grade_tier` — categorical bucket derived directly from `exam_score`. **Reporting/EDA use only — excluded from all model features via `NON_FEATURE_COLS` in `src/config.py`, since it's a deterministic function of the regression target and would leak.**

## Stack

SQLite · FastAPI · MLflow · Streamlit · scikit-learn / XGBoost · SMOTE (class imbalance) · SHAP (explainability) · pytest · Docker · GitHub Actions

## Project structure

```
notebooks/        sequential, numbered analysis notebooks (01_eda, 02_sql_layer, ...)
src/               reusable logic (config, database, preprocessing, features, training, prediction, explainability)
api/               FastAPI service exposing trained models
dashboard/         Streamlit frontend
tests/             unit tests, mirrors src/ structure
data/raw/          original CSV (gitignored — not redistributed)
data/processed/    cleaned exports, SQLite DB, EDA figures
models/            serialized trained model artifacts (gitignored)
```

## Progress

- [x] `01_eda.ipynb` — exploratory analysis, feature engineering, leakage check on `grade_tier`
- [x] `02_sql_layer.ipynb` — load cleaned data into SQLite, query layer (`src/database.py`)
- [x] `03_preprocessing.ipynb` — encoding, scaling, train/test split
- [x] `04_feature_engineering.ipynb`
- [ ] `05_training.ipynb` — regression + classification models, SMOTE for imbalance
- [ ] `06_explainability.ipynb` — SHAP
- [ ] FastAPI service
- [ ] Streamlit dashboard
- [ ] Docker + CI/CD

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in numbered order from `notebooks/`. `src/` is imported via `sys.path` insertion at the top of each notebook (see `02_sql_layer.ipynb` for the pattern).
