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
- [x] `04_feature_engineering.ipynb` — VIF, RF importance, feature selection review
- [x] `05_training.ipynb` — regression + classification models, SMOTE for imbalance
- [x] `06_model_diagnostics.ipynb` — residuals, CV, ablation, model selection
- [x] `07_explainability.ipynb` — SHAP
- [ ] FastAPI service
- [ ] Streamlit dashboard
- [ ] Docker + CI/CD

## Known limitations

**`dropout_risk` is a near-deterministic synthetic rule, not realistic behavioral data.** A depth-2 decision tree achieves perfect classification (100% accuracy/precision/recall/ROC-AUC) on held-out test data using only two features:

```
stress_level > 1.56 SD above mean  AND  motivation_level < 0.69 SD below mean  →  dropout = 1
(every other combination → dropout = 0)
```

XGBoost's perfect scores in `05_training.ipynb` reflect this dataset property, not a breakthrough model — there's no genuine generalization challenge in this target. Reported transparently rather than presented as a result.

**`exam_score` is dominated almost entirely by `previous_gpa`.** A naive single-feature linear baseline (`previous_gpa` only) achieves RMSE 4.155 / R² 0.870 — XGBoost on the full 56-feature set actually performs *worse* (RMSE 4.301 / R² 0.860), and a full linear model on all features performs nearly identically to the naive baseline (RMSE 4.157 vs 4.155) — no formal significance test was run to confirm this difference is not meaningful. The engineered behavioral features (study habits, wellness, distraction) contribute negligible incremental signal over prior academic performance in this dataset.

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in numbered order from `notebooks/`. `src/` is imported via `sys.path` insertion at the top of each notebook (see `02_sql_layer.ipynb` for the pattern).
