# ARCHITECTURE.md — Student Success Analytics

This file is append-only. When the pipeline changes, add a new dated
section below — don't rewrite history.

## Folder Responsibilities

```
data/        Raw and processed datasets (gitignored, except small samples)
notebooks/   Exploration only — finalized logic moves to src/
src/         Production code: preprocessing, features, training, inference
  features/  Shared transforms used by both training and inference paths
  models/    Model definitions / training scripts
  api/       FastAPI inference service
tests/       pytest suite
docs/        diagrams/, screenshots/, evaluation/, reports/
mlruns/      MLflow local tracking store (gitignored)
```

## Data Flow (current)

```
CSV
  │
  ▼
Preprocessing (src/features/)
  │
  ▼
Feature Engineering
  │
  ▼
Train/Test Split (stratified on dropout target)
  │
  ▼
Model Training (src/models/) ──► MLflow (params, metrics, artifacts)
  │
  ▼
Evaluation (cross-validation, SHAP)
  │
  ▼
FastAPI inference service (src/api/) ──► loads model from MLflow registry
  │
  ▼
Streamlit dashboard ──► calls FastAPI
```

## Model Pipeline Notes

- Two targets, two separate pipelines sharing the same preprocessing step:
  exam score (regression) and dropout risk (classification).
- SMOTE applied only inside training folds for the classification target.
- Models are not committed to git — MLflow's artifact store is the single
  source of truth for trained model binaries. `models/` is gitignored.

## Current Architecture

Containerized FastAPI inference service + Streamlit dashboard, both built
and passing CI. GitHub Actions runs the test suite and Docker build on
every push to main. Model binaries are committed directly to git under
`models/` (see 2026-07-13 entry below — this reverses the original
MLflow-registry-only plan). MLflow is still used for experiment tracking
during training, but is not the deployment path.

```
FastAPI service (containerized)
        │
        ▼
GitHub Actions CI/CD ──► run tests, build image, verify on push to main
        │
        ▼
Streamlit dashboard ──► calls FastAPI for predictions
```
---

## Dated Evolution

### 2026-06-28 — Initial pipeline
CSV → Preprocessing → Random Forest baseline (both targets) → manual
evaluation in notebooks. No MLflow tracking yet at this point — added
shortly after.

### 2026-07-13 — Model artifacts committed to git, not MLflow-only

Reversing the 2026-06-28 decision to use MLflow's artifact store as the
sole source of truth for model binaries. `models/*.joblib` (exam_score
model, dropout_risk model, encoder, scaler) are now committed directly
to git.

Reason: this is a portfolio project meant to be cloned and run
end-to-end (including in CI and Docker) without requiring a live MLflow
tracking server or registry to be stood up separately. Committing small
`.joblib` artifacts directly removes that external dependency for
reviewers running the repo cold. MLflow is retained for local experiment
tracking during training/tuning, just not as the deployment source of
truth.

Trade-off accepted: git history will grow with binary diffs each time a
model is retrained. Acceptable at current project scale; would need
revisiting (Git LFS or reverting to registry-based deployment) if the
project moved to production or artifacts grew significantly larger.
