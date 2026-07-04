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

Local training pipeline + MLflow tracking, run manually. No served API or
dashboard yet — see CONTEXT.md for live status.

## Target Architecture

```
FastAPI service (containerized)
        │
        ▼
GitHub Actions CI/CD ──► build, test, push image on merge to main
        │
        ▼
Streamlit dashboard ──► calls FastAPI for predictions
```

Docker, CI/CD, and the dashboard's FastAPI integration are not yet built.
Compare this section against Current Architecture above to see what's
left — don't rewrite this section as things land, just update Current.

---

## Dated Evolution

### 2026-06-28 — Initial pipeline
CSV → Preprocessing → Random Forest baseline (both targets) → manual
evaluation in notebooks. No MLflow tracking yet at this point — added
shortly after.
