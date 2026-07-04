# AGENTS.md — Student Success Analytics

## Purpose

ML system predicting student academic performance (exam score regression)
and dropout risk (binary classification) from an 80,000-row, 31-column
student dataset. Built as a portfolio piece demonstrating production-grade
ML engineering: tracked experiments, served models, reproducible pipeline,
CI/CD.

## Project Principles

Prioritize, in order:

1. Correctness over speed.
2. Readability over clever code.
3. Reproducibility over convenience.
4. Production-quality code over notebook hacks.

## Tech Stack

- Python 3.12 (conda env: see CONTEXT.md for current env name)
- pandas, numpy, scikit-learn, XGBoost
- MLflow (experiment tracking + model registry)
- SHAP (explainability)
- imbalanced-learn / SMOTE (class imbalance on dropout target)
- FastAPI (inference service)
- Streamlit (dashboard)
- Docker (containerization)
- GitHub Actions (CI/CD)
- pytest (testing)
- Black + Ruff (formatting/linting)

## Coding Standards

- Type hints required on all function signatures.
- Google-style docstrings on anything public (module-level functions, classes).
- Use `pathlib.Path`, never `os.path`.
- Functions should generally have one responsibility. Split a function when
  doing so improves readability, not to chase a line count.
- No duplicated preprocessing logic between training and inference paths.
  Shared transforms live in `src/features/` and get imported by both.
- Prefer composition over inheritance unless there's a clear is-a relationship.

## Machine Learning Rules

- Never let test data touch fit() — no leakage, including inside
  preprocessing pipelines (use sklearn Pipeline/ColumnTransformer, not
  manual fit-then-split).
- Always validate with cross-validation before reporting a metric as final.
- `random_state=42` everywhere a seed is accepted, for reproducibility.
- Every trained model gets logged to MLflow (params, metrics, artifact) —
  no "just ran it in a notebook" results.
- SMOTE (or other resampling) is applied only inside the training fold,
  never before the train/test split.
- Don't commit raw model binaries to git. Models live in MLflow's artifact
  store. See ARCHITECTURE.md for how that's wired up.
- Feature engineering must be deterministic. Inference must use exactly the
  same preprocessing pipeline as training — no separate "inference version"
  of a transform.

## Performance

- Avoid unnecessary DataFrame copies.
- Vectorize pandas operations where practical.
- Avoid premature optimization unless profiling shows a real bottleneck.

## Testing

- Run `pytest` before considering any feature done.
- New preprocessing/feature functions need at least one test covering the
  shape and dtype of their output, not just "it runs."
- Don't skip writing a test because "it's just a notebook experiment" —
  once it moves into `src/`, it needs coverage.

## Documentation

- Update README.md when adding a feature a recruiter/visitor would care about.
- Update ARCHITECTURE.md (append a dated entry, don't rewrite) when the
  pipeline shape changes.
- Update CONTEXT.md when a task materially changes project status — new
  phase, resolved blocker, model swap, etc. Not every single edit.

## Git

- Don't modify files outside what the task asked for.
- Never overwrite user changes unless explicitly requested.
- Preserve existing comments unless replacing outdated logic.
- Commit logically grouped changes — don't mix a refactor with a new feature
  in one commit.
- Don't touch `notebooks/` outputs unless explicitly asked to.
