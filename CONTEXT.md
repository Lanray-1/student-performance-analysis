# CONTEXT.md — Student Success Analytics

**Last Updated:** 2026-06-28 18:35 WAT
**Current Branch:** feature/shap-analysis
**Current Milestone:** Week 3
**Conda Environment:** student-success

## Current Phase

Week 3 — Feature Engineering / moving into SHAP

## Completed

- Data cleaning
- Missing value handling
- Baseline Random Forest (both targets: exam score regression, dropout classification)
- Cross-validation harness

## Working On

- SHAP explainability for the dropout classifier

## Next

- FastAPI inference service
- Hyperparameter tuning (RF vs XGBoost comparison)
- Streamlit dashboard

## Known Issues / Blockers

- Class imbalance on dropout target — SMOTE applied in-fold, but still
  need to check if it's actually improving recall on the minority class
  or just overfitting to synthetic samples.
- Missing parent education values — currently imputed with mode, revisit
  once SHAP shows how much this feature actually matters.
- Probability calibration not yet checked — RF probabilities may not be
  well-calibrated out of the box.

## Current Best Models

**Regression (exam score)**
- Model: Random Forest
- RMSE: 8.41

**Classification (dropout risk)**
- Model: Random Forest
- F1:
- Recall:
- ROC-AUC:

## Recent Decisions

- Decided against LightGBM for the baseline comparison — sticking with
  RF vs XGBoost for now.
- MLflow Model Registry is the source of truth for trained models, not
  files in `models/`.
- SHAP comes before hyperparameter tuning — want to understand feature
  importance before spending compute on tuning.

## Open Questions

- Does SHAP justify dropping the parent-education feature, or does it
  matter more than the missing-value rate suggests?
- Is XGBoost outperforming RF by enough to justify the added complexity?

## Notes

Waiting to compare XGBoost vs LightGBM before committing to one for the
final pipeline. Don't want to lock in MLflow model registry naming until
that's settled.
