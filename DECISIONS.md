# DECISIONS.md — Student Success Analytics

Append-only log of significant technical decisions. Don't edit past
entries — if a decision changes, add a new entry noting the reversal
and why.

## 2026-06-28

**Decision:** Use MLflow Model Registry as the source of truth for trained
models, instead of committing pickle files to git.

**Reason:** Versioning, reproducibility, and a clean path to deployment
without bloating the repo with binaries.

**Alternatives considered:** Git LFS, DVC.

**Affected files:** AGENTS.md, ARCHITECTURE.md

**Impact:** Model loading in inference code goes through the MLflow
Registry API instead of a local file path.

**Status:** Accepted.


### 2026-07-13 — Reverse MLflow-only artifact policy; commit models to git

Reverses the 2026-06-28 decision to use MLflow Model Registry as the sole
source of truth instead of committing pickle files to git.

`models/dropout_risk_model.joblib`, `models/exam_score_model.joblib`,
`models/encoder.joblib`, and `models/scaler.joblib` are now committed
directly to git.

Reason: this is a portfolio project intended to be cloned and run
end-to-end — including CI and Docker builds — without standing up a live
MLflow tracking server. Committing the small `.joblib` artifacts removes
that external dependency for anyone evaluating the repo cold. MLflow
remains in use for experiment tracking and comparison during training;
it's no longer the deployment source of truth for the served model.

Trade-off: git history accumulates binary diffs on every retrain.
Acceptable at current scale (four small artifacts); would need Git LFS
or a return to registry-based deployment if the project grew past a
portfolio-scale footprint.