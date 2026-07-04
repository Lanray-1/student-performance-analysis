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
