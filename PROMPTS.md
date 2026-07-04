# PROMPTS.md — Reusable Development Prompts

Copy/paste starting points for common requests to a coding agent on this
project. Adapt the specifics, keep the structure.

## Refactor

Refactor this module without changing behavior.
Preserve public APIs.
Add tests if coverage is missing for the refactored paths.

## Review

Review this code for:
- bugs
- data leakage (train/test contamination, target leakage)
- performance issues
- readability
- violations of AGENTS.md rules

## Feature

Implement this feature following AGENTS.md.
Update tests for the new behavior.
Update CONTEXT.md if this changes project status.
Update ARCHITECTURE.md if this changes the pipeline shape.

## Debug

Explain the root cause before proposing a fix.
Don't guess — trace the actual data/control flow that produces the bug.
If multiple plausible causes exist, say so before picking one to fix.
