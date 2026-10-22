# Codex instructions — ModelArena

## Source of truth
Read `docs/PROJECT_SPEC.md` completely before proposing work. Treat it as product requirements. If requirements conflict or are unclear, identify the conflict and propose a decision; do not silently discard P0 requirements.

## Planning gate — mandatory
This repository starts in PLANNING ONLY mode. Do not implement application code, install dependencies, scaffold an app, or modify anything beyond planning documentation until the user explicitly approves the implementation plan. You may inspect files and write `docs/ARCHITECTURE.md` and `docs/IMPLEMENTATION_PLAN.md` when requested.

## Technical priorities
1. No test-set leakage, including preprocessing, tuning, and MLP early stopping.
2. Reproducible, audited experiments and clear metrics.
3. CPU-friendly resource bounds and failure recovery.
4. Correct tests before polish.
5. Minimal complexity and no paid AI provider dependencies.

## Planning deliverables
- `docs/ARCHITECTURE.md`: components, API boundaries, job lifecycle, storage, data flow, ML protocol, security, tradeoffs, Windows/CPU considerations.
- `docs/IMPLEMENTATION_PLAN.md`: 28-day schedule, ordered small tasks with dependencies, file paths, acceptance tests, commands, estimated effort, risks, decision log, P0/P1/P2 gates.
- A short summary of open questions and a clear request for approval before implementation.

## Engineering rules after approval
Use typed interfaces, deterministic seeds where supported, dependency locking, meaningful logging, unit/integration tests, and concise docs. Never run training on the FastAPI request thread. Never load untrusted serialized models. Never introduce cloud or paid LLM requirements. Do not claim an experiment succeeded without executing its tests. Keep user data/artifacts out of Git.
