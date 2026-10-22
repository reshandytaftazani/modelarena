# How to hand this project to Codex CLI

1. Extract the archive into an empty `ModelArena` directory.
2. Open Windows PowerShell in that directory.
3. Run `codex` (assuming Codex CLI is already installed and authenticated).
4. Paste the following prompt:

Read `AGENTS.md` and `docs/PROJECT_SPEC.md` in full. We are in PLANNING ONLY mode. Do not write implementation code, install packages, or scaffold the app. First audit the specification for ambiguities, infeasible scope, data leakage risks, and Windows/CPU constraints. Then write `docs/ARCHITECTURE.md` and `docs/IMPLEMENTATION_PLAN.md`. The plan must decompose all P0 requirements into small ordered tasks with dependencies, estimated effort, specific files/modules, acceptance criteria, tests, and milestone gates across 28 days. Include job persistence and cancellation, holdout isolation, fold-local MLP early stopping, and reproducibility. Separate P1/P2 stretch goals. Finally summarize architectural decisions, top five risks, unresolved questions, and stop for my approval. Do not start implementation.

5. Review the two generated planning docs before asking Codex to implement the first task.
