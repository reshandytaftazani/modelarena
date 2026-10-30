# Validation evidence

## Day 11 — 2026-10-30

The user requested Day 11, English output, commit and push. Work continued on `feat/day-7-durable-queue` from clean commit `4ea6dbe`. No dependency changes, downloads, holdout evaluations, or user data were added.

- CV comparison verifies the frozen manifest, task, feature configuration, fold identities, primary metric/direction, model seed and resource configuration before mixing candidates. It recalculates fold means and population standard deviations from actual complete folds; failed, timed-out, duplicate or incomplete folds and undefined primary metrics remain unranked. Baseline deltas have positive-is-better direction, are null if Dummy is ineligible or absent, and ties within 1e-8 of a group's best use mean fit time and family name. Timing is separate from score.
- Diagnostics aggregate outer-validation predictions only. Each displayed prediction retains its row and fold identity; full OOF scalar metrics/confusion matrix and classification ROC/PR curves or regression residual/prediction points are explicitly labeled training OOF. Display points and each curve are evenly sampled deterministically, capped by the frozen `max_diagnostic_points` setting (at most 2,000). Fold-mean ranking never uses pooled OOF metrics. Failed or incomplete candidates have no pooled diagnostics. These predictions and labels are private data and are not logged or committed.
- The spawned child returns the comparison and diagnostics alongside candidates in the supervisor's in-memory result. **Durable persistence is not implemented:** the Day 11 persistence acceptance remains pending until Day 12's private MLflow/manifest and atomic-publication work. A completed job does not mean the data survives restart. The existing supervisor enforces the overall job deadline and discards timed-out job output; there is still no independently enforced per-fold native-fit deadline. No partial or timed-out candidate is promoted by the comparison function.
- Test-first: the new CV protocol suite initially failed collection because the comparison interface did not exist; the worker integration then failed because comparison was absent from the child payload. Both passed after implementation. `uv run --locked pytest tests/integration/test_cv_protocol.py tests/integration/test_baseline_cv.py -q`: **32 passed**. `uv run --locked pytest tests/integration/test_classical_suite.py -q`: **7 passed**, including actual Windows spawned children for both tasks. `uv run --locked pytest -q`: **387 passed in 101.55 s**, zero skips. `uv lock --check` and `git diff --check` passed.

G1 remains passed locally; G2–G4 remain pending. These fixture tests are not named demonstration, benchmark, final holdout or release evidence.

## Day 10 — 2026-10-30

The user requested Day 10, English output, commit and push. Work continued on `feat/day-7-durable-queue`; the starting checkout was clean at `c33ffca`. No dependencies, lockfiles, user data, dataset downloads, holdout scoring or frontend work were added.

The closed registry now supplies Dummy, LogisticRegression/Ridge, RandomForest and HistGradientBoosting for the relevant task, with the specified bounded defaults and HGB early stopping disabled. The suite serially evaluates each candidate against the same frozen outer folds using fresh guarded fold-local preprocessing, selects binary probability column 1 and reports fit/score timings and safe fold failures. Failed and partial candidates remain ineligible. Cancellation is checked between folds and candidates; the existing supervisor also handles cancellation during native fits.

The classical worker accepts internally prepared training-only input and verifies its experiment, split, configuration and run metadata against the queued immutable records before spawning. It does not accept a client path or estimator, and the child receives no holdout contents. Source-content provenance still depends on the trusted `prepare_experiment` / `training_from_manifest` boundary, not on reconstructing a manually modified training view. The current supervisor accepts one supplied prepared experiment, fails closed on a different queued job, and returns results only in `last_result` memory. Job completion does **not** mean candidate results were durably published: private source loading, candidate publication, MLflow tracking and leaderboard operations remain future work. The existing job deadline applies; a separate classical per-fold deadline is not implemented.

Verification on Windows and the existing locked environment:

- Test-first: the new registry and suite commands initially failed collection because their new interfaces/modules did not exist. After implementation, the registry and suite each passed.
- The first full regression run found **2 failures, 365 passes**: the new optional input had changed the probe child call signature used by the native-fit cancellation tests. The supervisor now passes that input only for the classical workload; the lifecycle and classical integration suites then passed **26 tests**, zero skips.
- Final `uv run --locked pytest -q`: **369 passed in 146.37 s**, zero skips. Named commands, rerun separately: `uv run --locked pytest tests/unit/test_classical_models.py -q` — **5 passed in 7.01 s**; `uv run --locked pytest tests/integration/test_classical_suite.py -q` — **7 passed in 37.24 s**.
- Real Windows-spawned children completed all four families on both synthetic tasks using five shared folds. Tests also check fresh fold-only preprocessing for all 20 fits, finite model outputs, safe failed-family visibility and ineligibility, deterministic cancellation between candidates, real worker cancellation and a second supervisor blocked by the occupied slot. The frozen-input mismatch is rejected before child spawn. Fixture outcomes are not named-dataset or performance evidence.
- `uv lock --check` passed. `graphify update .` refreshed the ignored AST graph; its labels and semantic documentation may require a later refresh.

G1 remains passed locally; G2–G4 and release remain pending. No model was selected or evaluated on the holdout, no baseline-win or benchmark claim is made, and no P0 requirement was removed.

## Day 9 — 2026-10-27

The user authorized Day 9 implementation, English output, commit and push, and chose the existing `feat/day-7-durable-queue` branch. The clean baseline was `4edf0f2`; `uv run --locked pytest -q` passed **336 tests in 38.20 s**, zero skips. No dependencies, lockfiles, downloads, frontend or holdout evaluation changed. Days 10–28 remain pending.

Delivered:

- Queued cancellation without spawn; running cancellation via a multiprocessing Event, five-second cooperative grace, then verified-child termination/join and two-second kill fallback. KeyboardInterrupt joins the active child, records interruption and preserves waiting jobs.
- SQLite transaction arbitration: cancellation persisted before completion prevents success; an already committed completion stays completed. Noncompleted attempts invalidate candidate status and discard in-memory result payloads, including late cancellation after result receipt.
- Schema v2 adds per-attempt supervisor/child PID and exact creation identity; an atomic v1 migration preserves existing metadata and queued jobs. The child cannot begin its workload before its identity is durably registered and its start gate opens.
- Recovery requires both supervisor and registered child to be verified dead before interruption, generation advancement and slot release. A live supervisor or orphan, stale heartbeat alone, inaccessible identity, legacy ownership or spawn-registration gap cannot authorize new fitting. No automatic orphan termination or force-unlock is provided.
- Explicit whole-CV retry reuses the immutable experiment, configuration and split, creates a new job/attempt, and retains the previous terminal record. Old attempts and stale generations cannot publish. Reopening the repository (the available API-restart boundary) does not stop active execution.

Test-first evidence included failures for cancel arbitration, cooperative exit, KeyboardInterrupt handling, missing recovery/retry contracts, generation fencing, candidate invalidation, late result cancellation and unobservable-child cleanup before implementation. The updated Day 8 crash assertion now expects `interrupted`, as required by the Day 9 lifecycle; allowlisted workload exceptions remain `failed`.

Fresh verification before review: `uv run --locked pytest -q` -> **356 passed in 64.11 s**, zero skips. The 20 new lifecycle tests include a real spawned scikit-learn HistGradientBoosting fit interrupted within ten seconds, cooperative exit without termination, actual supervisor/child termination, no-overlap restart, migration and fail-closed ownership. Synthetic fixtures only; no performance benchmark or named demonstration claim.

Independent read-only review found no Critical issues, one Important issue and no new minors: cancellation polling followed a configurable heartbeat that could reach 60 seconds. A regression using a real native fit and 60-second heartbeat failed with a 12-second timeout before the fix. Cancellation now polls independently at 0.2 seconds and bounds the monitoring sleep even with a five-second poll setting. Final checkout suite: `uv run --locked pytest -q` -> **357 passed in 72.59 s**, zero skips. The native-fit deadline regression passes at both short and maximum heartbeat/poll settings. The reviewer did not independently rerun the suite; its deterministic reproduction and the implementer's RED→GREEN run are separate evidence. Earlier staged export also passed 356 tests before this fix; final staged verification is recorded below when executed.

Review boundaries: future CV/leaderboard/MLflow/bundle integration remains out of Day 9 scope and requires its later acceptance tests. Public documentation was checked by the implementer, not the code reviewer. `uv lock --check` passed without lock changes. `graphify update .` refreshed the ignored AST code graph; semantic documentation nodes were not refreshed, and community labels may need a later relabel.

Final staged-tree verification: exported the exact index with `git checkout-index --all` into a fresh approved temporary directory, then ran `uv run --project <export> --directory <export> --locked pytest -q` using the existing locked environment via `UV_PROJECT_ENVIRONMENT`: **357 passed in 68.68 s**, zero skips. This verifies staged source independently of the checkout, not a fresh dependency installation. Only README's delivery-state wording and this evidence paragraph were added afterward. Explicit source/test/documentation paths were staged; local plans, graph output, generated data and artifacts remain ignored. No merge, PR or release is implied by committing/pushing Day 9.

Execution decisions: use the existing ignored per-day ledger for the daily-heading plan; block on live orphan/unknown ownership rather than automatically killing or guessing. This favors safety over automatic recovery; missing identity requires operator investigation. The historical direct-SQL nullable-generation CHECK minor remains deferred; repository claim validation and recovery fencing do not treat missing ownership as safe. Dataset CV, MLflow/bundle reconciliation, HTTP lifecycle routes and UI remain assigned to later days. G1 remains passed locally; G2–G4 and release acceptance remain pending.

## Day 8 — 2026-10-26

The user authorized Day 8 implementation, English output, commit and push, continuing on the existing checkout and branch `feat/day-7-durable-queue`. No dependency, lockfile, frontend scaffold, network dataset fetch or holdout evaluation was added. Days 9–28 remain unimplemented.

Delivered:

- `apps/api/workers/control.py`: a shared, import-light vocabulary for stages, terminal states and safe error codes, plus a typed workload registry. Stages and error codes are closed allowlists because they are persisted and rendered; free text, paths and exception messages cannot enter them. An unknown workload name or option key is rejected before any process is spawned, so no client-supplied path or import string is ever resolved.
- `apps/api/workers/monitor.py`: the single definition of normalized CPU (100% means all logical processors, so one busy core of twelve is ~8.3%), sampled child RSS/CPU plus system-available memory, a continuous-breach memory-pressure tracker, and `terminate_owned_child`, which compares the `psutil` creation time before signalling and returns `False` for an unverified PID. A recycled PID can therefore never be killed.
- `apps/api/workers/execute.py`: the spawn entrypoint. Module scope imports no numeric library, and `apply_thread_environment` runs before the first numerical import. Each measurement stage enables exactly one T-wide layer — BLAS at T with OpenMP and torch intra-op pinned to one, then torch intra-op at T with BLAS and OpenMP pinned to one — and reports both stages so nesting cannot hide inside one snapshot. A deliberate workload failure emits only the closed code `workload_failed`; the exception text never crosses the process boundary.
- `apps/api/workers/supervisor.py`: `run_once` claims at most one job, spawns exactly one non-daemon child through the `spawn` context, samples it, heartbeats on its own timer independently of child progress, enforces the RSS cap, the continuous system-reserve breach and the job deadline, and publishes one terminal transition. Publication requires the verified attempt owner and attempt id, so a stale supervisor cannot overwrite a result. A cancel request observed while the job is still running ends it as `cancelled` instead of publishing success. Publication degrades rather than escaping: unrenderable telemetry is dropped so the job still reaches a terminal state and releases the slot. The parent never imports a numeric library and never fits.
- `apps/api/database/repository.py`: `lease_generation`, `cancel_requested`, `heartbeat`, `record_progress`, `finish_job` and the ownership guard. Every worker write re-verifies job state, owner token, generation and the singleton slot inside the same short `BEGIN IMMEDIATE` transaction. `finish_job` releases the slot atomically with the terminal state and appends a numbers-only telemetry event to the append-only audit trail; a non-numeric or non-finite telemetry value is rejected rather than stored.
- `scripts/benchmark.py` now imports the shared `normalized_cpu_percent` instead of duplicating the formula, so the probe and the worker cannot drift apart.
- The deferred Day 7 schema minor was re-examined and deliberately **not** changed. Rebuilding the released `jobs` table to add an `IS NOT NULL` clause is a destructive migration outside Day 8's scope; the repository path always writes a validated positive generation, and the constraint belongs with the Day 9 schema work that already touches worker state.

Verification executed on Windows 11, Python 3.12.15, 6 physical / 12 logical processors, using the existing locked environment (`uv 0.13.5`):

| Command / check | Observed result |
|---|---|
| Spawn feasibility probe before implementation | A real spawned child ran and reported a different PID under pytest on Windows |
| Thread-pool probe before implementation | At T=2/4/6, BLAS, OpenMP and torch intra-op were confirmed controllable; the probe exposed that torch's intra-op pool defaults to the physical core count and ignored `OMP_NUM_THREADS`, which the implementation then pinned explicitly |
| New worker suites before implementation | Collection failed with `ModuleNotFoundError: apps.api.workers`, confirming the tests exercised absent code |
| `uv run --locked pytest tests/integration/test_worker_spawn.py -q` | 11 passed, zero skips (10.2–11.5 s) |
| `uv run --locked pytest tests/integration/test_resource_monitoring.py -q` | 15 passed, zero skips (18.3–19.4 s) |
| Both new suites together | 26 passed, zero skips, repeated eight times (26.3–28.8 s) |
| Full regression `uv run --locked pytest -q` | **336 passed, zero skips** in every run; observed wall time 36.9–48.4 s across dozens of consecutive runs after the final code change |
| `uv lock --check` / `git diff --check` | Both exited 0; Git's CRLF normalization notices are not whitespace errors |

Observed real-process evidence, not mocks:

- A completed job's `pid` differs from the supervising process, `parent_pid` equals it, and the persisted lease slot is released with no job left `running`.
- Progress stages (`starting`, `preparing`, `fitting`) are readable from SQLite while the child is still running, not only after it finishes.
- Heartbeat timestamps keep changing while the child emits nothing for 1.5 s, proving heartbeats do not depend on child progress.
- Two supervisors racing for two queued jobs produce a maximum of one `running` row, exactly one claim and one completion; the second job stays queued.
- Database reads from the supervising process stay responsive (mean well under 100 ms) throughout a 1.2 s child fit.
- At T=2/4/6 a real spawned child reports `torch_intra_threads == T`, `torch_interop_threads == 1`, exactly one T-wide pool per stage, and every other pool at one thread.
- A reduced test-only 256 MiB cap with a 512 MiB allocation fails the job as `memory_limit_exceeded`, terminates the child, releases the slot and creates no candidate row. The production cap remains 4 GiB.
- A 1 s job deadline against a 30 s workload is persisted as `timeout` in under 20 s.
- A child exiting with code 7 is persisted as `child_exited` with no partial result.
- A secret string injected into a deliberate workload failure appears in neither the persisted job/audit rows nor the captured logs; only `workload_failed` is stored.
- An unverified PID is never terminated, and a verified owned child is.
- A cancel requested while the job is running now finishes as `cancelled` with `safe_error = cancelled` and no `job_completed` audit event, rather than publishing success for abandoned work. This test was written first and failed with `completed` before the fix.
- Unrenderable telemetry no longer strands a job: with the audit renderer forced to reject a value, the job still reaches `completed`, the slot is released and the telemetry event is simply absent. This test was also written first and failed with the raw `ValueError` escaping publication before the fix, which would have left the job `running` and the queue blocked.

Known limitations of this milestone, stated so they are not mistaken for delivered features:

- The child runs the internal synthetic `probe` workload. Dataset cross-validation inside the spawned child arrives with the classical registry on Days 10–12; no user data, holdout partition or trained artifact is involved in these tests.
- Cancellation is a single check on the heartbeat cadence with no cooperative grace period, no `Event`, and no complete cancel-versus-result arbitration matrix. Queued cancellation already worked from Day 7. Full lifecycle handling, interruption recovery, stale-owner reconciliation and restart behaviour remain Day 9.
- Resource limits are sampled, so a transient RSS spike between samples can be missed. This is a soft cap, not an OS-enforced quota, and it does not justify a final thread or memory policy.
- No MLflow tracking, candidate publication, API transport, dashboard, or holdout evaluation exists yet; the supervisor is a library called by tests, not a running background service.

## Day 7 — 2026-10-26

The user authorized Day 7 implementation, English output, commit and push, and chose the current checkout on a new branch (`feat/day-7-durable-queue`). No dependencies, frontend scaffold, network dataset fetch or holdout evaluation were added. Days 8–28 remain unimplemented.

Delivered:

- Version-one local application SQLite schema, WAL, per-connection foreign keys and FULL synchronous commits. Immutable dataset/configuration/split/experiment records include hashes, frozen run limits, explicit task/positive label, duplicate acknowledgement and a training-only profile. Split JSON preserves outer/inner row identities; feature/target records are not stored. Schema also reserves candidates and the singleton worker lease; this is separate from MLflow.
- Parameterized enqueue/claim/cancel operations with short `BEGIN IMMEDIATE` transactions. Idempotency keys are globally unique and compare immutable experiment payload identity; retries return the original job even after reopening or filling the queue. Waiting queue defaults to ten, with lower frozen limits supported. Claims are FIFO, create unique attempts and bind the owner/generation and singleton slot in the same transaction as their audit event. A unique index additionally prohibits two running jobs.
- Queued cancellation is terminal and never claimable. Running cancellation persists intent without releasing the slot. Repeated cancellation and terminal-job cancellation are no-ops. Safe logs contain generated IDs, not keys, raw records or private paths. Audit events are append-only while records are retained; later retention work must explicitly handle their controlled deletion.
- Bounded contention policy: a five-second SQLite busy timeout per BEGIN attempt, at most two retries (three attempts total, roughly fifteen seconds plus scheduler overhead). Tests lower this to 25 ms with one retry without relaxing production limits. Transactions roll back job, lease and audit together on injected failures.
- Temporary-store integration for binary classification and regression: mixed numeric/categorical input with missing features, frozen split JSON round trip, saved configuration/run/profile metadata, training-only views, explicit test-driven Dummy evaluation on five folds and deterministic score summaries. Poison holdout records reject any CV indexing of reserved rows. Duplicate fixtures include full-row repeats and identical features with different labels; acknowledgement is required before any dataset/experiment metadata is persisted.

Verification executed on Windows, Python 3.12 and the existing locked environment:

| Command / check | Observed result |
|---|---|
| Baseline `uv run --locked pytest -q` | 277 passed in 12.89 s, zero skips |
| Initial queue tests before implementation | Collection failed because `apps.api.database` did not exist |
| First implemented queue suite | 14 passed in 4.58 s, zero skips |
| Expanded queue and integration checks | 31 passed, two integration fixture failures; missing numeric values plus repeated categories had unintentionally created duplicate features. Corrected the fixture, not the duplicate policy. |
| `uv run --locked pytest tests/unit tests/integration/test_data_flow.py tests/integration/test_baseline_cv.py tests/integration/test_job_repository.py -q` | 310 passed in 15.14 s, zero skips |
| Full regression `uv run --locked pytest -q` | 310 passed in 13.24 s, zero skips |
| `uv lock --check` / `git diff --check` | Both exited 0 |
| `graphify update .` | AST-only update completed; graph artifacts remain ignored. Community labels may need refreshing; no paid/API extraction was run. |

**G1 data-correctness gate passes locally.** This does not establish a working background process or a release. Day 7's integration explicitly calls CV from test code; preparing or enqueueing an experiment never runs training. Candidate publication, MLflow verification, worker execution/heartbeats, stale-owner liveness checks, cancellation of native fitting and terminal result transitions belong to Days 8–12. The initial lease generation is one; no expiry-based takeover is permitted. Final holdout reservation and one-time evaluation remain Days 14/19. Upload path persistence, HTTP transport and retention are not implemented by these metadata primitives.

Initial resource decision: retain **provisional T=4 and 4 GiB RSS**, with all existing caps unchanged. The actual Day 2 probe recorded 5.29885 s total wall time, 152.36 MiB peak sampled process RSS and 4.8202 GiB minimum sampled available RAM. It was a small battery-powered parser/Dummy probe, not a full classical/MLP or usability benchmark; it cannot justify raising/lowering limits or selecting a final thread count. Day 21's 2/4/6 comparison remains required.

Independent read-only review approved Day 7 with **no Critical/Important findings** and independently reran the 310-test G1 gate and 33-test Day 7 suite, with zero skips. One minor is deferred: SQLite's running-job `CHECK` accepts NULL `owner_generation` when written directly through SQL, because a NULL CHECK expression is not false. The implemented claim path always writes a validated positive generation; no repository-path failure was found. Add an explicit `IS NOT NULL` constraint and regression when extending worker state/schema. This does not authorize treating malformed ownership as safe during recovery.

Review scope decisions: native cancellation/liveness/recovery remain Days 8–9; tracking/publication remains Day 12; finalization crash boundaries remain Days 14/19; complete deletion remains Days 13/25. Existing Day 3/5 duplicate/preprocessing checks were rerun in G1 and their service boundaries reviewed; this is not a new exhaustive audit of those modules. None of these future features is claimed as delivered. If their later integration bypasses the recorded boundaries, G1 alone cannot establish overall safety.

Commit verification: `4b3ca13` — `feat(jobs): persist idempotent single-worker queue and experiment metadata`. All seven new implementation/test files were staged explicitly, and the complete staged diff and whitespace checks were reviewed. The full suite passed again immediately before commit: **310 passed in 14.69 s, zero skips**. Only README and this evidence file remained modified afterward; runtime/user data and local planning/graph files were not staged. A following documentation commit publishes the verified delivery and G1 status. Remote push is verified separately after these commits, not presumed from local commits.

## Day 6 — 2026-10-26

The user authorized Day 6 implementation, English output, commit and push, and chose a new branch in the existing checkout: `feat/day-6-metrics-baseline-cv`. The starting tree was clean and its full suite passed **226 tests**, zero skips. No dependencies changed or datasets were downloaded. Only small offline synthetic fixtures were fitted; no holdout scoring, named demonstration, performance benchmark or release gate was run. Day 7 onward remains unimplemented.

Delivered:

- Frozen metric/CV result contracts and preselected task-specific primary metrics in `RunConfig`.
- Exact binary precision, recall, F1, fixed-order confusion matrix, ROC-AUC and non-interpolated AP; encoded positive label 1, inclusive 0.5 probability threshold, aligned finite bounded scores. Undefined metrics are null with reasons. A no-positive-prediction baseline with actual positives retains F1=0 despite null precision.
- Regression MAE, RMSE and R2; constant-target and single-sample R2 are explicitly null rather than coerced to a finite sklearn default. Unrepresentable arithmetic fails safely.
- A bounded Dummy-only classical factory and serial frozen-fold evaluator using fresh Day 5 guarded preprocessing, training-only global-ID mapping, separate fit/score/total timings, complete-fold means and population standard deviations (`ddof=0`). Failed, missing or duplicate folds and undefined primary scores cannot rank. No partial-fold mean is published; undefined secondary scores do not invalidate a valid primary.
- Protocol/README updates. Generated Graphify files are ignored; `graphify update .` succeeded after implementation. No generated dataset, model or runtime report is committed.

Executed evidence:

- Metrics tests first failed because `ml.evaluation` did not exist, then passed **37 tests**. One test fixture was corrected because its prediction contradicted its supplied 0.6 probability and the fixed threshold.
- CV tests first failed because `TrainingContext` did not exist, then passed **14 integration tests**. Tests guard both held-out feature and target indexing, verify original fold identities and exactly five results, inspect fresh outer-fit-only preprocessing, hand-check the regression mean baseline, distinguish population/sample std, verify exact rerun metrics and reject partial/failed/undefined-primary results. Memory rejection produces failed folds, never scores.
- `uv run pytest -q`: **277 passed in 11.91 s, zero skips**, including all 226 prior tests.
- Independent read-only review reran all **277 tests** and separately exercised validation-only categories with entirely missing numeric/categorical columns. It found no Critical or Important issues.
- `uv lock --check` and `git diff --check` exited 0; Git's CRLF normalization notices are not whitespace errors.

Final pre-commit verification on the completed source: `uv run --locked pytest tests/unit/test_metrics.py -q` — **37 passed in 3.37 s**; `uv run --locked pytest tests/integration/test_baseline_cv.py -q` — **14 passed in 5.92 s**; `uv run --locked pytest -q` — **277 passed in 10.29 s**, all with zero skips. Lock and whitespace checks also passed. The requested delivery is a new remote branch, not a merge to `main` or a release; its commit identifier is available in Git history.

Nonblocking review Minor, deferred: fold failures currently share the safe `fold_execution_failed` code. Ranking is safe and raw exception text is suppressed, but categorized resource/data/execution diagnostics would make remediation easier. Detailed safe diagnostics belong with later worker/tracking integration; this limitation is not a claim of production-ready error reporting.

Execution decisions: use direct D6a/D6b tasks and a local ignored ledger rather than task-extraction scripts for standalone numbered plans (bookkeeping only). Continue to defer worker cancellation, hard native-fit deadlines, durable publication and additional classical families to their assigned days; this evaluator is not exposed on an API request thread. The source-content identity boundary remains `training_from_manifest`; the evaluator rechecks configuration/manifest/row identity, not raw dataset fingerprints from manually constructed views. In-progress documentation was checked separately by the implementer. G1 remains blocked on Day 7; G2–G4 remain unpassed.

## Day 1 — 2026-10-22

Scope authorized by the user's request to implement Day 1 only. Day 2 onward, frontend scaffolding, training, benchmarks and dataset downloads were not executed. The user confirmed 2–3 focused hours per day.

Environment observed: Windows 11 Home Single Language build 22621, AMD Ryzen 5 5500U, 6 physical / 12 logical processors, 16,455,942,144 bytes (15.326 GiB) usable physical memory. The inventory sample at 12:16 UTC recorded 4,186,640,384 bytes (3.899 GiB) **available** memory. This is a changing snapshot, below the future 4 GiB benchmark reserve, not evidence that a 4 GiB child is safe to start. Power scheme was Balanced, AC connected; Windows Settings power mode was not observed separately.

Toolchain: Python 3.12.15 (uv managed), uv 0.13.5, existing Node 24.19.0 and npm 12.2.0. The architecture's proposed Node 22 was provisional; Day 1 pins the installed Node 24 toolchain. React/Vite compatibility remains a Day 22 check. No existing Git repository was reinitialized. Work was performed in the requested checkout.

Locked direct versions: FastAPI 0.159.1, Pydantic 2.14.1, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, torch 2.14.1+cpu, MLflow 3.17.2, psutil 7.2.2, uvicorn 0.54.0, python-multipart 0.0.32, threadpoolctl 3.7.0, pytest 9.1.1 and httpx 0.28.1. All installed from wheels; torch uses the explicitly scoped CPU index. The full version/hash graph is in `uv.lock`; frontend packages will be locked on Day 22. Windows x86-64 is the verified target; other platforms are unverified.

Executed evidence:

- Initial `uv lock`, dependency installation into a new `.venv`, and final `uv sync --locked` succeeded. Direct pins were frozen from that resolution, then the lock was regenerated and locked sync repeated successfully.
- `uv run --locked python -c "import fastapi, sklearn, torch, mlflow; print(torch.__version__)"` and an expanded smoke check succeeded. torch reported `2.14.1+cpu`, `torch.version.cuda is None`, `torch.cuda.is_available() == False`; a CPU tensor calculation returned `[2.0, 2.0]`. No training ran.
- `npm.cmd ci --ignore-scripts` succeeded at repository root: zero package dependencies and zero audit findings. This verifies the toolchain manifest only. The planned `apps/web` install is deferred to Day 22.
- Settings tests were written before implementation; their first run failed because `apps.api.settings` did not exist. Hardware-inventory tests first failed because `scripts.profile_hardware` did not exist. After implementation, `uv run --locked pytest -q` passed **53 tests**, with no skips. A fresh reviewer then found an iCloud Drive denylist typo. The new sync-folder regression reproduced it (1 failed, 57 passed); the spelling was corrected and the expanded suite rerun. Tests exercise strict policy validation, immutable settings, local home resolution, environment inputs, available-memory reporting and lock identity.
- `uv run --locked python -m scripts.profile_hardware` succeeded, matched the confirmed hardware, and wrote the private `inventory/hardware.json`. Settings loads do not create directories. Inventory files are observations, not benchmark evidence.
- `uv run --locked mlflow gc --help` confirmed the pinned release exposes permanent deleted-run metadata/artifact cleanup with explicit `--backend-store-uri` and `--run-ids`. The CLI warns invalid artifact URLs can bypass artifact cleanup. Actual SQLite/artifact purge correctness remains a Day 12/25 acceptance test; help output alone does not prove deletion.
- `git check-ignore` confirmed representative secrets, uploads/data, models/artifacts, databases, runtime inventory, virtualenv and frontend dependencies are ignored. `git diff --check` passed.

Policy version 1 defaults remain provisional: T=4, RSS=4 GiB, one active job, serial CV, estimator n_jobs=1, torch inter-op=1 and DataLoader workers=0. Configuration rejects unreviewed cap increases; it does not yet enforce runtime resource limits. Thread policy, cancellation, spawn behavior and full 2/4/6 performance comparison remain future work. No experiment-success claim is made.

Execution decisions: Day 1 uses the installed Node 24 rather than the provisional Node 22 (frontend compatibility must be verified on Day 22). The root npm lock pins only the toolchain, preserving the plan's frontend deferral. The existing checkout was used for this setup work. The implementation and evidence were initially left uncommitted for user review; the dated entry below records subsequent commit verification and delivery. No P0 requirement was removed.

Final review fix: `test_known_sync_folders_cannot_host_active_storage[iCloudDrive]` failed before the denylist correction and passed afterward. Final `uv run --locked pytest -q`: **59 passed in 2.50 s, zero skips**. `uv lock --check` and `git diff --check` also passed. No review findings remain deferred.

## Day 1 commit verification — 2026-10-23

The user explicitly requested the four planned Day 1 commits. Existing implementation was packaged in dependency order, with no training, benchmark, frontend scaffold or Day 2 implementation. The prior Day 1 results above remain historical evidence.

| Commit | Delivered scope | Fresh verification |
|---|---|---|
| `2dc1dc7` | Locked Python/Node toolchain, CPU torch source, root npm manifest and local-workspace ignore rule | Exported staged tree: `uv lock --check`, `uv sync --locked`, CPU import smoke check and `npm.cmd ci --ignore-scripts` all exited 0. Fresh environment installed 105 packages; npm reported zero vulnerabilities. |
| `f989c16` | Immutable resource/run contracts, local storage settings and owning tests | Exported cumulative staged tree: `uv run --locked pytest tests/unit/test_settings.py -q` — 58 passed in 0.53 s, zero skips. |
| `de54447` | Hardware inventory command and owning test | Exported cumulative staged tree: inventory test — 1 passed in 3.24 s; full suite — 59 passed in 3.41 s, zero skips; inventory CLI exited 0. |

The staged trees were exported with `git checkout-index` to a new temporary directory outside the repository, using a separate freshly synchronized virtual environment. Only staged files and prior commits were exported; later unstaged source could not satisfy earlier checks. The export has no `.git`, so its inventory correctly reported code revision unavailable. No source or dependency versions were changed during packaging.

CPU smoke command executed on the first staged tree:

```powershell
uv run --locked python -c "import fastapi, sklearn, torch, mlflow; print(torch.__version__); assert torch.version.cuda is None; assert not torch.cuda.is_available()"
```

It reported `2.14.1+cpu` and exited 0. Root npm verification covers the toolchain manifest only, with no frontend dependencies. Representative `.env`, data, uploads, artifacts, runtime files, virtualenv, node_modules, model weights and SQLite paths were confirmed ignored with `git check-ignore`.

After `de54447`, the actual checkout also passed `uv run --locked pytest -q`: **59 passed in 3.29 s, zero skips**. `uv lock --check` and `git diff --check` exited 0. The inventory CLI was rerun successfully in this checkout and recorded revision `de54447533c87bc7bfefa9e436228a913655ba29` at `2026-10-22T22:43:11.860430+00:00` (2026-10-23 in Asia/Jakarta). It observed the reference hardware, 6.037 GiB available RAM, Balanced power scheme and battery power. This snapshot is inventory, not benchmark or resource-safety evidence. The generated JSON remains in private application storage outside Git.

Inventory lock checksums describe actual file bytes; Windows checkout line-ending conversion can produce different hashes between an exported tree and the original working copy. No deterministic experiment rerun is claimed by this setup check.

The fourth commit publishes this evidence and the README delivery table. Its identifier is available from Git history after creation; the document does not attempt to embed its own commit hash. Day 1 is committed and verified; all four milestone gates remain unpassed. No remote push or release is included in this delivery.

## Day 2 — 2026-10-23

The user authorized Day 2 implementation and requested English output. Work stayed in the requested checkout and added no dependencies. Day 3 onward remains unexecuted. The initial delivery was verified locally and left uncommitted for review; the subsequent authorized commits are recorded below. No push or release occurred.

Delivered structural `validate_csv` with UTF-8/BOM decoding, strict comma/quote syntax, preserved duplicate-header checks, missing-cell preservation, and byte/row/column/UTF-8-cell bounds. It hashes the same stream it parses and rejects ordinary file substitution using pre-open/post-read file identity, size, and modification checks. Trusted service callers can supply a private root; HTTP path handling is not implemented yet. Parsing scans 4 KiB decoded chunks and checks column/cell limits before CSV record allocation. A conservative three-times Python-object estimate additionally bounds retained parsed data against the existing preprocessing memory budget; this is an allocation guard, not an OS memory quota.

Canonical policy v1 hashes version-prefixed compact JSON records with explicit record separators. It includes headers, values, row order, and column order. BOM, equivalent quoting, and CR/CRLF/LF formatting do not change canonical identity; embedded newlines normalize to LF. Empty strings represent missing input structurally. Whitespace, `NA`, and numeric spellings remain literal strings; task/schema inference is deferred to training-only configuration. The result contains no analytical statistics, and its representation omits rows.

Delivered the initial resource recorder and CLI. It samples current-process RSS, process CPU-seconds, and available system memory at start, finish, and approximately every 0.5 seconds. Wall time uses the high-resolution monotonic `perf_counter`; normalized process CPU is `100 * CPU-seconds / wall-seconds / logical CPUs`. Samples do not include descendants or guarantee detection of transient RSS peaks. The command configures BLAS up to T and subordinate OpenMP to one thread, runs serial folds, records actual pools, and prevents overlapping initial probes with an exclusive lock. The future worker owns cross-process admission, cancellation, timeout, and RSS termination.

Executed checks:

- The CSV and benchmark tests initially failed because their production modules did not exist. After implementation, the named suites passed. A fresh reviewer identified excessive allocation before column rejection. `test_delimiter_flood_is_rejected_before_large_cell_list_allocation` reproduced it with an approximately 18.6 MB Python allocation for a 1 MB delimiter input; the streaming guard reduced it below the test's conservative 6 MB ceiling. This is parser allocation evidence, not whole-process RSS evidence.
- `test_recorder_handles_work_shorter_than_a_clock_tick` reproduced a zero-duration measurement on Windows. Switching from Python 3.12's coarse `monotonic` clock to high-resolution monotonic `perf_counter` fixed it.
- The existing equivalent-format and exact-boundary CSV cases exposed inconsistent Windows `stat`/`fstat` ctime semantics after rewrites. File identity checks now use device, inode, size, and mtime consistently; those tests and the replacement-before-open test pass.
- Observed pools exposed both BLAS and OpenMP configured T-wide. An added assertion failed, then passed after OpenMP was restricted to one. Earlier local probe reports remain retained as superseded observations; they are not comparative performance trials.
- Final source verification: `uv run --locked pytest -q` — **103 passed in 11.18 seconds, zero skips**. The two Day 2 suites contain 44 tests; the 59 existing tests also pass. `uv lock --check` and `git diff --check` exited 0.

The final `uv run --locked python -m scripts.benchmark --stage initial --threads 4` exited 0 and wrote private `benchmarks/initial-204df23feb7442d3a9df824c8edcb180.json` under application home at `2026-10-22T23:02:06.813667+00:00` (2026-10-23 Asia/Jakarta). It records the current revision plus a dirty-worktree flag, lock/package versions, seeds, partition/fold hashes, parser hashes, exact configuration, and raw samples. No user dataset, model, or runtime report was added to Git.

| Observation | Actual result |
|---|---|
| Dataset / partition | Bundled Breast Cancer; 455 scratch training rows, 114 reserved rows; reserved feature rows never indexed for parsing/fitting/scoring |
| Parsed scratch CSV | 96,791 bytes, 455 rows, parser wall time 0.1515 s |
| Raw SHA-256 | `26390b2747307a2d65951958373bd9cc82dc5b937c20dcf6bd195a12e91ad82f` |
| Canonical SHA-256 | `1fcceac5fd7616cb5e117e69f3147f983278d4ac36e920ae14342bddbd329c68` |
| Dummy workload | Five serial folds, 364 fitting / 91 validation rows each; most-frequent strategy; scratch CV mean F1 0.77027027 |
| Measured wall time | 5.29885 s, including numerical imports, CSV generation, parsing, and CV; inventory excluded |
| CPU | 0.875 process CPU-seconds; 1.3761% normalized over 12 logical CPUs |
| Peak sampled current-process RSS | 159,764,480 bytes / 152.36 MiB |
| Minimum sampled available system RAM | 5,175,668,736 bytes / 4.8202 GiB |
| Observed pools | Two BLAS pools at four threads; OpenMP at one; serial Dummy fits |
| Power conditions | Balanced scheme, battery power, 80%; Windows Settings power mode not separately observed |

This is an initial instrumentation probe on battery power, without excluded warm-up, repeated trials, full classical/MLP workloads, API latency, or desktop usability evidence. Its small workload and low utilization do not justify a thread or memory decision. T=4 and 4 GiB remain provisional; production immutable splits arrive on Day 4, worker integration on Day 8, and the full comparison on Day 21. No holdout evaluation or release gate passed.

Review decision: retained the existing preprocessing budget as an additional conservative parsed-input guard instead of adding a new configurable cap. If this proves too restrictive for otherwise admissible inputs, measure allocation and review the policy before changing it. The staged API will pass server-owned paths; this parser does not make arbitrary client paths safe.

Deferred minor from the independent review: strengthen the probe's isolation regression with identifiable sentinel rows and spies on parser/model inputs. Current tests check counts and report fields; source review verified the training-only indexing, but the test alone cannot detect every future row-substitution regression. The production partition/poison-holdout checks remain mandatory in later tasks.

## Day 2 commit verification — 2026-10-23

The user requested commits with appropriate messages. The existing implementation and its tests were committed in dependency order, followed by the documentation. No implementation or dependency versions changed during packaging.

| Commit | Delivered scope | Fresh staged verification |
|---|---|---|
| `6270779` — `feat(data): validate bounded CSV input and stable fingerprints` | Structural CSV parsing, safe validation errors, deterministic raw/canonical identity, and owning tests | Full exported staged tree: **90 passed in 4.01 s, zero skips**; benchmark files were absent from this snapshot. |
| `44be1cb` — `feat(benchmark): record initial CPU resource observations` | Monotonic resource sampling, training-only parser/Dummy probe, CLI, and owning tests | Full exported staged tree: **103 passed in 11.34 s, zero skips**. |

Each staged snapshot was exported with `git checkout-index --all` into a newly created temporary directory outside the repository. Tests ran from that directory using `uv run --project <export> --directory <export> --locked pytest -q` with `UV_PROJECT_ENVIRONMENT` pointing to the existing locked checkout environment. This isolates source files from later unstaged changes; it is not a fresh dependency-installation check. The exported snapshots have no Git metadata, so their hardware-inventory test accepts unavailable revision information.

Before each code commit, the staged names, complete diff, and `git diff --cached --check` were reviewed. `git status --short` was checked after each commit. `uv lock --check` also exited 0. Generated benchmark JSON, temporary verification exports, and ignored local planning documents were excluded from the commits. The documentation commit publishes the actual code hashes and evidence; its own identifier is available from Git history. Resource settings remain provisional, and the deferred minor above remains recorded.

## Day 3 — 2026-10-26

The user explicitly requested execution of Day 3 with English output. D3a and D3b are verified locally against the existing locked Windows/Python 3.12 environment. No dependencies changed, training ran, dataset downloads occurred, or commits were created. Days 4–28 remain unexecuted and G1–G4 remain unpassed.

D3a adds frozen `DatasetConfig`, ordered `FeatureSchema`/`FeatureColumn` declarations, explicit binary positive-label mapping, finite numeric/regression checks, feature exclusion, boolean parsing and duplicate acknowledgement. Full-data validation checks integrity and duplicate existence only; it returns private configured records without analytical summaries or detailed duplicate counts. Repeats use selected typed features, including numeric normalization, and full-row counts mean selected features plus target. All rows, including differing-label repeats, remain intact. Acknowledgement never overrides integrity errors.

D3b adds `profile_training(TrainingView) -> DatasetProfile`. The view materializes only selected training rows and keeps no parent dataset reference. Profiles contain dimensions, declared types, missingness, distinct counts, numeric count/mean/population std/min/max/linear quartiles, up to five categorical top counts, class counts/minority ratio or regression target summaries, and separate training-only duplicate counts. Warnings cover imbalance below 10%, ID-like features, possible target/post-outcome names, missing/constant features, repeated observations and independence assumptions. They are explicitly heuristic, bounded to twenty, and report truncation. No row examples or full-data analytical profile is returned. A view with no observed feature values fails safely.

Executed verification:

- Both new suites initially failed collection for the missing contracts/profile module before implementation.
- D3a first passed 31 tests. Two initial test defects were corrected: a purported multiclass fixture contained only two labels, and target-as-feature rejection correctly occurred during configuration construction.
- Added direct tests for normalized numeric duplicates, duplicate-plus-invalid-target interaction, reordered features, wide-profile warning truncation and training-only class balance.
- Independent read-only review found quartile underflow when a large outlier was mixed with small representable values. A failing regression reproduced zero quartiles instead of `1.75e-100` and `2.5e-100`. Another regression reproduced loss of a small mean residual after large-value cancellation. Original-value overflow-safe interpolation and compensated summation of divided observations fixed these cases; variance still uses scaling to avoid overflow.
- `uv run --locked pytest tests/unit/test_profile.py tests/unit/test_validation.py -q`: **47 passed in 0.74 s, zero skips** (34 validation, 13 profile).
- `uv run --locked pytest -q`: **150 passed in 9.68 s, zero skips**, including all 103 existing tests.
- `uv lock --check` and `git diff --check` exited 0. Git emitted line-ending normalization notices for files configured to use CRLF; no whitespace errors were reported.
- The reviewer rechecked the numerical fixes and duplicate/invalid-label interaction and reported no remaining Critical or Important findings.

Interface decisions: schema is supplied explicitly, never inferred from holdout values. Missing feature cells are empty/whitespace-only; categorical `NaN` remains a literal category, whereas numeric NaN/inf and nonfinite target spellings are invalid. Binary labels preserve their original strings and map the explicit positive label to 1. Boolean spellings are true/false or 1/0, case insensitive. Detailed duplicate counts and warnings are training-only. Day 4 will provide persisted training indices; `training_view` is a trusted internal helper, not a partition proof or client-facing selector. Day 5 owns fold-local imputation, empty-column policy and allocation guards. Profiles must never fit preprocessing. Bank pre-call configuration must exclude `duration` before training integration.

Deferred review Minor: averaging the smallest subnormal floats (for example three `5e-324` values) can round to zero because division precedes summation. No Critical/Important issue remains; this numerical precision limitation is recorded rather than claiming exact real-number summaries. Persistent acknowledgement, split provenance and partition-feasibility checks remain assigned to later integration tasks. Sentinel tests prove isolation of a supplied view, not correctness of an unimplemented split creator.

### Day 3 commit verification

The user subsequently requested pushing current progress. Commit `d58159f` (`feat(data): validate task configuration and profile training rows`) contains the Day 3 contracts, validation, profiles and both owning test suites. These were combined because configuration and profiling share the new contracts; the commit is independently usable with its predecessors. Earlier notes describe the initial uncommitted handover.

Before committing, the exact five code/test paths were staged and the staged names, complete diff and whitespace check were reviewed. A fresh `git checkout-index --all` export into a new temporary directory was tested using the existing locked environment via `UV_PROJECT_ENVIRONMENT`; `uv run --project <export> --directory <export> --locked pytest -q` passed **150 tests in 12.85 s, zero skips**. This checks the staged source tree independently of unstaged documentation; it is not a fresh dependency installation. `uv lock --check` and `git diff --cached --check` exited 0. Post-commit status showed only README and this evidence document pending.

The following documentation commit publishes the actual code hash and verification evidence. Ignored architecture/implementation planning documents and runtime artifacts remain local. G1–G4 remain unpassed, and Day 4 onward remains unexecuted. The requested push includes existing local history that was absent from `origin/main`; this is progress publication, not a release.

## Day 4 — 2026-10-26

The user requested continuation of an in-progress Day 4 implementation. The starting checkout contained modified `ml/contracts.py` and untracked `ml/data/splits.py`/`tests/unit/test_splits.py`; those changes were continued in place. D4a/D4b are now verified locally. No dependencies changed, training or downloads ran, or commits/pushes occurred in this session. Days 5–28 remain unexecuted and all four release gates remain unpassed.

Delivered frozen holdout, outer-fold and inner-MLP identities; deterministic canonical JSON/checksums; partition content hashes; splitter algorithm/version and seeds; descending 5/4/3/2 feasibility checks; atomic, idempotent publication that cannot replace a conflicting record; bounded JSON reload with structural/checksum validation; and `training_from_manifest` to bind persisted training indices to the exact configured source. Added [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) describing row independence, leakage boundaries, rounding, adaptation, persistence, failure behavior and remaining integration work.

Executed verification:

- Initial inherited suite: **19 failed, 15 passed**. The main failure was a mismatch between global dataset labels and training-only row positions in outer CV. Corrected label indexing and retained original global row identities for inner partitions. Other fixes addressed content-sensitive partition hashes and the inner expansion flag.
- Corrected inherited test defects: a manifest-construction exception occurred before its `raises` block; an 80-row dataset has 64 training rows, so five validation folds contain 12/13 rows rather than 16; nonempty warnings do not need the literal word `warning`; a supported `kfold` string alone is not invalid without task information. Replaced a duplicated feasibility algorithm in the tests with literal expected fold counts and partition assertions, and repaired the synthetic contract fixture to validate every training row exactly once.
- Added failing regressions for positive-label polarity changing splits, repeated/missing outer validation coverage, unsupported fold counts, conflicting writes replacing a frozen record, tampered persistence, unbounded inner-size search, missing manifest reload/source-binding behavior and corrupted persisted row counts. Each relevant suite was rerun after implementation.
- Independent review reproduced the corrupted-row-count resource issue: a tiny JSON record could previously request `set(range(10**12))` before checksum verification. A regression safely intercepted that allocation; validation now checks supplied identity cardinality and bounds without allocating from the declared count.
- `uv run --locked pytest tests/unit/test_splits.py -q --tb=short`: **55 passed in 4.53 s, zero skips**. Includes classification/regression, holdout/outer/inner disjointness, actual five/four/three-fold selections, two-fold feasibility, infeasible inner counts, fractional rounding, content identity, deterministic byte output, JSON reload, corruption, immutable publication and failure cleanup.
- A poisoned tuple accessor raises on held-out feature indexing. The persisted-manifest-to-profile test passes without triggering it and checks the training mean against the persisted row IDs. Mismatched source/configuration identities are rejected before applying indices.
- `uv run --locked pytest -q`: **205 passed in 10.29 s, zero skips**, including all 150 pre-Day-4 tests.
- `uv lock --check` and `git diff --check` exited 0. Git emitted CRLF normalization notices, not whitespace errors.
- Independent read-only review reported **no remaining Critical or Important findings**, reran all 55 split tests, and additionally confirmed binary-search inner sizing agrees with linear search for counts 4–34, every valid minority count and four seeds.

Interface decisions and rulings:

- All data remain under parser policy v1. Row IDs are positional and sorted on persistence. Partition hashes cover only their corresponding original records/columns/row IDs and partition name; the separate complete-dataset fingerprint binds the overall source. No raw rows are written to the manifest.
- Use ceiling for the default 20% sizes, matching a fractional split policy, and increase to the documented two-row minimum. Binary inner validation may grow further to the smallest feasible size; persist this expansion. Stable ordering of original label strings prevents positive-label polarity from changing assignments. Changing model seed has no effect on partitions.
- Keep the complete inner-MLP requirements during fold adaptation. Although the plan names 4/3/2 adaptation, two folds may never be the largest feasible choice under the present minimums. Two-fold candidates are supported and directly tested; the selected-five/four/three fixtures and rejection cases preserve all safety requirements. The reviewer agreed that no P0 minimum should be relaxed to manufacture a two-fold selection.
- JSON reload is bounded to 16 MB and rejects duplicate fields. Same-directory publication uses a closed/flushed/fsynced temporary file and a no-clobber hard link; unsupported filesystems fail rather than overwrite. Identical retries succeed, different records conflict, and simulated publication failure leaves no final or temporary file. Private service-owned paths remain required.
- The split module does not replace future SQLite evaluation-scope enforcement. Scope lookup must reuse the original split and prohibit fresh holdout opportunities after feature/metric/seed changes. Checksums are integrity checks, not authentication. Fold-local preprocessing, resource guards, worker lifecycle and one-time finalization remain assigned to later tasks.

This completes the Day 4 profile-partition verification deferred from Day 3, but does not claim G1 completion: Days 5–7 remain pending. Existing documented Day 3 subnormal-mean precision and Day 2 initial-probe test limitations remain unchanged.

### Day 4 commit verification

The user subsequently requested committing and pushing the changes. Commit `64a8804` (`feat(data): persist leakage-safe holdout and fold identities`) contains the split contracts, implementation, owning test suite and experiment protocol. These form one reviewable partitioning change; earlier notes describe the initial uncommitted handover.

Before committing, the four exact paths were staged, the staged names and diff were reviewed, and `git diff --cached --check` exited 0. A fresh `git checkout-index --all` export into a new temporary directory was tested with `uv run --project <export> --directory <export> --locked pytest -q`, using the existing locked environment via `UV_PROJECT_ENVIRONMENT`: **205 passed in 31.01 s, zero skips**. This verifies the staged source independently of the unstaged README/evidence changes; it is not a fresh dependency-installation check. `uv lock --check` also exited 0. Post-commit status showed only README and this validation document pending.

The following documentation commit publishes the actual code hash and verification result. Local planning documents remain ignored; runtime files, datasets and generated artifacts are excluded. Day 5 onward remains unexecuted and G1–G4 remain unpassed. Pushing this progress does not constitute a release.

## Day 5 — 2026-10-26

The user authorized Day 5 implementation, commit and push. Delivered a fresh preprocessing factory and guarded `FoldPreprocessor` in `ml/preprocessing/factory.py`, with owning tests in `tests/unit/test_preprocessing.py` and `tests/unit/test_resource_estimates.py`. No dependencies changed, candidate training or downloads ran. Days 6–28 remain unexecuted and G1–G4 remain unpassed.

Numeric median imputation/scaling and categorical/boolean missing imputation/one-hot encoding learn only from fitting rows. Empty numeric columns remain zero; constant features and empty categorical columns keep stable finite output. Namespaced categories cannot collide with the missing sentinel. Unknown categories transform without changing fitted width. The feature matrix excludes targets. The wrapper accepts isolated views with matching configuration, records fitting row IDs and creates a fresh transformer on refit; callers remain responsible for selecting persisted fitting identities.

Training-only estimates enforce policy-v1 category/width limits and the 512 MiB preprocessing budget before fit. The live-buffer estimate counts fitting and transform rows, float64 intermediates/copies, float32 cast space and a three-times safety factor. Transform allocation is checked before encoding; realized width/shape and budget are rechecked before explicit dense conversion. Output finiteness and float32 range are checked. See [experiment protocol](EXPERIMENT_PROTOCOL.md#day-5-preprocessing-boundary) for the exact formula, policies and caller obligations. This estimate does not enforce process RSS or cover model allocations; worker enforcement remains later work.

Executed checks in the locked environment:

- Initial test-first run against unimplemented interfaces: **21 failed**, demonstrating absent Day 5 behavior.
- `uv run --locked pytest tests/unit/test_preprocessing.py tests/unit/test_resource_estimates.py -q`: **21 passed in 6.37 s, zero skips**.
- `uv run --locked pytest -q`: **226 passed in 16.74 s, zero skips**, including all 205 previous tests.
- `uv lock --check`: exited 0, resolved 107 packages without lock changes.
- `git diff --check` and final `git diff --cached --check`: exited 0. The staged file list contains only the seven Day 5 source/test/documentation paths.

A separate read-only review found no actionable correctness or leakage issues and independently ran the two new test files: **21 passed, zero skips**. No candidate-score reproducibility, hardware performance, worker resource safety or milestone completion is implied. The local ignored implementation plan marks D5a/D5b complete; it remains outside Git under the existing repository policy. Source, tests, protocol and public validation evidence are included in the Day 5 commit.
