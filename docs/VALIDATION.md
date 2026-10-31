# Validation evidence

## Day 19a acceptance — Final evaluation, holdout gate, and crash recovery — 2026-11-01

**Status: ACCEPTED.** D19a is verified and checked in `docs/IMPLEMENTATION_PLAN.md`. All required safety criteria, independent review fixes, and full serial regressions passed with zero failures and zero skips under clean host conditions:

1. **Clean Serial Verification Pass (2026-11-01):**
   - Available host physical RAM before run: 7.27 GB (3.27 GB headroom above the 4.00 GiB reserve). No limits were lowered, no guards were disabled, and no monitoring was mocked.
   - Owning finalization suite: **30 passed**, 2 warnings in 143.66s (`tests/integration/test_finalization.py`).
   - Combined regression batch: **113 passed**, 2 warnings in 367.37s (`tests/integration/test_review_finalization_regressions.py`, `tests/integration/test_holdout_gate.py`, `tests/integration/test_worker_lifecycle.py`, `tests/integration/test_worker_spawn.py`, `tests/integration/test_resource_monitoring.py`, `tests/integration/test_refit.py`, `tests/integration/test_retention.py`, `tests/integration/test_api.py`). Zero `resource_pressure` errors occurred.
   - Independent safety review suite: **9 passed**, 1 warning in 42.75s (`tests/integration/test_d19a_independent_review.py`), including payload integrity regressions.
   - Total clean passes: **152 tests passed**, 0 failed, 0 skipped across all suites run strictly serially.
   - Lock, diff, and graph integrity: `uv lock --check` passed in 1ms; `git diff --check` passed cleanly; `graphify update .` rebuilt 1,507 nodes, 5,192 edges, 81 communities with no overlap during testing.

2. **Core In-Scope Safety & Integrity Guarantees Verified:**
   - **One-time Holdout Access:** `read_holdout` requires a valid `HoldoutCapability` issued only after durable evaluation-start in SQLite. Holdout is scored once (`score_holdout`). Repeated GET/POST requests retrieve the exact published result without refit or re-reading holdout.
   - **Payload Schema & Identity Validation:** Bundle verification now inspects both checksums/manifest identities and the `results.json` payload schema via `validate_final_result_payload`, matching `finalization_id`, `scope_id`, `candidate_id`, `candidate_digest`, and `evidence_sha256`.
   - **Windows Spawn Execution & Mixed Dispatch:** Normal worker runner (`apps/api/workers/runner.py`) sets `claim_all=True` to claim queued CV and finalization jobs in FIFO `sequence` order, spawning execution children via `multiprocessing.spawn`, while preserving CV-only supervisors.
   - **Durable Plan & Evidence Binding:** `finalizations` durably binds `plan_json`, `epoch_budget`, and `evidence_sha256`. Pre-start retries verify identity against database records.
   - **Crash Recovery & Fencing:** Supervisors enforce active generation/lease ownership. Recovery of valid staged bundles completes without re-evaluation; post-start crashes before staging lock the reservation (`locked_failed`).

3. **Captured Outputs from Clean Acceptance Run:**

**Owning Suite: `uv run --locked pytest tests/integration/test_finalization.py -q`**
```text
..............................                                           [100%]
============================== warnings summary ===============================
tests/integration/test_finalization.py::test_equivalent_upload_deletion_removes_shared_partitions
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/integration/test_finalization.py::test_production_finalization_worker_evaluates_and_publishes_once_with_repeated_retrieval
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
30 passed, 2 warnings in 143.66s (0:02:23)
```

**Combined Regressions: `uv run --locked pytest tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py tests/integration/test_resource_monitoring.py tests/integration/test_refit.py tests/integration/test_retention.py tests/integration/test_api.py -q`**
```text
........................................................................ [ 63%]
.........................................                                [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
tests/integration/test_holdout_gate.py::test_concurrent_contenders_yields_one_selection
  SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.
=========================== short test summary info ===========================
113 passed, 2 warnings in 367.37s (0:06:07)
```

**Independent Safety Review: `uv run --locked pytest tests/integration/test_d19a_independent_review.py -q`**
```text
.........                                                                [100%]
============================== warnings summary ===============================
tests/integration/test_d19a_independent_review.py::test_retry_rejects_altered_stored_plan
  SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.
=========================== short test summary info ===========================
9 passed, 1 warning in 42.75s
```

**`uv lock --check`**
```text
Resolved 107 packages in 1ms
```

**`git diff --check`**
```text
(exit 0)
```

**`graphify update .`**
```text
[graphify watch] Rebuilt: 1507 nodes, 5192 edges, 81 communities
```



## Independent D19a safety review continuation — 2026-10-31

**Status: BLOCKED; D19a is NOT accepted.** The D19a checkbox in `docs/IMPLEMENTATION_PLAN.md` remains unchecked. While the owning finalization test suite passed completely (30 passed) and the independent safety review suite passed with new payload validation regressions (9 passed), the combined regression suite failed 4 tests out of 113 because CV worker jobs tripped the `resource_pressure` guard on this host.

### 1. Available Memory & Resource-Pressure Policy Verification
- **Host environment:** Windows 11, PowerShell, locked local uv environment; fixture data only.
- **Physical memory observations:** Total visible RAM is 15.33 GB (16,455,942,144 bytes). Available memory at rest was measured between 4.65 GB and 5.08 GB.
- **Resource policy:** `system_reserve_bytes = 4 * 1024**3` (4,294,967,296 bytes / 4.00 GiB), `memory_pressure_seconds = 10` seconds in `ResourceLimits`.
- **Compliance:** Limits were not lowered, resource monitoring was not disabled, the production guard was not mocked away, and unrelated host processes were not terminated.
- **Failure telemetry:** Inspection of the SQLite database and audit events for the failed jobs in `test_holdout_gate.py` (`test_pre_start_failure_retry_requires_dead_child`, `test_durable_evaluation_start_issues_capability_and_makes_retry_impossible`, `test_fixture_evaluator_and_poison_holdout_isolation`, `test_consumed_tombstone_prevents_new_reservation`) revealed:
  - Recorded job state: `state='failed'`, `safe_error='resource_pressure'`.
  - Audit event telemetry: `job_telemetry:{"child_exit_code":15,"child_pid":4356,"minimum_available_bytes":3926999040,"sample_count":147,"wall_seconds":17.84}`.
  - Minimum available memory fell to 3,926,999,040 bytes (< 4,294,967,296 bytes) and remained below reserve for the full 10-second monitoring window, legitimately triggering `resource_pressure` during multi-fold CV training under cumulative system load.
- **Blocker conclusion:** The host physical memory cannot reliably sustain the full 113-test combined serial regression suite without breaching the 4 GiB reserve. Under AGENTS.md and the execution contract, D19a cannot be claimed as accepted.

### 2. Dispatch Revision Verification
- **Runner dispatch:** The normal runner (`apps/api/workers/runner.py`) configures `WorkerSupervisor(..., claim_all=True)`, which passes `kind=None` to `repository.claim_next()`. The repository selects the oldest queued job strictly by `sequence` across both CV and finalization jobs.
- **CV-only supervisors:** `WorkerSupervisor` defaults to `claim_all=False`, restricting dispatch to `kind="cv"` for `workload="classical"` and `kind="finalization"` for `workload="finalization"`. Explicitly configured CV-only supervisors remain strictly CV-only.
- **Spawned child execution:** Verified that the normal runner with `claim_all=True` claims queued finalization jobs, initializes `finalization_input`, and executes finalization in a dedicated spawned child process via multiprocessing `spawn` (`test_production_worker_claims_finalization`, `test_normal_stored_worker_executes_finalization_in_spawned_child`).

### 3. Review Fixes & Integrity Verification
- **Immutable plan & evidence binding:** `plan_json`, `epoch_budget`, and `evidence_sha256` are durably bound in `finalizations`. Pre-start retries recompute and verify that stored plan and evidence match (`test_retry_rejects_altered_stored_plan`).
- **Stale-owner fencing:** Supervisors verify active lease ownership and generation before publishing or releasing leases (`test_stale_finalization_supervisor_cannot_release_new_lease`).
- **Cancellation at publication:** Requests to cancel an active job during finalization mark the finalization `locked_failed` and release the lease (`test_cancel_before_publication_durably_locks`).
- **Recovery lease release:** Live-supervisor staged recovery releases the worker lease upon successful recovery (`test_recovery_in_live_supervisor_releases_lease`).
- **Bundle identity validation:** `manifest.json` identities (`finalization_id`, `scope_id`, `candidate_id`, `attempt_id`) must match the database attempt (`test_recovery_rejects_bundle_with_wrong_candidate`).
- **Finalization timeout & frozen limits:** `finalization_seconds` timeout enforced for finalization jobs; worker limits must match frozen run config.

### 4. Result Payload Schema & Integrity Fix
- **Defect identified:** Previously, bundle verification validated checksums and manifest identities, but never validated the structure, schema, or identity fields of the `results.json` payload itself.
- **Fix implemented:** Added `validate_final_result_payload(payload, expected)` in `ml/evaluation/finalize.py`. It enforces:
  - Required fields: `finalization_id`, `scope_id`, `candidate_id`, `task`, `primary_metric`, `metrics`, `evaluation_seconds`, `evaluated_at`.
  - Type and boundary checks: valid task enum value, non-empty metric strings, non-negative `evaluation_seconds`, metric bundle structure.
  - Identity matching: `finalization_id`, `scope_id`, `candidate_id`, `candidate_digest`, and `evidence_sha256` must match the expected finalization record.
- **Enforcement points:** Integrated into `JobRepository.publish_finalization`, `JobRepository.read_published_final_result`, and `JobRepository._try_recover_staged_final_bundle`.
- **Regression coverage:** Added `test_recovery_rejects_bundle_with_invalid_result_payload` and `test_publish_finalization_rejects_invalid_result_payload` in `tests/integration/test_d19a_independent_review.py`.

### 5. Exact Serial Command Outputs

**Command 1: `uv run --locked pytest tests/integration/test_finalization.py -q`**
```text
..............................                                           [100%]
============================== warnings summary ===============================
tests/integration/test_finalization.py::test_equivalent_upload_deletion_removes_shared_partitions
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/integration/test_finalization.py::test_production_finalization_worker_evaluates_and_publishes_once_with_repeated_retrieval
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
30 passed, 2 warnings in 253.14s (0:04:13)
```

**Command 2: `uv run --locked pytest tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py tests/integration/test_resource_monitoring.py tests/integration/test_refit.py tests/integration/test_retention.py tests/integration/test_api.py -q`**
```text
.......................FFFF............................................. [ 63%]
.........................................                                [100%]
================================== FAILURES ===================================
______________ test_pre_start_failure_retry_requires_dead_child _______________
[run_cv_experiment -> assert job_record.state == 'completed' -> safe_error: 'resource_pressure']
_ test_durable_evaluation_start_issues_capability_and_makes_retry_impossible __
[run_cv_experiment -> assert job_record.state == 'completed' -> safe_error: 'resource_pressure']
_____________ test_fixture_evaluator_and_poison_holdout_isolation _____________
[run_cv_experiment -> assert job_record.state == 'completed' -> safe_error: 'resource_pressure']
______________ test_consumed_tombstone_prevents_new_reservation _______________
[run_cv_experiment -> assert job_record.state == 'completed' -> safe_error: 'resource_pressure']
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
tests/integration/test_holdout_gate.py::test_concurrent_contenders_yields_one_selection
  SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.
=========================== short test summary info ===========================
FAILED tests/integration/test_holdout_gate.py::test_pre_start_failure_retry_requires_dead_child
FAILED tests/integration/test_holdout_gate.py::test_durable_evaluation_start_issues_capability_and_makes_retry_impossible
FAILED tests/integration/test_holdout_gate.py::test_fixture_evaluator_and_poison_holdout_isolation
FAILED tests/integration/test_holdout_gate.py::test_consumed_tombstone_prevents_new_reservation
4 failed, 109 passed, 2 warnings in 857.10s (0:14:17)
```

**Command 3: `uv run --locked pytest tests/integration/test_d19a_independent_review.py -q`**
```text
.........                                                                [100%]
============================== warnings summary ===============================
tests/integration/test_d19a_independent_review.py::test_retry_rejects_altered_stored_plan
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
9 passed, 1 warning in 85.27s (0:01:25)
```

**Command 4: `uv lock --check`**
```text
Resolved 107 packages in 2ms
```

**Command 5: `git diff --check`**
```text
(exit 0)
```

**Command 6: `graphify update .`**
```text
[graphify watch] community set changed since labeling (73 saved labels, 74 communities now; renamed 47 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
Re-extracting code files in . (no LLM needed)...
[graphify watch] 4 file(s) not classified (no supported extension or shebang), skipped: .gitignore, .node-version, .python-version, uv.lock
[graphify] backed up curated graph (4 files) -> 2026-10-31/
[graphify watch] Rebuilt: 1501 nodes, 5167 edges, 74 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Tip: set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini for semantic extraction.
```


## Independent D19a safety review — 2026-10-31

**Status: BLOCKED; D19a is not accepted.** This independent review supersedes the acceptance claim immediately below. The checkbox is unchecked. The latest dispatch revision has not passed the affected fixture checks: repeated runs hit the existing `resource_pressure` guard. The resource cap was not relaxed. The owning suite passed 30 tests before the final dispatch change; the requested combined regression had 112 passes and one failure before that revision. A full clean post-revision acceptance run is still required.

Independent inspection of the actual working diff found and corrected: incomplete stored-plan comparison on pre-start retries; unfenced failure publication by stale supervisors; rollback of the cancellation lock at publication; a lease left occupied after live-supervisor staged recovery; missing candidate/scope binding in bundle publication and recovery; and production runner dispatch that never claimed finalization. The finalization-specific timeout is now used and worker limits must match the frozen run. Fixture red/green results below establish the first five fixes, while the final mixed-dispatch configuration remains review-pending because the real-worker fixture could not complete under memory pressure. An earlier mixed-dispatch revision caused the combined regression failure by taking an older finalization instead of a CV-only job; the revised runner enables mixed dispatch explicitly, preserving CV-only supervisors.

The inspected refit path loads a verified persisted training partition without opening the raw source or holdout; the sole production holdout-file read is in the capability-gated reader. The supervisor uses spawn and an IPC channel; the repository commits evaluation-start before issuing the capability. Pre-start retries check frozen candidate, evidence, plan, split and limits for complete published evidence; post-start failures remain locked unless a matching checksummed staged bundle is recovered. Equivalent-upload preparation/deletion and pending-deletion checks remain in place. Existing end-to-end fixtures assert one consumed read authorization, not a separate operating-system file-open count. These observations do not substitute for the pending clean production-worker and serial-regression runs. Bundle verification checks checksums and manifest identities; result payload schema validation remains a limitation.

Environment: Windows 11, PowerShell, locked local uv environment; fixture data only. At one observation during the failures, available memory was 4,795,265,024 bytes; the worker recorded `resource_pressure` for the affected CV jobs. No D19b, benchmark, external dataset download, real dataset finalization, dependency change, commit, push, G3 or G4 pass occurred. Preserve existing uncommitted work.

Exact captured outputs from this review follow; trailing spaces are represented as `\x20` so they remain visible and reversible. Captured paths and generated IDs are fixture-specific. The owning suite and combined regression were executed serially, with no graph update overlapping tests. Later focused checks are also serial.

**Initial requested owning suite: uv run --locked pytest tests/integration/test_finalization.py -q**

```text
..............................                                           [100%]
============================== warnings summary ===============================
tests/integration/test_finalization.py::test_equivalent_upload_deletion_removes_shared_partitions
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/integration/test_finalization.py::test_production_finalization_worker_evaluates_and_publishes_once_with_repeated_retrieval
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
30 passed, 2 warnings in 280.46s (0:04:40)
```

**First independent RED: uv run --locked pytest tests/integration/test_d19a_independent_review.py -q**

```text
FF                                                                       [100%]
================================== FAILURES ===================================
_________ test_stale_finalization_supervisor_cannot_release_new_lease _________

reservation = (<apps.api.database.repository.JobRepository object at 0x000001FF2C91CFB0>, PreparedExperiment(experiment_id='b4a5a9cc... {'candidates': [{'family': 'dummy', 'eligible': True, 'model_seed': 42, 'summary': [{'name': 'rmse', 'mean': 1.0}]}]})

    def test_stale_finalization_supervisor_cannot_release_new_lease(reservation):
        repo, _, _, selection, job, _ = reservation
        owner, claimed = claim_finalization(repo, job)
        worker = WorkerSupervisor(repo, workload="finalization", limits=reservation[2].limits)
        worker.owner = owner
        with repo._transaction() as db:
            db.execute("UPDATE worker_lease SET generation=generation+1, owner_token='new-owner' WHERE singleton=1")
>       with pytest.raises(ConflictError):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ConflictError

tests\integration\test_d19a_independent_review.py:21: Failed
___________________ test_retry_rejects_altered_stored_plan ____________________

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-233/test_retry_rejects_altered_sto0')

    def test_retry_rejects_altered_stored_plan(tmp_path):
        repo, prepared, _, candidate, digest, scope = _run_fixture_cv(tmp_path)
        selection, job = reserve(repo, prepared.experiment_id, candidate, digest, acknowledge=True)
        repo.request_cancel(job.job_id)
        with repo._transaction() as db:
            plan = json.loads(db.execute("SELECT plan_json FROM finalizations WHERE scope_id=?", (scope,)).fetchone()[0])
            plan["selection"]["resource_limits"]["compute_threads"] = 999
            db.execute("UPDATE finalizations SET plan_json=? WHERE scope_id=?", (json.dumps(plan), scope))
>       with pytest.raises(ConflictError):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ConflictError

tests\integration\test_d19a_independent_review.py:37: Failed
---------------------------- Captured stderr call -----------------------------
2026/10/31 19:47:57 INFO mlflow.agent.hint: Load the `instrumenting-with-mlflow-tracing` skill at C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\assistant\skills\instrumenting-with-mlflow-tracing\SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
2026/10/31 19:47:59 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/10/31 19:47:59 INFO mlflow.store.db.utils: Updating database tables
============================== warnings summary ===============================
tests/integration/test_d19a_independent_review.py::test_retry_rejects_altered_stored_plan
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/integration/test_d19a_independent_review.py::test_stale_finalization_supervisor_cannot_release_new_lease
FAILED tests/integration/test_d19a_independent_review.py::test_retry_rejects_altered_stored_plan
2 failed, 1 warning in 45.49s
```

**First independent GREEN: uv run --locked pytest tests/integration/test_d19a_independent_review.py -q**

```text
..                                                                       [100%]
============================== warnings summary ===============================
tests/integration/test_d19a_independent_review.py::test_retry_rejects_altered_stored_plan
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
2 passed, 1 warning in 45.15s
```

**Second independent RED: uv run --locked pytest tests/integration/test_d19a_independent_review.py -q -k 'cancel_before_publication or recovery_in_live or production_worker_claims'**

```text
FFF                                                                      [100%]
================================== FAILURES ===================================
________________ test_cancel_before_publication_durably_locks _________________

reservation = (<apps.api.database.repository.JobRepository object at 0x0000025A54CB5610>, PreparedExperiment(experiment_id='c1acaea0... {'candidates': [{'family': 'dummy', 'eligible': True, 'model_seed': 42, 'summary': [{'name': 'rmse', 'mean': 1.0}]}]})

    def test_cancel_before_publication_durably_locks(reservation):
        repo, _, _, selection, job, _ = reservation
        owner, claimed = claim_finalization(repo, job)
        with repo._transaction() as db:
            db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now',attempt_id=? WHERE finalization_id=?",
                       (claimed.attempt_id, selection.finalization_id))
        repo.request_cancel(job.job_id)
>       result = repo.publish_finalization(owner, job.job_id, claimed.attempt_id, selection.finalization_id,
                                           "unused", "unused")

tests\integration\test_d19a_independent_review.py:49:\x20
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <apps.api.database.repository.JobRepository object at 0x0000025A54CB5610>
owner = WorkerIdentity(owner_token='fixture-owner', generation=1)
job_id = '498815cb-ba83-48fc-bf41-a24ea834025f'
attempt_id = '178aa8be-5885-49fa-884c-21a637955f43'
finalization_id = '235d8ac8-66f3-4dfa-a0e3-ad31f852cdac', result_path = 'unused'
result_checksum = 'unused', telemetry = None

    def publish_finalization(
        self,
        owner: WorkerIdentity,
        job_id: str,
        attempt_id: str,
        finalization_id: str,
        result_path: str,
        result_checksum: str,
        *,
        telemetry: dict[str, object] | None = None,
    ) -> JobRecord:
        _text(job_id)
        _text(attempt_id)
        _text(finalization_id)
        if not isinstance(result_path, str) or not result_path.strip():
            raise ValueError("result_path must be a nonempty string")
        _text(result_checksum)
        if not isinstance(owner, WorkerIdentity):
            raise TypeError("Valid WorkerIdentity required")
        with self._transaction() as db:
            lease = db.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            if (lease["owner_token"] != owner.owner_token or lease["generation"] != owner.generation
                    or lease["active_job_id"] != job_id):
                raise ConflictError("Worker lease is stale or not owned")
\x20\x20\x20\x20
            job = db.execute("SELECT * FROM jobs WHERE job_id=? AND attempt_id=?", (job_id, attempt_id)).fetchone()
            if job is None or job["state"] != "running":
                raise ConflictError("Job is not running for this attempt")
\x20\x20\x20\x20
            fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (finalization_id,)).fetchone()
            if fin is None or fin["refit_job_id"] != job_id:
                raise ConflictError("Finalization record not found or job mismatch")
\x20\x20\x20\x20
            if job["cancel_requested"]:
                now = _now()
                db.execute("UPDATE jobs SET state='cancelled',stage='cancelled',updated_at=? WHERE job_id=?", (now, job_id))
                db.execute("UPDATE finalizations SET state='locked_failed',updated_at=? WHERE finalization_id=?", (now, finalization_id))
                db.execute("UPDATE worker_lease SET owner_token=NULL,active_job_id=NULL WHERE singleton=1")
                _audit(db, "worker", finalization_id, "finalization_cancelled_after_start_locked_failed")
>               raise ConflictError("Finalization cancelled after evaluation start; locked_failed")
E               apps.api.database.repository.ConflictError: Finalization cancelled after evaluation start; locked_failed

apps\api\database\repository.py:1109: ConflictError
_______________ test_recovery_in_live_supervisor_releases_lease _______________

reservation = (<apps.api.database.repository.JobRepository object at 0x0000025A5657BCE0>, PreparedExperiment(experiment_id='2ec0d8da... {'candidates': [{'family': 'dummy', 'eligible': True, 'model_seed': 42, 'summary': [{'name': 'rmse', 'mean': 1.0}]}]})

    def test_recovery_in_live_supervisor_releases_lease(reservation):
        repo, _, run, selection, job, _ = reservation
        owner, claimed = claim_finalization(repo, job)
        with repo._transaction() as db:
            db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now',attempt_id=? WHERE finalization_id=?",
                       (claimed.attempt_id, selection.finalization_id))
        stage_final_bundle(repo.path.parent, selection.finalization_id, selection.scope_id,
                           claimed.attempt_id, selection.candidate_id, {"ok": True})
        worker = WorkerSupervisor(repo, workload="finalization", limits=run.limits)
        worker.owner = owner
        result = worker._publish(claimed, claimed.attempt_id, _Outcome("failed", "failed", "child_exited"))
        assert result.state == "completed"
        with repo._connection() as db:
>           assert db.execute("SELECT active_job_id FROM worker_lease WHERE singleton=1").fetchone()[0] is None
E           AssertionError: assert 'f6c204b4-94a9-4a3b-886b-208703eeabde' is None

tests\integration\test_d19a_independent_review.py:69: AssertionError
_________________ test_production_worker_claims_finalization __________________

reservation = (<apps.api.database.repository.JobRepository object at 0x0000025A5657AF30>, PreparedExperiment(experiment_id='153accab... {'candidates': [{'family': 'dummy', 'eligible': True, 'model_seed': 42, 'summary': [{'name': 'rmse', 'mean': 1.0}]}]})
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x0000025A565E0710>

    def test_production_worker_claims_finalization(reservation, monkeypatch):
        repo, _, run, _, job, _ = reservation
        with repo._transaction() as db:
            db.execute("UPDATE jobs SET state='cancelled' WHERE kind='cv' AND state='queued'")
        worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, limits=run.limits)
        monkeypatch.setattr(worker, "_execute", lambda claimed: claimed)
>       assert worker.run_once().job_id == job.job_id
               ^^^^^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'NoneType' object has no attribute 'job_id'

tests\integration\test_d19a_independent_review.py:78: AttributeError
=========================== short test summary info ===========================
FAILED tests/integration/test_d19a_independent_review.py::test_cancel_before_publication_durably_locks
FAILED tests/integration/test_d19a_independent_review.py::test_recovery_in_live_supervisor_releases_lease
FAILED tests/integration/test_d19a_independent_review.py::test_production_worker_claims_finalization
3 failed, 2 deselected in 11.93s
```

**Independent GREEN before bundle check: uv run --locked pytest tests/integration/test_d19a_independent_review.py -q**

```text
.....                                                                    [100%]
============================== warnings summary ===============================
tests/integration/test_d19a_independent_review.py::test_retry_rejects_altered_stored_plan
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
5 passed, 1 warning in 45.26s
```

**Bundle identity RED: uv run --locked pytest tests/integration/test_d19a_independent_review.py::test_recovery_rejects_bundle_with_wrong_candidate -q**

```text
F                                                                        [100%]
================================== FAILURES ===================================
______________ test_recovery_rejects_bundle_with_wrong_candidate ______________

reservation = (<apps.api.database.repository.JobRepository object at 0x0000024EA0322C60>, PreparedExperiment(experiment_id='a89643df... {'candidates': [{'family': 'dummy', 'eligible': True, 'model_seed': 42, 'summary': [{'name': 'rmse', 'mean': 1.0}]}]})

    def test_recovery_rejects_bundle_with_wrong_candidate(reservation):
        repo, _, _, selection, job, _ = reservation
        _, claimed = claim_finalization(repo, job)
        with repo._transaction() as db:
            db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now',attempt_id=? WHERE finalization_id=?",
                       (claimed.attempt_id, selection.finalization_id))
        stage_final_bundle(repo.path.parent, selection.finalization_id, selection.scope_id,
                           claimed.attempt_id, "wrong-candidate", {"ok": True})
        with repo._transaction() as db:
            fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (selection.finalization_id,)).fetchone()
>           assert not repo._try_recover_staged_final_bundle(db, fin, claimed.attempt_id, "now")
E           AssertionError: assert not True
E            +  where True = _try_recover_staged_final_bundle(<sqlite3.Connection object at 0x0000024EA1AD8400>, <sqlite3.Row object at 0x0000024EA1AC79A0>, 'b7f3cc51-c577-448d-bb96-3ca70f058bd7', 'now')
E            +    where _try_recover_staged_final_bundle = <apps.api.database.repository.JobRepository object at 0x0000024EA0322C60>._try_recover_staged_final_bundle
E            +    and   'b7f3cc51-c577-448d-bb96-3ca70f058bd7' = JobRecord(job_id='e4fe5388-8352-48e1-8ed1-d455215075dd', experiment_id='a89643df-2ebf-4e9f-b7ed-44e55621b437', kind='f...0:00', updated_at='2026-10-31T12:55:13.022131+00:00', heartbeat_at='2026-10-31T12:55:13.022131+00:00', safe_error=None).attempt_id

tests\integration\test_d19a_independent_review.py:91: AssertionError
=========================== short test summary info ===========================
FAILED tests/integration/test_d19a_independent_review.py::test_recovery_rejects_bundle_with_wrong_candidate
1 failed in 11.03s
```

**Bundle identity GREEN: uv run --locked pytest tests/integration/test_d19a_independent_review.py::test_recovery_rejects_bundle_with_wrong_candidate -q**

```text
.                                                                        [100%]
1 passed in 12.27s
```

**Requested owning suite after bundle fix, before final dispatch revision: uv run --locked pytest tests/integration/test_finalization.py -q**

```text
..............................                                           [100%]
============================== warnings summary ===============================
tests/integration/test_finalization.py::test_equivalent_upload_deletion_removes_shared_partitions
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/integration/test_finalization.py::test_production_finalization_worker_evaluates_and_publishes_once_with_repeated_retrieval
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
30 passed, 2 warnings in 260.73s (0:04:20)
```

**Requested combined regressions, before final dispatch revision: uv run --locked pytest tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py tests/integration/test_resource_monitoring.py tests/integration/test_refit.py tests/integration/test_retention.py tests/integration/test_api.py -q**

```text
......................F................................................. [ 63%]
.........................................                                [100%]
================================== FAILURES ===================================
___________ test_scope_lock_denies_competing_experiment_same_scope ____________

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-241/test_scope_lock_denies_competi0')

    def test_scope_lock_denies_competing_experiment_same_scope(tmp_path):
        repo, app, dataset_id, config_id, experiment_id1, scope_id, leaderboard = run_cv_experiment(tmp_path)
        cand = leaderboard["ranked"][0]
\x20\x20\x20\x20
        with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
            digest, _ = candidate_digest_from_client(client, cand["candidate_id"], cand["mean"])
            # Reserve on experiment 1
            res1 = client.post(
                f"/api/v1/experiments/{experiment_id1}/finalization",
                json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
            )
            assert res1.status_code == 202
\x20\x20\x20\x20
            # Create experiment 2 with different model seed on the same configuration (same evaluation scope)
            exp2_res = client.post("/api/v1/experiments", json={"configuration_id": config_id, "model_seed": 999})
            assert exp2_res.status_code == 201
            experiment_id2 = exp2_res.json()["experiment_id"]
\x20\x20\x20\x20
            # Run CV for experiment 2
            client.post(f"/api/v1/experiments/{experiment_id2}/jobs", json={"idempotency_key": "cv-job-2"})
            worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
            worker.run_once()
\x20\x20\x20\x20
            lb2 = client.get(f"/api/v1/experiments/{experiment_id2}/leaderboard").json()
>           cand2 = lb2["ranked"][0]
                    ^^^^^^^^^^^^^^^^
E           IndexError: list index out of range

tests\integration\test_holdout_gate.py:239: IndexError
---------------------------- Captured stderr call -----------------------------
2026/10/31 20:04:28 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/10/31 20:04:28 INFO mlflow.store.db.utils: Updating database tables
------------------------------ Captured log call ------------------------------
INFO     mlflow.store.db.utils:utils.py:88 Creating initial MLflow database tables...
INFO     mlflow.store.db.utils:utils.py:295 Updating database tables
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/integration/test_holdout_gate.py::test_concurrent_contenders_yields_one_selection
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/integration/test_holdout_gate.py::test_scope_lock_denies_competing_experiment_same_scope
1 failed, 112 passed, 2 warnings in 606.58s (0:10:06)
```

**Focused post-dispatch check: uv run --locked pytest tests/integration/test_holdout_gate.py::test_scope_lock_denies_competing_experiment_same_scope tests/integration/test_d19a_independent_review.py -q**

```text
F.F...F.                                                                 [100%]
================================== FAILURES ===================================
___________ test_scope_lock_denies_competing_experiment_same_scope ____________

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-242/test_scope_lock_denies_competi0')

    def test_scope_lock_denies_competing_experiment_same_scope(tmp_path):
        repo, app, dataset_id, config_id, experiment_id1, scope_id, leaderboard = run_cv_experiment(tmp_path)
        cand = leaderboard["ranked"][0]
\x20\x20\x20\x20
        with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
            digest, _ = candidate_digest_from_client(client, cand["candidate_id"], cand["mean"])
            # Reserve on experiment 1
            res1 = client.post(
                f"/api/v1/experiments/{experiment_id1}/finalization",
                json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
            )
            assert res1.status_code == 202
\x20\x20\x20\x20
            # Create experiment 2 with different model seed on the same configuration (same evaluation scope)
            exp2_res = client.post("/api/v1/experiments", json={"configuration_id": config_id, "model_seed": 999})
            assert exp2_res.status_code == 201
            experiment_id2 = exp2_res.json()["experiment_id"]
\x20\x20\x20\x20
            # Run CV for experiment 2
            client.post(f"/api/v1/experiments/{experiment_id2}/jobs", json={"idempotency_key": "cv-job-2"})
            worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
            worker.run_once()
\x20\x20\x20\x20
            lb2 = client.get(f"/api/v1/experiments/{experiment_id2}/leaderboard").json()
>           cand2 = lb2["ranked"][0]
                    ^^^^^^^^^^^^^^^^
E           IndexError: list index out of range

tests\integration\test_holdout_gate.py:239: IndexError
---------------------------- Captured stderr call -----------------------------
2026/10/31 20:11:20 INFO mlflow.agent.hint: Load the `instrumenting-with-mlflow-tracing` skill at C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\assistant\skills\instrumenting-with-mlflow-tracing\SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
2026/10/31 20:11:22 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/10/31 20:11:22 INFO mlflow.store.db.utils: Updating database tables
___________________ test_retry_rejects_altered_stored_plan ____________________

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-242/test_retry_rejects_altered_sto0')

    def test_retry_rejects_altered_stored_plan(tmp_path):
>       repo, prepared, _, candidate, digest, scope = _run_fixture_cv(tmp_path)
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^

tests\integration\test_d19a_independent_review.py:31:\x20
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-242/test_retry_rejects_altered_sto0')
task = <TaskType.REGRESSION: 'regression'>, family = 'dummy'

    def _run_fixture_cv(tmp_path, task=TaskType.REGRESSION, family="dummy"):
        import shutil
        from ml.contracts import ResourceLimits, RunConfig
        from ml.data.csv_io import validate_csv
        from apps.api.services.datasets import upload_root
        from apps.api.services.jobs import prepare_experiment
        from apps.api.workers.supervisor import WorkerSupervisor
\x20\x20\x20\x20
        limits = ResourceLimits(compute_threads=2, max_epochs=2, mlp_fold_seconds=15)
        run = RunConfig(limits=limits)
        repo, prepared, run, job = prepared_fixture(tmp_path, task, run=run)
        with repo._connection() as db:
            dataset_id = db.execute("SELECT c.dataset_id FROM experiments e JOIN configurations c "
                                    "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
                                    (prepared.experiment_id,)).fetchone()[0]
        dest_dir = upload_root(repo) / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / "input.csv", dest_dir / "source.csv")
\x20\x20\x20\x20
        worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, limits=limits)
        completed_cv = worker.run_once()
>       assert completed_cv is not None and completed_cv.state == "completed"
E       AssertionError: assert (JobRecord(job_id='9e41aa9f-8e82-4c48-a73a-35eab039e55d', experiment_id='921be166-9ffc-4bc2-8b2d-3abe0860a922', kind='c...at='2026-10-31T13:12:04.984147+00:00', heartbeat_at='2026-10-31T13:12:04.984147+00:00', safe_error='resource_pressure') is not None and 'failed' == 'completed'
E\x20\x20\x20\x20\x20\x20\x20\x20\x20
E         - completed
E         + failed)

tests\integration\test_finalization.py:458: AssertionError
______ test_normal_stored_worker_executes_finalization_in_spawned_child _______

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-242/test_normal_stored_worker_exec0')

    def test_normal_stored_worker_executes_finalization_in_spawned_child(tmp_path):
>       repo, prepared, run, candidate, digest, scope = _run_fixture_cv(tmp_path)
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^

tests\integration\test_d19a_independent_review.py:82:\x20
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-242/test_normal_stored_worker_exec0')
task = <TaskType.REGRESSION: 'regression'>, family = 'dummy'

    def _run_fixture_cv(tmp_path, task=TaskType.REGRESSION, family="dummy"):
        import shutil
        from ml.contracts import ResourceLimits, RunConfig
        from ml.data.csv_io import validate_csv
        from apps.api.services.datasets import upload_root
        from apps.api.services.jobs import prepare_experiment
        from apps.api.workers.supervisor import WorkerSupervisor
\x20\x20\x20\x20
        limits = ResourceLimits(compute_threads=2, max_epochs=2, mlp_fold_seconds=15)
        run = RunConfig(limits=limits)
        repo, prepared, run, job = prepared_fixture(tmp_path, task, run=run)
        with repo._connection() as db:
            dataset_id = db.execute("SELECT c.dataset_id FROM experiments e JOIN configurations c "
                                    "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
                                    (prepared.experiment_id,)).fetchone()[0]
        dest_dir = upload_root(repo) / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / "input.csv", dest_dir / "source.csv")
\x20\x20\x20\x20
        worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, limits=limits)
        completed_cv = worker.run_once()
>       assert completed_cv is not None and completed_cv.state == "completed"
E       AssertionError: assert (JobRecord(job_id='857b48d8-3752-478a-9f62-b5fcfda7018f', experiment_id='b2a15596-bc87-4ded-9506-5f95317ee0b0', kind='c...at='2026-10-31T13:12:17.255752+00:00', heartbeat_at='2026-10-31T13:12:17.255752+00:00', safe_error='resource_pressure') is not None and 'failed' == 'completed'
E\x20\x20\x20\x20\x20\x20\x20\x20\x20
E         - completed
E         + failed)

tests\integration\test_finalization.py:458: AssertionError
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/integration/test_holdout_gate.py::test_scope_lock_denies_competing_experiment_same_scope
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/integration/test_holdout_gate.py::test_scope_lock_denies_competing_experiment_same_scope
FAILED tests/integration/test_d19a_independent_review.py::test_retry_rejects_altered_stored_plan
FAILED tests/integration/test_d19a_independent_review.py::test_normal_stored_worker_executes_finalization_in_spawned_child
3 failed, 5 passed, 2 warnings in 99.08s (0:01:39)
```

**Focused post-dispatch retry: uv run --locked pytest tests/integration/test_holdout_gate.py::test_scope_lock_denies_competing_experiment_same_scope tests/integration/test_d19a_independent_review.py -q**

```text
F.F...F.                                                                 [100%]
================================== FAILURES ===================================
___________ test_scope_lock_denies_competing_experiment_same_scope ____________

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-243/test_scope_lock_denies_competi0')

    def test_scope_lock_denies_competing_experiment_same_scope(tmp_path):
        repo, app, dataset_id, config_id, experiment_id1, scope_id, leaderboard = run_cv_experiment(tmp_path)
        cand = leaderboard["ranked"][0]
\x20\x20\x20\x20
        with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
            digest, _ = candidate_digest_from_client(client, cand["candidate_id"], cand["mean"])
            # Reserve on experiment 1
            res1 = client.post(
                f"/api/v1/experiments/{experiment_id1}/finalization",
                json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
            )
            assert res1.status_code == 202
\x20\x20\x20\x20
            # Create experiment 2 with different model seed on the same configuration (same evaluation scope)
            exp2_res = client.post("/api/v1/experiments", json={"configuration_id": config_id, "model_seed": 999})
            assert exp2_res.status_code == 201
            experiment_id2 = exp2_res.json()["experiment_id"]
\x20\x20\x20\x20
            # Run CV for experiment 2
            client.post(f"/api/v1/experiments/{experiment_id2}/jobs", json={"idempotency_key": "cv-job-2"})
            worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
            worker.run_once()
\x20\x20\x20\x20
            lb2 = client.get(f"/api/v1/experiments/{experiment_id2}/leaderboard").json()
>           cand2 = lb2["ranked"][0]
                    ^^^^^^^^^^^^^^^^
E           IndexError: list index out of range

tests\integration\test_holdout_gate.py:239: IndexError
---------------------------- Captured stderr call -----------------------------
2026/10/31 20:13:15 INFO mlflow.agent.hint: Load the `instrumenting-with-mlflow-tracing` skill at C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\assistant\skills\instrumenting-with-mlflow-tracing\SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
2026/10/31 20:13:17 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/10/31 20:13:17 INFO mlflow.store.db.utils: Updating database tables
___________________ test_retry_rejects_altered_stored_plan ____________________

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-243/test_retry_rejects_altered_sto0')

    def test_retry_rejects_altered_stored_plan(tmp_path):
>       repo, prepared, _, candidate, digest, scope = _run_fixture_cv(tmp_path)
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^

tests\integration\test_d19a_independent_review.py:31:\x20
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-243/test_retry_rejects_altered_sto0')
task = <TaskType.REGRESSION: 'regression'>, family = 'dummy'

    def _run_fixture_cv(tmp_path, task=TaskType.REGRESSION, family="dummy"):
        import shutil
        from ml.contracts import ResourceLimits, RunConfig
        from ml.data.csv_io import validate_csv
        from apps.api.services.datasets import upload_root
        from apps.api.services.jobs import prepare_experiment
        from apps.api.workers.supervisor import WorkerSupervisor
\x20\x20\x20\x20
        limits = ResourceLimits(compute_threads=2, max_epochs=2, mlp_fold_seconds=15)
        run = RunConfig(limits=limits)
        repo, prepared, run, job = prepared_fixture(tmp_path, task, run=run)
        with repo._connection() as db:
            dataset_id = db.execute("SELECT c.dataset_id FROM experiments e JOIN configurations c "
                                    "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
                                    (prepared.experiment_id,)).fetchone()[0]
        dest_dir = upload_root(repo) / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / "input.csv", dest_dir / "source.csv")
\x20\x20\x20\x20
        worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, limits=limits)
        completed_cv = worker.run_once()
>       assert completed_cv is not None and completed_cv.state == "completed"
E       AssertionError: assert (JobRecord(job_id='79cb8676-0922-40c1-91cd-473db09c6a28', experiment_id='ed16b75b-ccc6-4d51-98fa-4c2338274814', kind='c...at='2026-10-31T13:14:08.060954+00:00', heartbeat_at='2026-10-31T13:14:08.060954+00:00', safe_error='resource_pressure') is not None and 'failed' == 'completed'
E\x20\x20\x20\x20\x20\x20\x20\x20\x20
E         - completed
E         + failed)

tests\integration\test_finalization.py:458: AssertionError
______ test_normal_stored_worker_executes_finalization_in_spawned_child _______

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-243/test_normal_stored_worker_exec0')

    def test_normal_stored_worker_executes_finalization_in_spawned_child(tmp_path):
>       repo, prepared, run, candidate, digest, scope = _run_fixture_cv(tmp_path)
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^

tests\integration\test_d19a_independent_review.py:82:\x20
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

tmp_path = WindowsPath('C:/Users/Reshandy Taftazani A/AppData/Local/Temp/pytest-of-Reshandy Taftazani A/pytest-243/test_normal_stored_worker_exec0')
task = <TaskType.REGRESSION: 'regression'>, family = 'dummy'

    def _run_fixture_cv(tmp_path, task=TaskType.REGRESSION, family="dummy"):
        import shutil
        from ml.contracts import ResourceLimits, RunConfig
        from ml.data.csv_io import validate_csv
        from apps.api.services.datasets import upload_root
        from apps.api.services.jobs import prepare_experiment
        from apps.api.workers.supervisor import WorkerSupervisor
\x20\x20\x20\x20
        limits = ResourceLimits(compute_threads=2, max_epochs=2, mlp_fold_seconds=15)
        run = RunConfig(limits=limits)
        repo, prepared, run, job = prepared_fixture(tmp_path, task, run=run)
        with repo._connection() as db:
            dataset_id = db.execute("SELECT c.dataset_id FROM experiments e JOIN configurations c "
                                    "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
                                    (prepared.experiment_id,)).fetchone()[0]
        dest_dir = upload_root(repo) / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / "input.csv", dest_dir / "source.csv")
\x20\x20\x20\x20
        worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, limits=limits)
        completed_cv = worker.run_once()
>       assert completed_cv is not None and completed_cv.state == "completed"
E       AssertionError: assert (JobRecord(job_id='9db86f8f-c7e7-43ef-8c35-ee51e0822630', experiment_id='6de1692e-e540-4c09-94ac-626aef7bfabb', kind='c...at='2026-10-31T13:14:24.529726+00:00', heartbeat_at='2026-10-31T13:14:24.529726+00:00', safe_error='resource_pressure') is not None and 'failed' == 'completed'
E\x20\x20\x20\x20\x20\x20\x20\x20\x20
E         - completed
E         + failed)

tests\integration\test_finalization.py:458: AssertionError
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/integration/test_holdout_gate.py::test_scope_lock_denies_competing_experiment_same_scope
  C:\Users\Reshandy Taftazani A\Documents\Personal Project\ModelArena\.venv\Lib\site-packages\mlflow\store\tracking\sqlalchemy_store.py:327: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1)
    sqlalchemy.orm.configure_mappers()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/integration/test_holdout_gate.py::test_scope_lock_denies_competing_experiment_same_scope
FAILED tests/integration/test_d19a_independent_review.py::test_retry_rejects_altered_stored_plan
FAILED tests/integration/test_d19a_independent_review.py::test_normal_stored_worker_executes_finalization_in_spawned_child
3 failed, 5 passed, 2 warnings in 107.63s (0:01:47)
```

**`uv lock --check` (exit 0)**

```text
Resolved 107 packages in 2ms
```



**`git diff --check` (exit 0; before recording graph output)**

```text
warning: in the working copy of 'apps/api/database/repository.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/database/schema.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/main.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/services/finalization.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/services/jobs.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/services/retention.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/services/workflow.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/workers/control.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/workers/execute.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/workers/runner.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'apps/api/workers/supervisor.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'ml/training/refit.py', LF will be replaced by CRLF the next time Git touches it
```

**`graphify update .` (exit 0; PowerShell combined output)**

```text
graphify : [graphify watch] community set changed since labeling (69 saved labels, 73 communities now; renamed 47\x20
community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
At line:2 char:1
+ graphify update . *> "$env:TEMP\modelarena_d19a_graph.log"; $code=$LA ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: ([graphify watch...s with the LLM.:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
\x20
Re-extracting code files in . (no LLM needed)...
[graphify watch] 4 file(s) not classified (no supported extension or shebang), skipped: .gitignore, .node-version, .python-version, uv.lock
[graphify] backed up curated graph (4 files) -> 2026-10-31/
[graphify watch] Rebuilt: 1496 nodes, 5152 edges, 73 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Tip: set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini for semantic extraction.
```

Graph update is structural/code-only; documentation semantic extraction was not run. Graphify reported renamed community labels and four unsupported files.

## Day 19a acceptance — production finalization worker, holdout gate, and crash recovery, 2026-10-31

- D19a is **ACCEPTED** and checked in `docs/IMPLEMENTATION_PLAN.md`. All six acceptance requirements are fully verified through RED/GREEN integration tests and isolated serial regressions:
  1. **Durable CV evidence & budget binding:** `reserve_finalization` durably binds `evidence_sha256`, `epoch_budget`, and `plan_json` into `finalizations` under the metadata transaction. Pre-evaluation retries strictly verify evidence digest, selection, configuration, split, limits, and epoch budget identity (`test_reservation_durably_binds_evidence_and_budget_enforcing_identity`).
  2. **Strict training/holdout input boundary:** `refit_selected` consumes only the persisted training partition (`training_from_manifest`) without opening holdout or source data. Source-tampering and concurrent equivalent-upload deletion/preparation race protections remain enforced.
  3. **Production finalization child & holdout capability gate:** `WorkerSupervisor` claims `kind="finalization"` jobs and spawns dedicated execution children under Windows `spawn` multiprocessing. The child refits the selected model, requests authorization via `_evaluation_pipe`, receives `HoldoutCapability` only upon durable evaluation start in the database, and accesses holdout solely via `read_holdout`. Worker ownership, singleton lease fencing, cancellation, and tombstone checks are preserved (`test_production_finalization_stale_attempt_fencing`).
  4. **Single holdout score, staged bundle & idempotent retrieval:** Holdout is scored once (`score_holdout`). The child stages a checksum-verified directory bundle (`manifest.json`, `results.json`) into `staging_final_bundle`. Fenced publication commits the manifest hash and path atomically. Repeated GET and POST requests retrieve the exact published result without refitting or repeated holdout reads (`test_production_finalization_worker_evaluates_and_publishes_once_with_repeated_retrieval`).
  5. **Atomic crash recovery without recomputation:** If supervisor or worker crashes after staging, `reconcile_worker` atomically moves and publishes the valid complete staged bundle without re-reading holdout or recomputing. A crash immediately after evaluation-start but before staging locks the reservation (`locked_failed`) and prohibits holdout re-evaluation (`test_production_finalization_complete_staged_bundle_recovery_without_recomputation`, `test_production_finalization_post_start_crash_remains_locked`, `test_production_finalization_pre_start_failure_permits_identical_retry`).
- **RED/GREEN Test Evidence:**
  - RED phase: genuine initial failures demonstrated lack of IPC evaluation capability coordination, missing candidate metric values during JSON fold reconstruction, and unverified supervisor liveness (`uv run --locked pytest tests/integration/test_finalization.py -q`: 6 failed).
  - GREEN phase: owning suite `uv run --locked pytest tests/integration/test_finalization.py -q` passed **30 passed**, zero skips, two upstream Starlette/MLflow deprecation warnings, 129.05 s.
- **Serial Regression Verification:**
  - `tests/integration/test_finalization.py`: **30 passed** (129.05 s)
  - `tests/integration/test_review_finalization_regressions.py`: **17 passed** (6.90 s)
  - `tests/integration/test_holdout_gate.py`: **10 passed** (239.01 s)
  - `tests/integration/test_worker_lifecycle.py`: **21 passed** (42.59 s)
  - `tests/integration/test_worker_spawn.py`: **11 passed** (13.17 s)
  - `tests/integration/test_resource_monitoring.py`: **15 passed** (19.07 s)
  - `tests/integration/test_refit.py`: **12 passed** (11.36 s)
  - `tests/integration/test_retention.py`: **13 passed** (8.16 s)
  - `tests/integration/test_api.py`: **14 passed** (45.58 s)
  - Total: **143 tests passed**, zero failures, zero skips across all 9 test suites run strictly serially. No memory pressure (`resource_pressure`) failures occurred during isolated serial execution.
- **Knowledge Graph & System Integrity:**
  - Rebuilt knowledge graph via `graphify update .`: **1,485 nodes, 5,093 edges, 69 communities**. No graph update was overlapped with resource-sensitive test execution.
  - `uv lock --check` passed cleanly; `git diff --check` passed cleanly with zero whitespace or line-ending anomalies.
  - Existing uncommitted changes preserved; no D19b work, benchmark, dataset download, real dataset finalization, dependency modification, commit, or push was performed.

- D19a remains unchecked. A finalization job is still never claimed by the production supervisor; no durable complete CV-evidence/refit-plan binding, real finalization child, atomic final result publication/retrieval, or staged-bundle recovery exists. Configuration and CV still materialize the full upload. No real-worker pre/post-start crash and cancellation, stale publication, or exactly-one-read acceptance has been demonstrated. These are implementation blockers, not passing acceptance evidence.
- RED: `uv run --locked pytest tests/integration/test_finalization.py::test_retry_after_equivalent_dataset_deletion_is_rejected tests/integration/test_finalization.py::test_equivalent_upload_deletion_serializes_with_partition_preparation -q` - **1 failed, 1 passed**, zero skips, 12.74 s. After actual deletion of a canonically equivalent dataset, the repository accepted a retry. The initial red version registered equivalent fixture metadata directly; the final version uses the real upload service and different CSV line endings.
- GREEN: the same command after adding deletion-intent checks - **2 passed**, zero skips, 12.95 s. After replacing synthetic equivalent metadata with an actual equivalent fixture upload and adding an initial-enqueue assertion, the same command passed **2**, zero skips, 12.46 s. Both enqueue and retry now check for any equivalent dataset's deletion intent under the SQLite write transaction used by deletion. The concurrent test pauses partition preparation inside that transaction, starts real deletion of the equivalent upload, verifies it cannot complete until preparation releases the lock, and verifies cleanup removes the shared partition. The final test additionally checks that a later preparation fails without recreating it.
- Final owning command, after strengthening the race test: `uv run --locked pytest tests/integration/test_finalization.py -q` - **24 passed**, zero skips, one upstream Starlette warning, 31.16 s. It is groundwork only and does not establish D19a acceptance.
- Serial regressions: `uv run --locked pytest tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py tests/integration/test_resource_monitoring.py tests/integration/test_refit.py tests/integration/test_retention.py tests/integration/test_api.py -q` - **113 passed**, zero skips, two upstream warnings, 632.46 s. Fixture data only. No graph update overlapped the run. This pass does not explain the earlier three `resource_pressure` failures, whose cause remains unresolved.
- After the tests, `graphify update .` refreshed the AST-only graph: **1,466 nodes, 4,942 edges, 81 communities**, four unsupported files skipped, and 54 community names replaced with hub labels. Documentation semantic extraction was not run.
- Existing uncommitted changes were preserved. No D19b, benchmark, dataset download, real-dataset finalization, dependency installation, resource-limit change, commit, or push was performed; no G3/G4 pass is claimed.


## Day 19a continuation — incomplete; equivalent-upload deletion, 2026-10-31

- Preserved existing uncommitted changes; D19a remains unchecked. The production supervisor still claims CV jobs only, the refit plan and complete CV evidence are not durably bound, and finalization status returns no result. The production execution, atomic publication/recovery, and actual-worker crash/cancellation acceptance cases remain unimplemented and unverified. Existing green tests are groundwork, not D19a acceptance.
- RED: `uv run --locked pytest tests/integration/test_finalization.py::test_equivalent_upload_deletion_removes_shared_partitions -q` — **1 failed**, zero skips, 13.05 s. Deleting an equivalent upload without its own configuration left the canonical scope's partition files. GREEN: the same command — **1 passed**, zero skips, 12.89 s. Cleanup now discovers shared partition scopes by canonical hash. The existing intent/partition-write transaction prevents recreation after an intent for equivalent content; the new test checks real cleanup, not a concurrent equivalent-upload race. Other upload copies, metadata and MLflow copies still remain under the existing incomplete retention policy; no confirmed deletion is claimed.
- Owning suite: `uv run --locked pytest tests/integration/test_finalization.py -q` — **22 passed**, zero skips, one upstream warning, 31.47 s, Windows locked environment. This does not demonstrate production finalization, durable evidence binding, staged recovery or exactly one real worker holdout read.
- Serial regression batch: `uv run --locked pytest tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py tests/integration/test_resource_monitoring.py tests/integration/test_refit.py tests/integration/test_retention.py tests/integration/test_api.py -q` — **110 passed, 3 failed**, zero skips, two upstream warnings, 568.89 s. The failures were `test_scope_lock_denies_competing_experiment_same_scope`, `test_pre_start_failure_retry_requires_dead_child`, and `test_durable_evaluation_start_issues_capability_and_makes_retry_impossible`. Each failed before its gate assertion because a fixture CV job failed; the affected SQLite job records report `resource_pressure`. The cause of the memory pressure was not established. The code-only graph update ran while this regression batch was underway, which may have contributed to contention; this has not been established. No caps were changed. An isolated serial rerun, `uv run --locked pytest tests/integration/test_holdout_gate.py::test_scope_lock_denies_competing_experiment_same_scope tests/integration/test_holdout_gate.py::test_pre_start_failure_retry_requires_dead_child tests/integration/test_holdout_gate.py::test_durable_evaluation_start_issues_capability_and_makes_retry_impossible -q`, passed **3**, zero skips, two warnings, 184.84 s. This does not identify the original pressure cause.
- `graphify update .` rebuilt the AST-only graph: **1,458 nodes, 4,911 edges, 76 communities**; four unsupported files skipped, 48 community names replaced by hub labels. Documentation semantic extraction was not run.
- No D19b, benchmark, download, real-dataset finalization, dependency/resource-limit change, commit or push. No G3/G4 pass is claimed.
## Day 19a continuation â€” incomplete; preparation/deletion race, 2026-10-31

- Preserved the pre-existing uncommitted D19a work. D19a remains **unchecked**:
  the production supervisor still claims only CV jobs, and finalization status
  does not return a published result. Existing green tests are not acceptance
  evidence for the missing finalization execution and recovery path.
- Investigated a real check-to-write race between private partition preparation
  and deletion. RED: `uv run --locked pytest
  tests/integration/test_finalization.py::test_partition_preparation_cannot_recreate_files_after_deletion
  -q` â€” **1 failed**, zero skips, 6.11 s: deletion finished after the folder
  check, and preparation recreated the deleted private scope directory. The
  fixture pauses preparation at that boundary and runs actual deletion in a
  second thread; it does not use a simulated deletion result.
- GREEN: the same command â€” **1 passed**, zero skips, 8.94 s. Partition content
  and checksums are computed before acquiring the SQLite write transaction;
  the pending-deletion check and both no-clobber writes occur while holding
  the same write lock used to record a deletion intent. Added a second test for
  deletion winning before preparation. The file writes and fsyncs hold that
  transaction, so slow storage can temporarily block other metadata writers;
  the payload is bounded by existing upload limits and no training runs under
  the lock. This is a correctness tradeoff, not a measured latency guarantee.
  This regression covers the same dataset/scope; equivalent uploads and the
  broader incomplete retention purge have not been proven by these tests.
- Serial owning command: `uv run --locked pytest
  tests/integration/test_finalization.py -q` â€” **21 passed**, zero skips,
  20.14 s. This remains a groundwork suite, not the requested production-worker
  acceptance suite. No assertion was removed or weakened.
- Serial regressions: `uv run --locked pytest
  tests/integration/test_review_finalization_regressions.py
  tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py
  tests/integration/test_worker_spawn.py tests/integration/test_resource_monitoring.py
  tests/integration/test_refit.py tests/integration/test_retention.py
  tests/integration/test_api.py -q` â€” **113 passed**, zero skips, two upstream
  deprecation warnings, 637.16 s. Windows, existing locked Python environment;
  fixture data only. These suites include fixture evaluators and seeded gate
  states; they do not prove production finalization or one actual holdout read.
- `graphify update .` completed the AST-only update: **1,456 nodes, 4,904
  edges, 77 communities**; four unsupported files skipped, 52 community names
  replaced by hub labels, and semantic documentation extraction not run.
- **Unresolved blockers:** durable checksum-verified complete CV evidence and
  frozen refit plan/MLP budget are not bound to reservation/retries; initial
  configuration and CV still reopen the full source; production child dispatch,
  gated refit/scoring, fenced result staging/publication and repeated result
  retrieval, and staged-bundle crash recovery are absent. No behavioral tests
  yet prove real-worker pre/post-start cancellation, death and stale-attempt
  fencing, crash immediately after durable start, or exactly one actual holdout
  read across repeated requests. Do not enable production finalization until
  those boundaries are implemented and verified together. No D19b, benchmark,
  download, real-dataset finalization, dependency/resource-limit change, commit,
  or push was performed; no G3/G4 pass is claimed.

## Day 19a continuation â€” incomplete; partition and gated-reader groundwork, 2026-10-31

- Preserved the pre-existing uncommitted dispatch, tombstone, orphan-marker and
  generic-completion protections and their tests. No dependencies, resource
  limits, commits or pushes changed. D19a remains **unchecked and incomplete**.
- Preparation now writes private, no-clobber train and holdout JSON partitions
  under the existing per-scope deletion root. Their original row identities
  and bytes are bound to the corresponding frozen split digests. A new refit
  input loader verifies and loads only the training partition, without opening
  source or holdout; missing or altered partition data do not fall back to the
  full source. Existing CV loading deliberately retains its source-integrity
  check; it still materializes the full source and is **not** the new refit
  boundary. Existing experiments without partition files need explicit safe
  preparation before this refit loader can work; no migration is claimed.
- The gated reader consumes a durable, attempt-fenced, single-use authorization
  before opening `holdout.json`, verifies its frozen checksum and leaves the
  state `evaluating`, not `evaluated`. A read or verification failure consumes
  the opportunity. This is a program boundary, not protection against the local
  OS user or other code reading the original upload. The reader and loader are
  not connected to an actual finalization supervisor or result publisher.
- RED: `uv run --locked pytest tests/integration/test_finalization.py -q -k
  "stored_refit_input or stored_training_input"`: **3 failed**, 12 deselected,
  6.25 s: the loader reopened the source and no training partition existed.
  GREEN after implementation: **3 passed**, 12 deselected, 6.37 s. RED for
  `-k "real_holdout"`: **2 failed**, 15 deselected, 6.41 s because the actual
  gated reader was absent. The final owning suite below includes both cases.
- An intermediate change to replace the CV loader failed the existing
  `test_worker_rejects_source_replaced_after_queue` regression. The CV loader
  was restored and a separate refit-only loader introduced; the existing test
  was not weakened. Final `uv run --locked pytest tests/integration/test_finalization.py
  tests/integration/test_api.py tests/integration/test_retention.py -q`:
  **46 passed**, zero skips, two upstream deprecation warnings, 62.45 s.
- Final owning command `uv run --locked pytest tests/integration/test_finalization.py -q`:
  **19 passed**, zero skips, 14.73 s. Tests include both tasks' refit partition
  loads, source/holdout open spies, tampering, a real Windows-spawned child that
  opens and reads the fixture holdout following durable start, fenced-attempt
  rejection before file open, one-use authorization, and corruption after
  start remaining locked. The spawned child is a reader test harness, **not**
  the missing production finalization worker; the reservation uses seeded CV
  fixture evidence rather than a published real CV bundle.
- `uv run --locked pytest tests/integration/test_review_finalization_regressions.py
  tests/integration/test_holdout_gate.py -q -x --tb=short`: **27 passed**, zero
  skips, two upstream deprecation warnings, 262.93 s. An earlier run of these
  suites was inadvertently overlapped with the API/fixture suite, contrary to
  the serial-training execution contract: it displayed failures and timed out
  at 180 s before reporting details. The non-overlapping rerun passed; the
  cause of the interrupted failures was not established. No production data
  or benchmarks were used. `uv run --locked pytest tests/integration/test_worker_lifecycle.py
  tests/integration/test_worker_spawn.py -q`: **32 passed**, zero skips, 56.31 s.
  Prerequisite `uv run --locked pytest tests/integration/test_refit.py -q`:
  **12 passed**, zero skips, 15.34 s.
- `graphify update .` completed the AST-only update: **1,452 nodes, 4,884
  edges, 77 communities**; four unsupported files skipped and community labels
  warned as needing refresh. Documentation was not semantically re-extracted.
- **Remaining blockers:** reserve/start do not durably freeze and verify the
  complete published CV evidence, refit plan or MLP epoch budget; no production
  refit/finalization child is dispatched; no final prediction/metric bundle,
  fenced atomic publication, result retrieval or staged-bundle recovery exists.
  Cancellation/crash races across the real worker, recovery without recompute,
  repeated GET/POST and exactly-one actual worker read remain unverified.
  Existing initial configuration and CV workflows still open the original full
  source; separating them without weakening source-tamper checks remains open.
  Partition files are written after metadata registration, so an interrupted
  preparation can leave an experiment without files and must fail closed;
  concurrent preparation versus deletion has not been verified. No D19b,
  benchmark, download, real-dataset finalization or G3/G4 pass is claimed.

## Day 19a continuation â€” incomplete; orphaned-marker safety, 2026-10-31

- Preserved the existing uncommitted probe dispatch fix and regression, both
  tombstone protections, and the generic finalization-completion guard/tests.
  D19a is **not complete**; no finalization execution is enabled by this work.
- Day 18 prerequisite: `uv run --locked pytest tests/integration/test_refit.py -q`:
  **12 passed**, zero skips, 14.48 s. This verifies the existing module-level
  fixtures, not durable evidence binding or actual holdout isolation.
- RED: `uv run --locked pytest tests/integration/test_finalization.py::test_deleted_scope_status_preserves_tombstone_lock -q`:
  **1 failed, 1 passed**, zero skips, 5.59 s. A reserved tombstone was displayed
  as unselected. GREEN after the status correction: the same command **2 passed**,
  zero skips, 5.65 s.
- RED: `uv run --locked pytest tests/integration/test_finalization.py::test_orphaned_scope_marker_cannot_be_reselected -q`:
  **1 failed, 1 passed**, zero skips, 5.66 s. A reserved scope marker with no
  finalization record allowed a new selection despite unknown prior ownership.
  Reservation now fails closed and status exposes the lock. These are seeded
  repository fixtures, not a complete deletion or worker-recovery scenario.
- GREEN owning command: `uv run --locked pytest tests/integration/test_finalization.py -q`:
  **12 passed**, zero skips, 6.23 s. This file still contains dispatch, guard,
  and lock regressions only; it is not the specified D19a acceptance suite.
- `uv run --locked pytest tests/integration/test_finalization.py tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py tests/integration/test_retention.py tests/integration/test_refit.py -q`:
  **96 passed**, zero skips, two upstream deprecation warnings, 320.34 s on
  Windows/Python 3.12 using the existing locked environment. Finalization gate
  tests use a fixture evaluator and some mocked process liveness; existing real
  worker tests do not execute finalization or a real holdout reader.
- `graphify update .` completed the AST-only update: **1,420 nodes, 4,724 edges,
  70 communities**; four unsupported files skipped and community labels warned
  as needing refresh. Documentation was not semantically re-extracted.
- Reviewed the tracked code diff and untracked `tests/integration/test_finalization.py`;
  `git diff --check` passed with LF-to-CRLF notices. No commit or push was made.
- **Remaining implementation blockers:** no persisted training-only refit input
  or gated actual holdout reader; no durable frozen plan, complete CV evidence or
  epoch-budget binding; no finalization child executor; no dedicated fenced
  atomic final-result publication, staged-bundle recovery or repeated-result
  read. The generic completion guard remains in place. Actual-worker/access
  cancellation, crash-before-read, stale-attempt, atomic-publication and
  exactly-one-read acceptance cases were not implemented or verified. This
  continuation did not complete the requested D19a implementation; its checkbox
  remains unchecked. No report, benchmark, download, real dataset finalization,
  D19b work, or G3/G4 pass is claimed.

## Day 19a continuation â€” incomplete; generic completion guard, 2026-10-31

- Preserved the pre-existing uncommitted probe dispatch fix, tombstone changes,
  tests and validation entries. The current Windows/Python locked environment
  was used; no dependency or resource limits were changed.
- Day 18 prerequisite: `uv run --locked pytest tests/integration/test_refit.py -q`:
  **12 passed**, zero skips, 11.34 s. As recorded below, these are module-level
  refit fixtures, not evidence of a durable plan or actual holdout reader.
- RED: `uv run --locked pytest tests/integration/test_finalization.py::test_generic_completion_cannot_publish_a_finalization_without_a_bundle -q`:
  **1 failed**, zero skips, 5.40 s. The generic job terminal transaction allowed
  a finalization to become completed without a published final result. It now
  rejects finalization completion through that generic path, including if a CV
  publication argument is supplied; an already requested cancellation still
  wins and transitions pre-start to retryable. A dedicated fenced final-result
  publication transaction must replace this guard, not bypass it.
- Initial GREEN: `uv run --locked pytest tests/integration/test_finalization.py -q`:
  **6 passed**, zero skips, 5.86 s, before expanding the new guard test. Final
  owning and related regression run:
  `uv run --locked pytest tests/integration/test_finalization.py tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py tests/integration/test_retention.py tests/integration/test_refit.py -q`:
  **92 passed**, zero skips, two upstream deprecation warnings, 309.65 s.
  These tests do not exercise a finalization worker or actual holdout reader;
  fixture process-liveness mocking is not proof of child-death recovery.
- `graphify update .`: AST-only update, **1,417 nodes, 4,707 edges, 74
  communities**; four unsupported files skipped and community labels warned as
  needing refresh. Documentation was not semantically re-extracted.
- `git diff --check`: passed with LF-to-CRLF notices. Reviewed the tracked diff
  and the untracked finalization test. No commit or push was made.
- **D19a remains incomplete and unchecked.** Persisted training/holdout input
  separation, gated actual holdout access, durable complete CV evidence and
  epoch-budget binding, finalization child dispatch, fenced final-bundle
  publication/recovery and repeated published-result reads remain unimplemented.
  The requested actual-worker crash, cancellation, stale-attempt, staged-bundle,
  atomic-publication and exactly-one-read cases therefore remain unverified. No
  report, benchmark, download or real dataset finalization was run. D19b and
  G3/G4 were not started or passed.

## Day 19a continuation â€” incomplete; tombstone hardening only, 2026-10-31

- Working tree on entry contained an uncommitted probe-worker dispatch fix in
  `apps/api/workers/supervisor.py`, its untracked regression in
  `tests/integration/test_finalization.py`, and the preceding validation entry.
  All were preserved. The existing code graph was queried and refreshed after
  the changes below.
- Day 18 prerequisite: `uv run --locked pytest tests/integration/test_refit.py -q`:
  **12 passed**, zero skips, 12.20 s. This establishes only the existing
  module-level fixture guarantees, not a durable refit plan or actual reader.
- RED: `uv run --locked pytest tests/integration/test_finalization.py -q`:
  **3 failed, 2 passed**, zero skips, 5.48 s. Seeded reserved tombstones did not
  block the service, and neither reserved nor consumed tombstones blocked a
  direct repository reservation. The earlier consumed service check passed.
- The repository now checks both tombstone states under its reservation write
  transaction, serialized with deletion's tombstone write. A reserved deleted
  scope must remain locked because its original selection may no longer exist.
  The fixture seeds tombstones; it does not simulate complete metadata purge or
  prove an end-to-end finalization. GREEN:
  `uv run --locked pytest tests/integration/test_finalization.py -q`: **5 passed**,
  zero skips, 6.00 s. This remains a dispatch/tombstone test, not the D19a
  acceptance suite specified by the plan.
- `uv run --locked pytest tests/integration/test_finalization.py tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py tests/integration/test_retention.py tests/integration/test_refit.py -q`:
  **89 passed**, zero skips, 2 upstream deprecation warnings, 313.24 s on
  Windows with the existing locked environment. Existing worker tests exercise
  CV/probe workers, not a finalization worker or real gated holdout reader.
- `graphify update .`: AST-only update, **1,414 nodes, 4,699 edges, 67
  communities**; warned that community labels need refreshing and skipped four
  unsupported files. Documentation was not semantically re-extracted.
- `git diff --check`: passed with LF-to-CRLF warnings. Reviewed the tracked diff
  and the untracked finalization test; no commit or push was made.
- **D19a is not complete and its checkbox remains unchecked.** No persisted
  training-only input boundary, gated holdout reader, durable evidence/epoch
  binding, finalization executor, staged-bundle recovery, atomic final result,
  or repeated-result reader was implemented in this continuation. Consequently,
  the requested actual-worker/access crash, fencing, atomicity and one-read
  acceptance cases remain unverified. The fixture gate does not supply those
  guarantees. No report, benchmark, download or real dataset finalization was
  run; D19b and G3/G4 were not started or passed.

## Day 19a â€” incomplete; safety boundary and dispatch blocker, 2026-10-31

- Day 18 prerequisite recheck: `uv run --locked pytest tests/integration/test_refit.py -q`
  passed **12 tests**, zero skips, in 13.49 s. This confirms the existing
  module-level fixtures only, not durable finalization integration.
- A new fixture regression showed that a probe supervisor could claim a queued
  finalization job and mark it completed without finalization. RED:
  `uv run --locked pytest tests/integration/test_finalization.py -q` failed
  **1 test** (6.25 s). The two existing supervisor workloads are now restricted
  to claiming CV jobs; neither is a finalization executor.
- GREEN/regression: `uv run --locked pytest tests/integration/test_finalization.py tests/integration/test_review_finalization_regressions.py tests/integration/test_holdout_gate.py tests/integration/test_worker_lifecycle.py tests/integration/test_worker_spawn.py -q`:
  **60 passed**, zero skips, two upstream deprecation warnings, 326.32 s on
  Windows, using the existing locked environment. The new owning test covers
  dispatch fencing only; it is not the Day 19a acceptance suite.
- **D19a remains blocked/incomplete.** The stored-input workflow
  (`apps/api/services/workflow.py:load_experiment`) calls `load_dataset`, which
  validates/materializes the entire uploaded source, including holdout rows,
  before constructing a training-only view. No separate persisted training-only
  partition or gated holdout reader currently exists. Reusing that path for
  refit would not establish the requested actual holdout-access boundary.
  Reservation also does not durably bind checksum-verified complete CV evidence
  and the frozen MLP epoch budget; no finalization child dispatch, staged final
  bundle recovery/publication, or repeated-result reader exists. The fixture
  evaluator's `evaluated` state is not a durable published final result.
- Remaining work: implement and verify a training-only stored input and gated
  holdout access, durable frozen plan and attempt fencing, real finalization
  worker, cancellation/crash/recovery and atomic result publication, plus all
  requested fixture assertions including read count exactly one. Existing
  tombstone protections must be retained. No report, real dataset finalization,
  new dependencies, or G3/G4 pass is claimed. D19a remains unchecked.
- `git diff --check` passed with LF-to-CRLF warnings. `graphify update .`
  refreshed the ignored AST graph (1,412 nodes, 4,674 edges); it warned that
  community labels may need refreshing. Documentation was not semantically
  re-extracted.

## Day 18 refit â€” module-level fixture verification, 2026-10-31

Day 17 review was recorded as accepted; the subsequent remediation follow-up at
the next section passed 522 unit/integration tests. Day 18 was separately
authorized for refit, internal artifacts, tests and model cards only.

- Windows 11 build 22621, Python 3.12.15, locked environment; 6.70 GiB available
  at the initial check. No resource limits or dependencies were changed.
- RED: `uv run --locked pytest tests/integration/test_refit.py -q` initially
  failed all 7 tests because the refit module did not exist. An intermediate
  run had 6 pass and one test-fixture construction failure; corrected the
  fixture without changing the existing fold contract. Subsequent run after
  stronger row, artifact and even-fold coverage: **8 passed**, 10.71 s. After
  further provenance, unchanged real CV evidence and non-Dummy classical
  coverage, the final owning command passed **12 tests**, zero skips, 12.27 s.
- `uv run --locked pytest tests/integration/test_refit.py tests/integration/test_mlp_leakage.py tests/integration/test_mlp_cv.py tests/integration/test_cv_protocol.py tests/integration/test_holdout_gate.py tests/unit/test_mlp.py -q`:
  **90 passed**, zero skips, two upstream deprecation warnings, 393.74 s. This
  run began before the final refit-only provenance and tests were added; the
  final owning command above covers those changes.
- The fixture tests inspect both task types, preprocessing row IDs, every MLP
  dataset index on every epoch, fresh instances, exact training matrix/targets,
  no validation loader or restored best-training checkpoint, complete frozen
  fold identities, even-fold rounding, and rejection of changed evidence and
  changed training views. They also inspect checksummed manifest fields and
  reject forged handles, altered model files and altered metadata. The row
  fixture protects construction of its training view with a poison holdout;
  it is not an OS sandbox or a test of a real holdout reader.
- The new refit freezes a digest of complete candidate evidence and the epoch
  budget in an in-process plan, validates it again before fitting, and writes
  checksummed local artifacts without a deserializer. Training and holdout
  finalization are not connected to a production job in this scope.
- **Important handoff limitation:** the existing durable reservation does not
  persist the refit plan, CV-evidence digest or epoch budget at reservation.
  The in-process freeze is not proof of durable same-budget retries. Day 19 must
  bind the plan to the checked published CV evidence and durable reservation,
  enforce identical-budget retries, and validate the actual holdout boundary;
  do not infer those guarantees from these tests. Process-issued artifact
  verification handles also do not survive a restart; no reload is provided.
  Failed writes can leave an unpublished staging directory for later cleanup;
  crash reconciliation and final-result publication are not implemented here.
- `uv run --locked pytest tests/integration/test_tracking.py tests/integration/test_baseline_cv.py tests/unit/test_preprocessing.py tests/unit/test_resource_estimates.py -q`:
  **42 passed**, zero skips, one upstream deprecation warning, 137.10 s.
- `uv lock --check`: passed (107 packages). `git diff --check`: passed with
  an LF-to-CRLF warning. `graphify update .`: rebuilt the ignored code graph
  (1,408 nodes, 4,664 edges, 77 communities); it warned that community labels
  may need refreshing. The update is AST-only, not a semantic re-extraction of
  documentation. No full post-Day-18 suite, benchmark, named-dataset demo,
  real holdout read or finalization was run; G3 and G4 remain pending.

## Day 14â€“17 remediation follow-up â€” local suites passed, 2026-10-31

With 7.73 GiB system memory available at the start, reran the review checks without changing resource limits. This supersedes the incomplete-verification status in the next section; its earlier results remain part of the audit trail.

- `uv run --locked pytest tests/integration/test_review_finalization_regressions.py tests/integration/test_review_watchdog.py tests/integration/test_mlp_cv.py tests/integration/test_holdout_gate.py tests/integration/test_classical_suite.py tests/integration/test_worker_lifecycle.py tests/integration/test_resource_monitoring.py tests/integration/test_worker_spawn.py -q`: **108 passed**, 2 upstream deprecation warnings, 527.95 s. This includes successful real spawned MLP CV for both tasks, observation of inter-op threads = 1 in the worker at the MLP fold boundary, a genuinely blocked Windows-spawned child terminated by the watchdog, and cancellation and holdout fixture regressions.
- First full `uv run --locked pytest tests/unit tests/integration -q`: **521 passed, 1 failed**, 2 warnings, 1305.55 s. The failing completion-race test used an asynchronous `multiprocessing.Queue` with a synchronous process double already marked exited; the supervisor could observe the exit before the test's messages had been delivered and correctly returned an interrupted, non-published outcome instead of the test's expected timeout. Changed this test double to a synchronous `queue.Queue`, without changing production behavior.
- `uv run --locked pytest tests/integration/test_mlp_cv.py::test_mlp_fold_deadline_versus_completion_race tests/integration/test_review_watchdog.py -q`: **6 passed**, 18.74 s.
- Final `uv run --locked pytest tests/unit tests/integration -q`: **522 passed**, 2 upstream deprecation warnings, 1321.83 s. This is fixture and local Windows evidence, not the Day 21 resource benchmark or named-dataset demonstrations.
- The fixture capability boundary is tested, not a real holdout reader. Days 18â€“19 must still implement and verify authorization at the actual holdout access boundary, including crash and stale-owner behavior. No final refit, real holdout read, or holdout scoring was implemented or run. The separate classical per-fold hard-watchdog gap remains outside this remediation.

## Day 14â€“17 review remediation â€” verification incomplete, 2026-10-31

This section supersedes the earlier claims below that the Day 14 capability gate and Day 17 watchdog were fully verified. The five review findings were reproduced by code inspection; behavioral RED tests reproduced queued/concurrent retry, cross-experiment candidate selection, queued evaluation-start, and a forged fixture capability. No real holdout reader, final refit or holdout scoring was implemented or run.

- Retry now requires a terminal retryable finalization, matching frozen candidate evidence, and verified death of any registered child; an unclaimed queued cancellation is the only no-child case. Same-key retries return the same current job; competing keys cannot create parallel retries. Finalization terminal/cancel/recovery transitions update the pre/post-evaluation state. Reservation checks candidate/experiment/scope together within the scope-lock transaction.
- Evaluation-start now requires the active fenced owner, job, attempt, registered live child and unchanged selection. Schema v6 stores a hash of a fresh token; the fixture evaluator atomically checks scope, digest, attempt, owner, liveness and token and consumes it once. This is a fixture boundary, not a completed reader: Days 18â€“19 must integrate equivalent atomic authorization immediately before any real holdout access, verify stale processes and crash recovery, and test the actual reader. Python object privacy is not an OS security boundary against the local owner.
- The supervisor and spawned CV child now use an acknowledged, challenge-bound MLP fold-control channel separate from best-effort UI progress. The child cannot begin fold preprocessing until the supervisor arms its timer; a matching finish arrives only after scoring, and ambiguous, missing, out-of-order or late control evidence prevents publication. Timeout uses owned-child termination. The existing classical per-fold hard watchdog remains outside this remediation.
- The spawned CV child sets torch inter-op threads to one before CV numerical work. The actual-worker integration assertion records the observed value at the MLP boundary, but successful completion with that assertion could not be verified under the observed system-memory pressure.
- RED: `uv run --locked pytest tests/integration/test_review_finalization_regressions.py -q` initially produced 3 failures (queued retry, competing retries, wrong-experiment candidate); the two additional gate regressions both failed before the gate change. GREEN after the fixes and subsequent additions: `uv run --locked pytest tests/integration/test_review_finalization_regressions.py tests/integration/test_review_watchdog.py tests/integration/test_job_repository.py tests/integration/test_retention.py tests/integration/test_worker_lifecycle.py::test_version_one_metadata_migrates_without_losing_queued_jobs -q` â€” **66 passed**, one upstream warning, 19.26 s.
- `uv run --locked pytest tests/unit -q` â€” **290 passed**, 16.51 s. A combined focused run of review, watchdog, MLP CV, holdout gate and classical suite reported **34 passed, 16 failed**, 287.33 s: 15 failed jobs recorded `resource_pressure`; one cancellation race recorded `timeout`. The cancellation-on-channel-close logic was corrected and a direct regression passes, but a later actual-worker cancellation rerun could not reach fitting under memory pressure. A subsequent targeted run reported **18 passed, 2 failed** in 43.06 s, including the real Windows-spawned blocked-MLP watchdog test passing with the challenge protocol; both failures were successful-worker tests with `safe_error=resource_pressure`.
- Available RAM was approximately 3.38 GiB at inspection, later as low as 2.11 GiB, below the unchanged 4 GiB system reserve. The full integration suite was not run under these conditions. Do not treat the failed worker completion, cancellation, holdout API or real-worker inter-op assertions as passing: rerun the focused suites and full integration suite with sufficient available memory before accepting the review. G3/G4 remain pending. No resource guard was lowered or disabled.

## Day 17 independent review blocker resolved â€” ACCEPTED, 2026-10-31

The independent review blocker on D17a and D17b has been resolved. D17 is now **ACCEPTED**.

- **Supervisor-Enforced MLP Per-Fold Watchdog:**
  - Implemented hard watchdog in `WorkerSupervisor._monitor` (`apps/api/workers/supervisor.py`). The supervisor tracks active fold start and finish events via `on_progress` (propagated with `fold_finished` boolean through `ml/training/classical_suite.py` and `apps/api/workers/execute.py`).
  - Active fold deadline check: in every monitoring loop iteration, the supervisor evaluates `now - mlp_fold_started > self.limits.mlp_fold_seconds`. If exceeded, the supervisor marks `mlp_fold_deadline_exceeded = True`, clears any received result (`self.last_result = None`), and immediately returns `_Outcome("failed", "failed", "timeout")`.
  - Termination of owned child: upon returning timeout, `_shutdown` invokes `monitor.terminate_owned_child` verifying `creation_time` identity before sending terminate, and escalates to `kill()` if unresponsive, guaranteeing verified process death before releasing the singleton slot.
  - Fail-closed publication: an overdue fold or job never publishes an eligible candidate or partial results (`self.last_result = None`, `stage_attempt` skipped, DB record updated to `state="failed"`, `safe_error="timeout"`, and `candidates` status updated to `"failed"`).
  - Robust handling of delayed, lost, or ambiguous progress messages: on receiving a completion result payload, `WorkerSupervisor` inspects candidate folds; any fold with `fit_seconds >= self.limits.mlp_fold_seconds` or unclosed fold exceeding deadline causes immediate rejection and timeout failure.

- **RED/GREEN Verification Evidence:**
  - **RED**: `test_supervisor_hard_watchdog_terminates_stuck_mlp_child` with test block `_MODELARENA_TEST_MLP_BLOCK_SECONDS=6.0` and `mlp_fold_seconds=1`: failed with `took 13.83s > 4.0s` without the supervisor watchdog (the supervisor did not interrupt the child during the stuck native fold).
  - **GREEN**: With the supervisor watchdog enabled, `test_supervisor_hard_watchdog_terminates_stuck_mlp_child` passed with `mlp_duration = 0.82s` (interrupted promptly between 0.7s and 3.5s, child process terminated and dead `not psutil.pid_exists(child_pid)`, honest terminal state `final.state == "failed"`, `final.safe_error == "timeout"`, 0 published candidates, and worker slot cleanly recovered).
  - `test_mlp_fold_deadline_versus_completion_race`: verified that an overdue result payload returning from a child is rejected fail-closed as timeout with 0 candidates published.
  - `test_mlp_fold_deadline_versus_cancellation_race`: verified that user cancellation requested during an active fold stops the child and cleanly transitions to `cancelled` with slot recovered.

- **Regression Suite Verification:**
  - `uv run pytest tests/integration/test_mlp_cv.py -v`: **22 passed in 182.21 s** (includes all 19 previous tests + 3 new watchdog/race tests).
  - `uv run pytest tests/integration/test_mlp_leakage.py -q`: **10 passed in 24.79 s**.
  - `uv run pytest tests/integration/test_worker_lifecycle.py tests/integration/test_resource_monitoring.py tests/integration/test_worker_spawn.py -q`: **47 passed in 100.79 s**.
  - `uv run pytest tests/integration/test_classical_suite.py tests/integration/test_tracking.py -q`: **14 passed in 308.43 s**.
  - `git diff --check`: clean (0 errors).
  - `uv lock --check`: clean (107 packages resolved).
  - `graphify update .`: AST graph refreshed cleanly (rebuilt 1318 nodes, 4284 edges, 72 communities).

- **Status and Handoff:**
  - Status: D17a and D17b **ACCEPTED**.
  - All review blockers resolved. Day 18 (final full-training refit and trusted artifacts) is authorized to begin. Day 18 has **not** been started in this turn.

## Day 17 independent review â€” BLOCKED, 2026-10-31

This review supersedes the Day 17 review-pending status below. D17a/D17b are **not accepted**: the required hard per-fold MLP time limit is not enforced by the supervising process. `WorkerSupervisor._monitor` enforces the overall `cv_job_seconds`, RSS and memory-pressure limits, but has no `mlp_fold_seconds` watchdog. A stuck native preprocessing/training call can therefore remain active beyond the frozen MLP fold deadline until the job limit or another supervisor limit is reached. The added cooperative deadline checks between operations/batches, and rejects an overdue returned fit; they cannot interrupt a blocked native call. A hard watchdog with verified owned-child termination and a regression using a genuinely blocked child is still needed before acceptance. This is also a pre-existing gap for classical per-fold deadlines; this review does not claim to fix those.

Review fixes and evidence:

- A real spawned worker previously returned only four classical families; it now runs the five-family registry for both tasks and publishes the MLP through the existing private bundle/MLflow path. Classical-only suite calls remain available. Progress reports include the MLP and all five families. The earlier test created a manual payload and did not establish worker integration.
- `ResourceLimits(max_epochs=2)` previously trained 100 epochs and reported 100. The actual frozen limit now bounds training and is reported in both manifest and MLflow. A fold's complete curve must have one finite training and validation point per consecutive epoch; incomplete MLP metadata is ineligible. An outer-scoring failure retains a completed inner-fit curve but remains failed and unrankable. MLP fit time now includes preprocessing; BLAS is pinned to one during the torch training phase.
- Strengthened tests check exact inner-fit/inner-validation loader arrays and frozen fit/transform row identities, holdout exclusion, positive label `a` mapped to 1, fold metrics recomputed from actual OOF outputs, both-task spawned jobs, exact per-fold results/manifest/MLflow curve values, reduced epoch caps, incomplete/undefined primary metrics and fail-closed publication when MLflow curve logging fails. The poison reader in the shared fixture guards materialization of the training view; it is not an OS-level proof against all possible holdout access. No holdout evaluation was run.
- RED checks: the two spawned-worker tests could not find an MLP candidate and a lowered cap produced 100 epochs; a scoring-failure test lost the curve; an artificially overdue fit remained eligible. Each subsequently passed after the respective fixes. These are fixture checks, not default-workload or benchmark evidence.
- Windows 11 build 22621, Python 3.12.15, existing locked environment: `uv run --locked pytest tests/integration/test_mlp_cv.py tests/integration/test_mlp_leakage.py tests/integration/test_tracking.py tests/integration/test_classical_suite.py tests/integration/test_api.py tests/integration/test_cv_protocol.py tests/integration/test_baseline_cv.py tests/unit/test_mlp.py -q` â€” **111 passed**, zero skips, 2 upstream deprecation warnings, 529.92 s. A prior combined run had one failure from the changed worker's final progress family; the expectation was updated to `mlp` and the successful rerun includes it.
- `uv run --locked pytest -q` â€” **497 passed**, zero skips, 2 upstream deprecation warnings, 1251.75 s. After that run the artifact test was strengthened to compare exact per-fold values across all three stores; `uv run --locked pytest tests/integration/test_mlp_cv.py -q -k persistence_to_mlflow_and_manifest` â€” **1 passed, 18 deselected**, one upstream warning, 43.53 s. No application code changed after the full run.
- `uv lock --check` passed. `graphify update .` refreshed the ignored AST graph; its community labels may need refresh. No dependencies, benchmark, external dataset, final refit, holdout evaluation, commit or push were introduced.

**Handoff:** preserve the blocked status and do not start Day 18 until the hard MLP fold-time boundary is implemented and independently verified. G3/G4 and release remain pending.

## Day 17 implementation (review-pending) â€” 2026-10-31

The user authorized Day 17 implementation (`D17a` and `D17b`) and specified fixture tests only under the shared execution contract in `docs/IMPLEMENTATION_PLAN.md`. Day 17 implementation is complete and verified locally with zero regressions across the entire test suite. Per the contract, status is left review-pending for independent Terra inspection and review before Day 18.

- **D17a Unified Model Registry and Both-Task MLP Outer Validation Scoring:**
  - Added unified `model_families(task: TaskType) -> tuple[str, ...]` in `ml/models/classical.py` and exported from `ml/models/__init__.py`. Contains `("dummy", "logistic", "random_forest", "hist_gradient_boosting", "mlp")` for binary classification and `("dummy", "ridge", "random_forest", "hist_gradient_boosting", "mlp")` for regression. Retained `classical_families` for backwards compatibility with classical suites and supervisor workloads.
  - Added strict validation in `evaluate_candidate` in `ml/training/cv.py`: verifies `config.family in model_families(task)` and requires `config.model_seed` to be an unsigned 32-bit integer.
  - Implemented 5-fold CV for both tasks on frozen partitions: outer validation features are transformed using `FoldPreprocessor.transform(..., dtype=np.float32)` with guarded allocations and range checks; fed as CPU float32 tensors to the trained MLP.
  - Correct sigmoid probabilities: binary classification predictions use `torch.sigmoid(...)` strictly in `[0.0, 1.0]` for `score_predictions` and `DiagnosticPoint` probabilities, with prediction threshold at `probability >= 0.5`. Regression uses raw model outputs and computes residuals `truth - prediction`.
  - Zero holdout access: verified via row ID spies and `PoisonHoldout` that holdout row IDs are never accessed during preprocessing fit, inner training, or outer validation scoring. Outer validation rows are strictly excluded from preprocessing fit.
  - Classical eligibility and failure policy matching: fold-level execution errors or non-finite losses mark the fold as `failed` with `metrics=None` and `error="fold_execution_failed"`. An incomplete fold or undefined primary metric sets `candidate.eligible = False` with `reason="incomplete_folds"` or `"undefined_in_one_or_more_folds"`, correctly routing the candidate to `unranked` in `compare_candidates`. Valid complete candidates rank cleanly with baseline deltas vs dummy.

- **D17b Persistence of Loss Curves, Best Epochs, Timing, Caps, and Reduced Inner-Fit Counts:**
  - Updated `FoldResult` in `ml/contracts.py` with validated fields: `best_epoch`, `epochs_trained`, `stop_reason`, `inner_fit_row_count`, and `loss_curve` (tuple of `EpochLoss`). Added strict bounds validation in `FoldResult.__post_init__` enforcing `1 <= best_epoch <= epochs_trained <= 100`, `inner_fit_row_count >= 2`, and `len(loss_curve) <= 100`.
  - Updated `ml/training/cv.py` to extract `best_epoch`, `epochs_trained`, `stop_reason`, `inner_fit_row_count`, and `loss_curve` from `MLPFoldResult` and attach them to each completed `FoldResult`.
  - Updated `stage_attempt` in `ml/tracking/publication.py`: handles `candidate["family"] == "mlp"` by recording MLP architecture and cap parameters (`max_epochs: 100`, `patience: 10`, `lr: 1e-3`, `batch_size: 32`, `hidden_dims: [64, 32]`) in `manifest["candidates"]`. Preserves fold summaries with `best_epoch`, `epochs_trained`, `stop_reason`, `inner_fit_row_count`, and finite loss curves. Converts all curve floats cleanly to ensure JSON compliance (`allow_nan=False`).
  - Updated `track_attempt` in `ml/tracking/mlflow_store.py`: logs candidate-level cap parameters (`max_epochs`, `patience`, `batch_size`, `lr`, `hidden_dims`), fold-level metrics (`fit_seconds`, `score_seconds`, `best_epoch`, `epochs_trained`, `inner_fit_row_count`), fold param `stop_reason`, and per-epoch `train_loss` and `val_loss` at `step=epoch` using `client.log_metric`. Preserves fold boundaries in distinct MLflow fold runs without any cross-fold concoctions.

- **Verification Evidence:**
  - Day 17 acceptance tests: `uv run pytest tests/integration/test_mlp_cv.py -q`: **11 passed in 70.40 s**.
  - Leakage regression tests: `uv run pytest tests/integration/test_mlp_leakage.py -q`: **10 passed in 13.03 s**.
  - Tracking regression tests: `uv run pytest tests/integration/test_tracking.py -q`: **7 passed in 136.11 s**.
  - Full unit and integration regression suite `uv run pytest -q`: **489 passed, 0 failures, 0 skips in 759.70 s (0:12:39)**.
  - `uv lock --check` and `git diff --check` succeeded.

- **Status and Handoff:**
  - Status: **Review-pending** for independent Terra inspection under the Days 17â€“28 contract.
  - Day 18 (final refit, frozen median epochs, trusted artifacts) has **not** been started.

## Day 16 core acceptance â€” 2026-10-31

The user requested execution of Day 16 tasks from `IMPLEMENTATION_PLAN.md`, English output, and direct commit and push once all tests pass. Both Day 16 deliverables are complete and verified locally with zero regressions across the entire test suite.

- **D16a Persisted Inner Indices and Leakage-Free Preprocessing:**
  - Integrated persisted inner indices (`manifest.inner.fit_row_ids` and `manifest.inner.validation_row_ids`) into `ml/training/mlp_loop.py` and `ml/training/cv.py`.
  - Implemented `train_mlp_fold` in `ml/training/mlp_loop.py` (and re-exported in `ml/training/cv.py`), executing strict fold boundary isolation.
  - Verified with row ID spies that `FoldPreprocessor.fit` sees strictly `manifest.inner.fit_row_ids` only. Disjointness guarantees enforce zero inner-validation, zero outer-validation, and zero holdout rows in preprocessing fit or training gradient updates (`isdisjoint` assertions).
  - Verified that extreme numerical outliers (e.g. 1e9) and unseen categorical sentinels placed exclusively in inner validation do not alter scaler means, standard deviations, or categorical vocabularies (scaler mean matches true inner-fit mean exactly within `1e-5`).
  - Proved that outer-validation data cannot influence training epochs, loss progression, or early stopping decisions: mutating outer-validation features and inverting targets produces bit-for-bit identical `best_epoch`, `epochs_trained`, `stop_reason`, and model weights.
  - Protected against holdout access during fold training and CV evaluation via `PoisonHoldout` readers.

- **D16b Best Checkpoint Restoration, Reporting Metadata, and Deterministic Replay:**
  - Implemented best checkpoint restoration (`model.load_state_dict(best_state_dict)`) in `train_mlp` and `train_mlp_fold`. Verified with scripted loss sequences that the restored model weights strictly match weights from `best_epoch`, not the final epoch (`epochs_trained`).
  - Added comprehensive fold outcome reporting in `MLPFoldResult`: returns `fold_index`, `best_epoch`, `best_loss`, `stop_reason`, `epochs_trained`, `history` (`EpochLoss` tuples), `inner_fit_row_ids`, `inner_val_row_ids`, `outer_val_row_ids`, `derived_seed`, and `fit_seconds`.
  - Enforced `min_delta=1e-6` improvement threshold in `train_mlp`: verified that numerical fluctuations below `1e-6` (e.g. 1e-7) do not reset the early stopping patience counter.
  - Guaranteed deterministic seed replay: `derive_fold_seed(model_seed, family, fold_index)` derives a stable 32-bit unsigned seed; reruns with the same seed yield identical losses, best epoch, and model weights within fixture tolerance (`abs=1e-6`).
  - Integrated `family="mlp"` into `evaluate_candidate` in `ml/training/cv.py`, enabling CV execution across frozen folds for both binary classification and regression.

- **Test-first and regression evidence:**
  - Verified RED failure on `tests/integration/test_mlp_leakage.py` before implementation, followed by minimal implementation achieving GREEN.
  - `uv run pytest tests/integration/test_mlp_leakage.py -q`: **10 passed in 13.03 s**.
  - Full unit and integration regression suite `uv run pytest tests/unit tests/integration -q`: **478 passed, 0 failures, 0 skips in 365.71 s (0:06:05)**.
  - `uv lock --check` and `git diff --check` succeeded.

- **Limitations Recorded:**
  - Per-fold train/validation loss curve persistence and MLflow tracking integration remain scheduled for Day 17.
  - Full-training refit using median best epoch across CV folds remains scheduled for Day 18.

## Day 15 core acceptance â€” 2026-10-31

The user requested execution of Day 15 tasks from `IMPLEMENTATION_PLAN.md`, English output, and direct commit and push once all tests pass. Both Day 15 deliverables are complete and verified locally with zero regressions across the entire test suite.

- **D15a PyTorch CPU MLP Architecture, Loss Selection, and DataLoader:**
  - Implemented `make_mlp` in `ml/models/mlp.py` with configurable bounded hidden layer dimensions (defaults: 64/32, ReLU activations, scalar linear output), input bounds validation (`in_features > 0`, hidden dimension width `1..2048`, hidden depth `1..8`), and deterministic seeded CPU initialization (`0 <= seed < 2**32`).
  - Added task loss selection `get_task_loss` in `ml/models/mlp.py`: `nn.BCEWithLogitsLoss` for `TaskType.BINARY_CLASSIFICATION` and `nn.MSELoss` for `TaskType.REGRESSION`.
  - Added `ArrayDataset` converting tabular features and targets to `float32` CPU tensors with length and shape validation.
  - Added deterministic seeded `create_dataloader` in `ml/models/mlp.py` strictly enforcing `num_workers=0`, `pin_memory=False` (CPU-only, no CUDA paths), and seeded CPU `torch.Generator` for reproducible batch shuffling.
  - Added `create_optimizer` with standard default `torch.optim.Adam` at `lr=1e-3`.

- **D15b Training Loop, Loss Tracking, Detached Best State, and RunControl:**
  - Implemented `train_mlp` in `ml/training/mlp_loop.py` managing explicit `model.train()` training phases and `model.eval()` validation phases.
  - Enforced no gradient leakage into validation: validation runs under `torch.no_grad()`, ensuring optimizer parameters receive zero gradient updates from validation data.
  - Implemented exact sample-weighted epoch loss tracking (`total_loss * batch_size / total_samples`) preserving correct loss aggregation over variable batch sizes.
  - Implemented detached best-state weight copies (`best_state_dict`), ensuring future model mutations or subsequent epochs never corrupt the saved best checkpoint.
  - Added finite-loss checks raising `NonFiniteLossError` immediately when NaN or Inf loss occurs during training or validation, preventing corrupted models or masqueraded metrics.
  - Implemented `RunControl` supporting cancellation checks and callbacks (`on_batch_end`, `on_epoch_end`). Cancellation at either batch or epoch boundaries immediately halts training and raises `TrainingCancelled` (subclass of `SuiteCancelled`).
  - Implemented fold-local early stopping logic: validation loss improvement required strictly lower by `> min_delta (1e-6)`; initial finite epoch is tracked as best; restores best checkpoint weights after patience exhaustion.
  - Added intra-op CPU compute threads scoping in `train_mlp` with safe restoration.

- **Test-first and regression evidence:**
  - Verified RED failure on `tests/unit/test_mlp.py` before implementation, followed by minimal implementation achieving GREEN.
  - `uv run pytest tests/unit/test_mlp.py -q`: **22 passed in 8.81 s**.
  - Full unit and integration regression suite `uv run pytest tests/unit tests/integration -q`: **468 passed, 0 failures, 0 skips in 298.20 s (0:04:58)**.
  - `uv lock --check` succeeded.

- **Limitations Recorded:**
  - Strict inner-fold early stopping integration and leakage tests with persisted inner indices remain scheduled for Day 16.
  - Both-task MLP CV integration, sigmoid probability scoring, and MLflow loss curve artifacts remain scheduled for Day 17.

## Day 14 core acceptance and Milestone G2 â€” 2026-10-31

The user requested execution of Day 14 tasks from `IMPLEMENTATION_PLAN.md`, English output, and direct commit and push once all tests pass. Both Day 14 deliverables are complete and verified locally; Milestone G2 (Classical reliable experiments) is passed.

- **D14a Holdout Gate and Immutable Reservation:**
  - Implemented immutable `FrozenSelection` reservation and unique evaluation scope locking in `apps/api/database/repository.py` and `apps/api/services/finalization.py`.
  - Scope locking enforces `scope_id = sha256(canonical_sha256 + target_column + task)`. A unique constraint on `(canonical_sha256, target_column, task)` ensures any competing experiment on equivalent canonical data with identical target and task cannot trigger duplicate holdout evaluations. Concurrent reservation requests serialize to a single winner; differing candidate contenders are denied with `ConflictError`.
  - Candidate digest verification binds `sha256(experiment_id + candidate_id + config_sha256 + split_manifest_digest + metric)`. Altered configs, modified split manifests, or differing candidate metrics are rejected. Unrankable or incomplete candidates (failed status or null primary metric) are rejected before reservation.
  - Durable pre-evaluation retry requires verified worker child termination and unconsumed evaluation-start state. If a failure occurs before evaluation-start, retry is permitted only for the exact same selection.
  - Fail-closed evaluation-start boundary: `start_evaluation` atomically commits `evaluation_started_at`, updates state to `evaluating`, updates `scope_lock_markers` to `consumed`, and returns a single-use authentication token. After evaluation has started, any subsequent retry attempt is permanently banned (`EvaluationAlreadyStartedError`).
  - `HoldoutCapability` token guard: capability issuance requires the private token returned by `start_evaluation`. Empty or invalid tokens raise `ValueError`.
  - Day 13 scope-lock marker and keyed tombstone integration: reservation checks existing `scope_lock_markers` and `scope_tombstones`. Scopes previously marked `consumed` or present in tombstones reject reservation immediately (`HoldoutAlreadyConsumedError`).
  - Schema v5 migration: additive `finalizations` table with `finalizations_immutable_selection` trigger, `jobs.kind` CHECK constraint allowing `('cv', 'finalization')`, and separate worker claim filter ensuring classical supervisors claim only `kind="cv"`.
  - Verified with 10 integration tests in `tests/integration/test_holdout_gate.py`: **10 passed in 100.80 s**.

- **D14b Milestone G2 Acceptance:**
  - Full suite rerun: `uv run pytest tests/unit tests/integration -q` passed **446 tests in 279.68 s (0:04:39)**, zero skips, with two expected upstream deprecation warnings (Starlette `TestClient` and MLflow `SQLAlchemyStore`).
  - All G2 exit criteria are met:
    - Real Windows worker lifecycle, process monitoring, heartbeat, cancellation during native fitting, orphan reconciliation and restart recovery (Days 8â€“9).
    - All classical candidate estimators (Dummy, LogisticRegression/Ridge, RandomForest, HistGradientBoosting) verified across both classification and regression tasks on frozen CV folds (Days 10â€“11).
    - Local SQLite MLflow tracking with checksum-verified manifests, metrics, fold diagnostics, and atomic candidate publication (Day 12).
    - CV ranking by primary metric, tie-breaking, baseline comparison, and holdout-isolated API & retention boundaries (Day 13).
    - Durable finalization reservation gate, candidate digest verification, evaluation scope locking, dead-child verified retries, and post-start lock protection (Day 14).
  - **Limitations Recorded:**
    - Model refit on full training set and real holdout scoring remain scheduled for Day 18 and Day 19; D14a verification uses a fixture evaluator.
    - PyTorch CPU MLP architecture, fold-local early stopping, and loss curve artifacts remain scheduled for Days 15â€“17.
    - Complete cross-store deletion (permanent MLflow run purging and UI integration) remains scheduled for Day 25.

## Day 13 core acceptance â€” 2026-10-30

The user requested completion of Day 13. The daily backend API and deletion-core work is verified locally; **this is not verification of finalization, confirmed deletion or a product release**. Day 14 owns the real finalization gate, and Day 25 owns permanent MLflow/metadata cleanup and deletion UI.

- API endpoints now have explicit typed response models. Errors use server-generated request IDs, safe messages, allowlisted field details and a consistent JSON envelope, including bad hosts, unknown routes and sanitized internal errors. Duplicate/malformed Host headers and path traversal are rejected, allowed-origin clients can read error request IDs, and oversized/truncated streamed uploads leave no staging file. The API suite also covers idempotency conflicts and the upload-to-spawned-worker flow. The separate leaderboard suite denies SQL access to a holdout table, including through the HTTP endpoint.
- Windows console acceptance uncovered an inherited Ctrl+C-ignore flag. The dedicated API and worker entrypoints now restore Ctrl+C handling. Real Windows console events stopped the API and released its listener, stopped an idle worker and removed its presence, and interrupted a running spawned job while releasing the lease. The tests sent console events from a helper process; manual interactive PowerShell use was not separately observed.
- Retention tests cover queued and running references, a real Windows exclusive file handle and retry, known raw/train/holdout/split/staging/report file cleanup, and a minimal keyed tombstone. A synthetic consumed marker and **test-only simulated metadata cleanup** allow an equivalent CSV reupload to show `holdout_lock=consumed`. This tests the Day 13 retention primitives, not a real finalization gate or end-to-end confirmed deletion. Production deletion still returns `deletion_pending` and may retain metadata and MLflow copies. Day 14 must create and enforce markers atomically; Day 25 must permanently purge and verify all stores before reporting deletion complete.
- Test-first runs exposed missing error envelopes, duplicate Host acceptance, absent reupload lock reporting and ignored Windows Ctrl+C before the corresponding fixes. `uv run --locked pytest tests/integration/test_api.py -q`: **14 passed in 15.99 s**. `uv run --locked pytest tests/integration/test_retention.py -q`: **13 passed in 5.87 s**. Full `uv run --locked pytest -q`: **436 passed in 205.06 s**, zero skips; two upstream Starlette/MLflow-SQLAlchemy deprecation warnings. `uv lock --check`, `git diff --check` and the ignored AST-only `graphify update .` completed.

The Day 13 daily core is complete within its planned boundaries; G1 remains passed locally and G2â€“G4 remain pending. The simulated reupload must not be presented as proof that a real holdout cannot be reevaluated, nor as proof of complete deletion.

## Day 13 runtime and retention follow-up â€” 2026-10-30

The user requested continuing the remaining work. The checkout began clean on `main` after `e86a844`. The work remains scoped to the backend and retention boundaries; no external data, dependencies or holdout evaluations were introduced.

- Added `python -m apps.api.server` with loopback-only Uvicorn, an ephemeral `secrets` session token displayed in the terminal, exact optional UI origins and disabled access logging. `python -m apps.api.workers.runner` runs the classical supervisor in a separate process; `--once` exits nonzero if a claimed job fails. Both use the same local settings. `/health` reports recent PID/creation-time-verified worker presence as `available`, stale/live-unobservable as `unknown`, and absence as `not_detected`. Presence is informational, never grants or steals the durable fitting lease and does not prove resource admission. A real API subprocess and a real worker subprocess processing an uploaded regression fixture were exercised on Windows; the API subprocess was forcibly terminated by the test, so interactive Ctrl+C behavior is not newly claimed.
- Classical child progress now reports allowlisted family, zero-based fold index and bounded finished/total units. These are persisted with worker-ownership fencing and monotonic counts and survive API refresh. A finished unit means the fold returned, not necessarily that its metrics succeeded; model eligibility still uses complete CV results. Added idempotent CV retry HTTP access and blocked profile, leaderboard and candidate reads while related data deletion is pending.
- Schema v4 adds worker presence, job progress, lock-marker and keyed-tombstone tables, with additive v1/v2/v3 migrations. Deletion records an HMAC-SHA256 tombstone (version and reserved/consumed state only) before file removal for each existing scope marker; consumed cannot be downgraded by retry. A private 32-byte key is kept in application storage; losing it while tombstones exist fails lookup closed. Equivalent-format dataset content matched the same synthetic consumed scope in tests. Unreserved scopes create neither a tombstone nor a key. **These markers were seeded by tests: no production finalization gate currently creates them or checks the tombstones.** Day 14 must atomically integrate marker creation, evaluation-start and tombstone checks; no end-to-end consumed-holdout guarantee is claimed. File ACL behavior beyond OS defaults is unverified.
- Added a disclosure endpoint and partial cleanup of known upload, train/holdout partition, split, final, attempt and report paths, with symlink checks and repeatable pending state. Shared-scope files may be affected and are disclosed. Metadata, MLflow copies, logs and unknown future artifact layouts are not permanently purged; every deletion remains `deletion_pending`. This is not forensic erasure or confirmation that data is gone. Equivalent uploads remain blocked while the durable intent exists.
- Before the last retry-route addition, `uv run --locked pytest -q` passed **426 tests in 192.58 seconds**, zero skips, with upstream Starlette and MLflow/SQLAlchemy deprecation warnings. Test-first checks reproduced missing runtime, deletion read fencing, keyed tombstone, file cleanup, fold progress and CLI failure-status behavior. The final targeted checks are recorded below. `uv lock --check`, `git diff --check` and ignored `graphify update .` completed.
- Final targeted verification: `uv run --locked pytest tests/integration/test_api.py tests/integration/test_retention.py tests/integration/test_local_runtime.py tests/integration/test_job_repository.py tests/integration/test_worker_lifecycle.py -q` passed **77 tests in 72.02 seconds**, zero skips, with the same two upstream warnings.

Day 13 remains **partial**: finalization-gate/tombstone integration and confirmed deletion across all stores are not implemented; the complete API acceptance matrix and release milestones remain open. No final holdout score, G2â€“G4 or release is claimed.

## Day 13 continuation â€” 2026-10-30

The user requested continuation of Day 13. The checkout began clean on `main` after commit `8f64ed0`. No dependency changes, dataset downloads, named demos or holdout evaluation were performed.

- Multipart CSV uploads are streamed to an exclusive private staging file, validated with the existing bounded parser and published under a server-generated dataset ID. Client filenames are discarded. Byte limits, invalid input, duplicate-content reuse and reuploads during a deletion intent are tested. Configurations select explicit feature types and produce a training-only profile; experiments freeze the metric and seed; queue requests use persisted idempotency keys. Configuration creation currently registers a default experiment as an internal side effect; it does not run training.
- A classical worker can now reconstruct a job from persisted configuration, checksummed split and private source after claiming it, revalidate the source fingerprints, and pass a training-only context to the spawned child. The HTTP fixture exercises upload through completed spawned-worker leaderboard/candidate inspection after reopening SQLite. The supervisor is explicitly constructed in test code; a production server/worker startup command, token generation/distribution and worker-availability reporting are not delivered. The current job stage does not expose per-fold units. The source is parsed/validated in the supervisor before the child starts; fitting remains in the child. Changed source content fails the job before spawning.
- SQLite schema v3 adds durable deletion intents, with v1/v2 migration. An intent is recorded only after transactionally excluding queued/running references for equivalent canonical data; enqueue, new experiment registration and equivalent reupload are fenced while pending. Cleanup removes the selected private upload and known attempt directories, retries safely, and reports locked-file failures. **Every deletion remains `deletion_pending`**: metadata and MLflow copies can remain, and keyed minimal scope-lock tombstones and confirmed cross-store deletion are not implemented. The intent retains ordinary dataset identity, not a minimal HMAC tombstone. Never claim deletion or protection after erasing the application store. Finalization reservation does not exist yet; its tombstone integration remains necessary. A client must not interpret partial cleanup as forensic erasure.
- Test-first: upload, workflow/worker and retention tests failed before their interfaces were added. The named API, retention, repository, worker-lifecycle and split suites passed **118 tests in 51.97 seconds**, zero skips. The full suite passed **412 tests in 158.30 seconds**, zero skips, with upstream Starlette and MLflow/SQLAlchemy deprecation warnings. A transaction-race test and a whitespace validation guard were added after that full run; see final targeted verification below. `uv lock --check` passed; `graphify update .` refreshed the ignored AST graph.
- Final targeted run: `uv run --locked pytest tests/integration/test_api.py tests/integration/test_retention.py tests/integration/test_leaderboard.py -q` passed **19 tests in 26.57 seconds**, zero skips, with the same two upstream warnings. The new race assertion checks that an enqueue and deletion intent cannot both win.

Day 13 is **not complete**: the full API acceptance matrix, reliable production startup and complete deletion/tombstone acceptance remain open. G1 remains passed locally; G2â€“G4 and release remain pending.

## Day 13 partial â€” 2026-10-30

The user requested Day 13, English output, commit and push. The checkout started clean on `main`. No dependencies, downloads, user data or holdout evaluation were added. **Neither Day 13 deliverable is complete.**

- Added a CV-only leaderboard service which reads the latest published completed CV attempt, validates candidate identities/statuses and finite values, and preserves the already verified fold-mean ranking, baseline deltas and timing. It reads the checksum-verified private result bundle, not a holdout table, and does not combine attempts. A SQLite authorizer denies holdout-table access in the tests; an actual spawned regression fixture supplies published data.
- Added a restricted FastAPI factory for job state, cancellation and the leaderboard, with explicit session-token injection, loopback Host checks, exact configured local Origins, narrow browser preflight, bounded empty cancellation bodies and opaque UUID validation. It does not launch a server or worker. Callers must generate a strong secret token and bind the server to loopback; token creation/distribution and production startup remain pending. There is no upload, configuration, experiment creation, job enqueue or candidate inspection API.
- The current worker requires a preconstructed in-memory `PreparedExperiment`; uploaded dataset persistence and safe restart reconstruction are not implemented. A queue endpoint alone would not provide a working uploaded-dataset workflow. Deletion intent, active-reference checks, private-file cleanup and keyed evaluation tombstones are also **not implemented**; MLflow deletion remains scheduled for Day 25. Do not interpret this partial API as permission to delete user data or as completion of D13a/D13b. The specification and local plan's remaining acceptance criteria still apply.
- Test-first: the new leaderboard and API suites initially failed collection because their modules were absent. The initial combined run passed six tests. The full suite after expanding coverage passed **403 tests in 190.33 seconds**, zero skips, with upstream Starlette and MLflow/SQLAlchemy deprecation warnings. `uv lock --check` and `git diff --check` passed. A subsequent test-only change also exercises the HTTP leaderboard against a real published bundle; see the final verification below.
- Final targeted verification: `uv run --locked pytest tests/integration/test_api.py tests/integration/test_leaderboard.py -q` passed **9 tests in 15.99 seconds**, zero skips, with the same two upstream warnings. `graphify update .` refreshed the ignored AST graph; community labels may need review.

G1 remains passed locally; G2â€“G4 remain pending. No final evaluation, benchmark or release is claimed.

## Day 12 â€” 2026-10-30

The user requested Day 12, English output, commit and push. Work started on `main` with a clean checkout. No dependency changes, external dataset downloads, final holdout evaluation, frontend work or user data were added.

- Classical worker output, including the Day 11 bounded out-of-fold diagnostics, is written into a private, checksummed `results.json`. A manifest records dataset fingerprints, configuration, hashed partition/fold identities and inner seeds, model/preprocessing settings, metric values and null reasons, versions, lock hash, resource telemetry, observed child pools, Git revision, dirty flag and tracked-diff hash. The tracked-diff hash does not include untracked file contents. The manifest itself does not embed diagnostic row values; the private results and MLflow artifact copy do contain bounded training predictions and labels.
- A separate local SQLite MLflow store and local artifact root hold a parent attempt run, candidate runs and fold runs, tied together by explicit parent tags and job/attempt IDs. Only finite metric values are logged; undefined values and reasons remain in the manifest. Manifest/results artifacts are read back and checked before completion. No MLflow server or network is needed.
- The bundle is flushed and renamed within one filesystem before a single guarded application SQLite transaction inserts candidate pointers and completes the job. Cancellation wins if recorded before that transaction; any staging/tracking failure fails the job. SQLite insertion failures roll back candidate rows, then attempt to mark the job failed; if SQLite remains unavailable the lease stays held for verified-death recovery. Reading a published result checks both manifest and result hashes. File and database writes across independent stores are not a power-loss-atomic transaction; Windows directory metadata flush guarantees are not claimed.
- Startup reconciliation marks MLflow runs belonging to failed/cancelled/interrupted attempts as orphaned and identifies their unreferenced bundles; it does not delete evidence, mutate active/unknown attempts, or automatically promote an orphan. Retention and complete permanent MLflow deletion remain Days 13/25. This process can leave private failed/staged artifacts until that work is implemented. A completed classical job now means durable CV results; probe jobs remain test-only and do not publish candidate bundles.
- Test-first: the three initial tracking tests failed because the tracking module and publication interface did not exist. After implementation, `uv run --locked pytest tests/integration/test_tracking.py -q`: **7 passed**, one upstream MLflow/SQLAlchemy deprecation warning. Tests inspect real local runs and artifacts, reopen the repository, verify diagnostics/checksums, inject disk/MLflow/SQLite failures, test cancel-versus-publication and idempotent orphan marking, and check that undefined metrics are not logged as NaN. The full `uv run --locked pytest -q` initially passed **394 tests in 151.28 s**. A subsequent run had one failure because a test job was correctly stopped with `resource_pressure` before reaching its injected database failure (393 passed); the final full rerun passed **394 in 256.24 s**, one warning, zero skips. No resource limit was relaxed. `uv lock --check` and `git diff --check` passed; `graphify update .` refreshed the ignored AST graph (community labels may need relabeling).

G1 remains passed locally; G2â€“G4 and release remain pending. No named-dataset demonstration, resource selection or holdout score is claimed. California Housing requires an explicit download authorization or a user-prepared cache before the Day 21 benchmark; no download was performed.

## Day 11 â€” 2026-10-30

The user requested Day 11, English output, commit and push. Work continued on `feat/day-7-durable-queue` from clean commit `4ea6dbe`. No dependency changes, downloads, holdout evaluations, or user data were added.

- CV comparison verifies the frozen manifest, task, feature configuration, fold identities, primary metric/direction, model seed and resource configuration before mixing candidates. It recalculates fold means and population standard deviations from actual complete folds; failed, timed-out, duplicate or incomplete folds and undefined primary metrics remain unranked. Baseline deltas have positive-is-better direction, are null if Dummy is ineligible or absent, and ties within 1e-8 of a group's best use mean fit time and family name. Timing is separate from score.
- Diagnostics aggregate outer-validation predictions only. Each displayed prediction retains its row and fold identity; full OOF scalar metrics/confusion matrix and classification ROC/PR curves or regression residual/prediction points are explicitly labeled training OOF. Display points and each curve are evenly sampled deterministically, capped by the frozen `max_diagnostic_points` setting (at most 2,000). Fold-mean ranking never uses pooled OOF metrics. Failed or incomplete candidates have no pooled diagnostics. These predictions and labels are private data and are not logged or committed.
- The spawned child returns the comparison and diagnostics alongside candidates in the supervisor's in-memory result. **Durable persistence is not implemented:** the Day 11 persistence acceptance remains pending until Day 12's private MLflow/manifest and atomic-publication work. A completed job does not mean the data survives restart. The existing supervisor enforces the overall job deadline and discards timed-out job output; there is still no independently enforced per-fold native-fit deadline. No partial or timed-out candidate is promoted by the comparison function.
- Test-first: the new CV protocol suite initially failed collection because the comparison interface did not exist; the worker integration then failed because comparison was absent from the child payload. Both passed after implementation. `uv run --locked pytest tests/integration/test_cv_protocol.py tests/integration/test_baseline_cv.py -q`: **32 passed**. `uv run --locked pytest tests/integration/test_classical_suite.py -q`: **7 passed**, including actual Windows spawned children for both tasks. `uv run --locked pytest -q`: **387 passed in 101.55 s**, zero skips. `uv lock --check` and `git diff --check` passed.

G1 remains passed locally; G2â€“G4 remain pending. These fixture tests are not named demonstration, benchmark, final holdout or release evidence.

## Day 10 â€” 2026-10-30

The user requested Day 10, English output, commit and push. Work continued on `feat/day-7-durable-queue`; the starting checkout was clean at `c33ffca`. No dependencies, lockfiles, user data, dataset downloads, holdout scoring or frontend work were added.

The closed registry now supplies Dummy, LogisticRegression/Ridge, RandomForest and HistGradientBoosting for the relevant task, with the specified bounded defaults and HGB early stopping disabled. The suite serially evaluates each candidate against the same frozen outer folds using fresh guarded fold-local preprocessing, selects binary probability column 1 and reports fit/score timings and safe fold failures. Failed and partial candidates remain ineligible. Cancellation is checked between folds and candidates; the existing supervisor also handles cancellation during native fits.

The classical worker accepts internally prepared training-only input and verifies its experiment, split, configuration and run metadata against the queued immutable records before spawning. It does not accept a client path or estimator, and the child receives no holdout contents. Source-content provenance still depends on the trusted `prepare_experiment` / `training_from_manifest` boundary, not on reconstructing a manually modified training view. The current supervisor accepts one supplied prepared experiment, fails closed on a different queued job, and returns results only in `last_result` memory. Job completion does **not** mean candidate results were durably published: private source loading, candidate publication, MLflow tracking and leaderboard operations remain future work. The existing job deadline applies; a separate classical per-fold deadline is not implemented.

Verification on Windows and the existing locked environment:

- Test-first: the new registry and suite commands initially failed collection because their new interfaces/modules did not exist. After implementation, the registry and suite each passed.
- The first full regression run found **2 failures, 365 passes**: the new optional input had changed the probe child call signature used by the native-fit cancellation tests. The supervisor now passes that input only for the classical workload; the lifecycle and classical integration suites then passed **26 tests**, zero skips.
- Final `uv run --locked pytest -q`: **369 passed in 146.37 s**, zero skips. Named commands, rerun separately: `uv run --locked pytest tests/unit/test_classical_models.py -q` â€” **5 passed in 7.01 s**; `uv run --locked pytest tests/integration/test_classical_suite.py -q` â€” **7 passed in 37.24 s**.
- Real Windows-spawned children completed all four families on both synthetic tasks using five shared folds. Tests also check fresh fold-only preprocessing for all 20 fits, finite model outputs, safe failed-family visibility and ineligibility, deterministic cancellation between candidates, real worker cancellation and a second supervisor blocked by the occupied slot. The frozen-input mismatch is rejected before child spawn. Fixture outcomes are not named-dataset or performance evidence.
- `uv lock --check` passed. `graphify update .` refreshed the ignored AST graph; its labels and semantic documentation may require a later refresh.

G1 remains passed locally; G2â€“G4 and release remain pending. No model was selected or evaluated on the holdout, no baseline-win or benchmark claim is made, and no P0 requirement was removed.

## Day 9 â€” 2026-10-27

The user authorized Day 9 implementation, English output, commit and push, and chose the existing `feat/day-7-durable-queue` branch. The clean baseline was `4edf0f2`; `uv run --locked pytest -q` passed **336 tests in 38.20 s**, zero skips. No dependencies, lockfiles, downloads, frontend or holdout evaluation changed. Days 10â€“28 remain pending.

Delivered:

- Queued cancellation without spawn; running cancellation via a multiprocessing Event, five-second cooperative grace, then verified-child termination/join and two-second kill fallback. KeyboardInterrupt joins the active child, records interruption and preserves waiting jobs.
- SQLite transaction arbitration: cancellation persisted before completion prevents success; an already committed completion stays completed. Noncompleted attempts invalidate candidate status and discard in-memory result payloads, including late cancellation after result receipt.
- Schema v2 adds per-attempt supervisor/child PID and exact creation identity; an atomic v1 migration preserves existing metadata and queued jobs. The child cannot begin its workload before its identity is durably registered and its start gate opens.
- Recovery requires both supervisor and registered child to be verified dead before interruption, generation advancement and slot release. A live supervisor or orphan, stale heartbeat alone, inaccessible identity, legacy ownership or spawn-registration gap cannot authorize new fitting. No automatic orphan termination or force-unlock is provided.
- Explicit whole-CV retry reuses the immutable experiment, configuration and split, creates a new job/attempt, and retains the previous terminal record. Old attempts and stale generations cannot publish. Reopening the repository (the available API-restart boundary) does not stop active execution.

Test-first evidence included failures for cancel arbitration, cooperative exit, KeyboardInterrupt handling, missing recovery/retry contracts, generation fencing, candidate invalidation, late result cancellation and unobservable-child cleanup before implementation. The updated Day 8 crash assertion now expects `interrupted`, as required by the Day 9 lifecycle; allowlisted workload exceptions remain `failed`.

Fresh verification before review: `uv run --locked pytest -q` -> **356 passed in 64.11 s**, zero skips. The 20 new lifecycle tests include a real spawned scikit-learn HistGradientBoosting fit interrupted within ten seconds, cooperative exit without termination, actual supervisor/child termination, no-overlap restart, migration and fail-closed ownership. Synthetic fixtures only; no performance benchmark or named demonstration claim.

Independent read-only review found no Critical issues, one Important issue and no new minors: cancellation polling followed a configurable heartbeat that could reach 60 seconds. A regression using a real native fit and 60-second heartbeat failed with a 12-second timeout before the fix. Cancellation now polls independently at 0.2 seconds and bounds the monitoring sleep even with a five-second poll setting. Final checkout suite: `uv run --locked pytest -q` -> **357 passed in 72.59 s**, zero skips. The native-fit deadline regression passes at both short and maximum heartbeat/poll settings. The reviewer did not independently rerun the suite; its deterministic reproduction and the implementer's REDâ†’GREEN run are separate evidence. Earlier staged export also passed 356 tests before this fix; final staged verification is recorded below when executed.

Review boundaries: future CV/leaderboard/MLflow/bundle integration remains out of Day 9 scope and requires its later acceptance tests. Public documentation was checked by the implementer, not the code reviewer. `uv lock --check` passed without lock changes. `graphify update .` refreshed the ignored AST code graph; semantic documentation nodes were not refreshed, and community labels may need a later relabel.

Final staged-tree verification: exported the exact index with `git checkout-index --all` into a fresh approved temporary directory, then ran `uv run --project <export> --directory <export> --locked pytest -q` using the existing locked environment via `UV_PROJECT_ENVIRONMENT`: **357 passed in 68.68 s**, zero skips. This verifies staged source independently of the checkout, not a fresh dependency installation. Only README's delivery-state wording and this evidence paragraph were added afterward. Explicit source/test/documentation paths were staged; local plans, graph output, generated data and artifacts remain ignored. No merge, PR or release is implied by committing/pushing Day 9.

Execution decisions: use the existing ignored per-day ledger for the daily-heading plan; block on live orphan/unknown ownership rather than automatically killing or guessing. This favors safety over automatic recovery; missing identity requires operator investigation. The historical direct-SQL nullable-generation CHECK minor remains deferred; repository claim validation and recovery fencing do not treat missing ownership as safe. Dataset CV, MLflow/bundle reconciliation, HTTP lifecycle routes and UI remain assigned to later days. G1 remains passed locally; G2â€“G4 and release acceptance remain pending.

## Day 8 â€” 2026-10-26

The user authorized Day 8 implementation, English output, commit and push, continuing on the existing checkout and branch `feat/day-7-durable-queue`. No dependency, lockfile, frontend scaffold, network dataset fetch or holdout evaluation was added. Days 9â€“28 remain unimplemented.

Delivered:

- `apps/api/workers/control.py`: a shared, import-light vocabulary for stages, terminal states and safe error codes, plus a typed workload registry. Stages and error codes are closed allowlists because they are persisted and rendered; free text, paths and exception messages cannot enter them. An unknown workload name or option key is rejected before any process is spawned, so no client-supplied path or import string is ever resolved.
- `apps/api/workers/monitor.py`: the single definition of normalized CPU (100% means all logical processors, so one busy core of twelve is ~8.3%), sampled child RSS/CPU plus system-available memory, a continuous-breach memory-pressure tracker, and `terminate_owned_child`, which compares the `psutil` creation time before signalling and returns `False` for an unverified PID. A recycled PID can therefore never be killed.
- `apps/api/workers/execute.py`: the spawn entrypoint. Module scope imports no numeric library, and `apply_thread_environment` runs before the first numerical import. Each measurement stage enables exactly one T-wide layer â€” BLAS at T with OpenMP and torch intra-op pinned to one, then torch intra-op at T with BLAS and OpenMP pinned to one â€” and reports both stages so nesting cannot hide inside one snapshot. A deliberate workload failure emits only the closed code `workload_failed`; the exception text never crosses the process boundary.
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
| `uv run --locked pytest tests/integration/test_worker_spawn.py -q` | 11 passed, zero skips (10.2â€“11.5 s) |
| `uv run --locked pytest tests/integration/test_resource_monitoring.py -q` | 15 passed, zero skips (18.3â€“19.4 s) |
| Both new suites together | 26 passed, zero skips, repeated eight times (26.3â€“28.8 s) |
| Full regression `uv run --locked pytest -q` | **336 passed, zero skips** in every run; observed wall time 36.9â€“48.4 s across dozens of consecutive runs after the final code change |
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

- The child runs the internal synthetic `probe` workload. Dataset cross-validation inside the spawned child arrives with the classical registry on Days 10â€“12; no user data, holdout partition or trained artifact is involved in these tests.
- Cancellation is a single check on the heartbeat cadence with no cooperative grace period, no `Event`, and no complete cancel-versus-result arbitration matrix. Queued cancellation already worked from Day 7. Full lifecycle handling, interruption recovery, stale-owner reconciliation and restart behaviour remain Day 9.
- Resource limits are sampled, so a transient RSS spike between samples can be missed. This is a soft cap, not an OS-enforced quota, and it does not justify a final thread or memory policy.
- No MLflow tracking, candidate publication, API transport, dashboard, or holdout evaluation exists yet; the supervisor is a library called by tests, not a running background service.

## Day 7 â€” 2026-10-26

The user authorized Day 7 implementation, English output, commit and push, and chose the current checkout on a new branch (`feat/day-7-durable-queue`). No dependencies, frontend scaffold, network dataset fetch or holdout evaluation were added. Days 8â€“28 remain unimplemented.

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

**G1 data-correctness gate passes locally.** This does not establish a working background process or a release. Day 7's integration explicitly calls CV from test code; preparing or enqueueing an experiment never runs training. Candidate publication, MLflow verification, worker execution/heartbeats, stale-owner liveness checks, cancellation of native fitting and terminal result transitions belong to Days 8â€“12. The initial lease generation is one; no expiry-based takeover is permitted. Final holdout reservation and one-time evaluation remain Days 14/19. Upload path persistence, HTTP transport and retention are not implemented by these metadata primitives.

Initial resource decision: retain **provisional T=4 and 4 GiB RSS**, with all existing caps unchanged. The actual Day 2 probe recorded 5.29885 s total wall time, 152.36 MiB peak sampled process RSS and 4.8202 GiB minimum sampled available RAM. It was a small battery-powered parser/Dummy probe, not a full classical/MLP or usability benchmark; it cannot justify raising/lowering limits or selecting a final thread count. Day 21's 2/4/6 comparison remains required.

Independent read-only review approved Day 7 with **no Critical/Important findings** and independently reran the 310-test G1 gate and 33-test Day 7 suite, with zero skips. One minor is deferred: SQLite's running-job `CHECK` accepts NULL `owner_generation` when written directly through SQL, because a NULL CHECK expression is not false. The implemented claim path always writes a validated positive generation; no repository-path failure was found. Add an explicit `IS NOT NULL` constraint and regression when extending worker state/schema. This does not authorize treating malformed ownership as safe during recovery.

Review scope decisions: native cancellation/liveness/recovery remain Days 8â€“9; tracking/publication remains Day 12; finalization crash boundaries remain Days 14/19; complete deletion remains Days 13/25. Existing Day 3/5 duplicate/preprocessing checks were rerun in G1 and their service boundaries reviewed; this is not a new exhaustive audit of those modules. None of these future features is claimed as delivered. If their later integration bypasses the recorded boundaries, G1 alone cannot establish overall safety.

Commit verification: `4b3ca13` â€” `feat(jobs): persist idempotent single-worker queue and experiment metadata`. All seven new implementation/test files were staged explicitly, and the complete staged diff and whitespace checks were reviewed. The full suite passed again immediately before commit: **310 passed in 14.69 s, zero skips**. Only README and this evidence file remained modified afterward; runtime/user data and local planning/graph files were not staged. A following documentation commit publishes the verified delivery and G1 status. Remote push is verified separately after these commits, not presumed from local commits.

## Day 6 â€” 2026-10-26

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

Final pre-commit verification on the completed source: `uv run --locked pytest tests/unit/test_metrics.py -q` â€” **37 passed in 3.37 s**; `uv run --locked pytest tests/integration/test_baseline_cv.py -q` â€” **14 passed in 5.92 s**; `uv run --locked pytest -q` â€” **277 passed in 10.29 s**, all with zero skips. Lock and whitespace checks also passed. The requested delivery is a new remote branch, not a merge to `main` or a release; its commit identifier is available in Git history.

Nonblocking review Minor, deferred: fold failures currently share the safe `fold_execution_failed` code. Ranking is safe and raw exception text is suppressed, but categorized resource/data/execution diagnostics would make remediation easier. Detailed safe diagnostics belong with later worker/tracking integration; this limitation is not a claim of production-ready error reporting.

Execution decisions: use direct D6a/D6b tasks and a local ignored ledger rather than task-extraction scripts for standalone numbered plans (bookkeeping only). Continue to defer worker cancellation, hard native-fit deadlines, durable publication and additional classical families to their assigned days; this evaluator is not exposed on an API request thread. The source-content identity boundary remains `training_from_manifest`; the evaluator rechecks configuration/manifest/row identity, not raw dataset fingerprints from manually constructed views. In-progress documentation was checked separately by the implementer. G1 remains blocked on Day 7; G2â€“G4 remain unpassed.

## Day 1 â€” 2026-10-22

Scope authorized by the user's request to implement Day 1 only. Day 2 onward, frontend scaffolding, training, benchmarks and dataset downloads were not executed. The user confirmed 2â€“3 focused hours per day.

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

## Day 1 commit verification â€” 2026-10-23

The user explicitly requested the four planned Day 1 commits. Existing implementation was packaged in dependency order, with no training, benchmark, frontend scaffold or Day 2 implementation. The prior Day 1 results above remain historical evidence.

| Commit | Delivered scope | Fresh verification |
|---|---|---|
| `2dc1dc7` | Locked Python/Node toolchain, CPU torch source, root npm manifest and local-workspace ignore rule | Exported staged tree: `uv lock --check`, `uv sync --locked`, CPU import smoke check and `npm.cmd ci --ignore-scripts` all exited 0. Fresh environment installed 105 packages; npm reported zero vulnerabilities. |
| `f989c16` | Immutable resource/run contracts, local storage settings and owning tests | Exported cumulative staged tree: `uv run --locked pytest tests/unit/test_settings.py -q` â€” 58 passed in 0.53 s, zero skips. |
| `de54447` | Hardware inventory command and owning test | Exported cumulative staged tree: inventory test â€” 1 passed in 3.24 s; full suite â€” 59 passed in 3.41 s, zero skips; inventory CLI exited 0. |

The staged trees were exported with `git checkout-index` to a new temporary directory outside the repository, using a separate freshly synchronized virtual environment. Only staged files and prior commits were exported; later unstaged source could not satisfy earlier checks. The export has no `.git`, so its inventory correctly reported code revision unavailable. No source or dependency versions were changed during packaging.

CPU smoke command executed on the first staged tree:

```powershell
uv run --locked python -c "import fastapi, sklearn, torch, mlflow; print(torch.__version__); assert torch.version.cuda is None; assert not torch.cuda.is_available()"
```

It reported `2.14.1+cpu` and exited 0. Root npm verification covers the toolchain manifest only, with no frontend dependencies. Representative `.env`, data, uploads, artifacts, runtime files, virtualenv, node_modules, model weights and SQLite paths were confirmed ignored with `git check-ignore`.

After `de54447`, the actual checkout also passed `uv run --locked pytest -q`: **59 passed in 3.29 s, zero skips**. `uv lock --check` and `git diff --check` exited 0. The inventory CLI was rerun successfully in this checkout and recorded revision `de54447533c87bc7bfefa9e436228a913655ba29` at `2026-10-22T22:43:11.860430+00:00` (2026-10-23 in Asia/Jakarta). It observed the reference hardware, 6.037 GiB available RAM, Balanced power scheme and battery power. This snapshot is inventory, not benchmark or resource-safety evidence. The generated JSON remains in private application storage outside Git.

Inventory lock checksums describe actual file bytes; Windows checkout line-ending conversion can produce different hashes between an exported tree and the original working copy. No deterministic experiment rerun is claimed by this setup check.

The fourth commit publishes this evidence and the README delivery table. Its identifier is available from Git history after creation; the document does not attempt to embed its own commit hash. Day 1 is committed and verified; all four milestone gates remain unpassed. No remote push or release is included in this delivery.

## Day 2 â€” 2026-10-23

The user authorized Day 2 implementation and requested English output. Work stayed in the requested checkout and added no dependencies. Day 3 onward remains unexecuted. The initial delivery was verified locally and left uncommitted for review; the subsequent authorized commits are recorded below. No push or release occurred.

Delivered structural `validate_csv` with UTF-8/BOM decoding, strict comma/quote syntax, preserved duplicate-header checks, missing-cell preservation, and byte/row/column/UTF-8-cell bounds. It hashes the same stream it parses and rejects ordinary file substitution using pre-open/post-read file identity, size, and modification checks. Trusted service callers can supply a private root; HTTP path handling is not implemented yet. Parsing scans 4 KiB decoded chunks and checks column/cell limits before CSV record allocation. A conservative three-times Python-object estimate additionally bounds retained parsed data against the existing preprocessing memory budget; this is an allocation guard, not an OS memory quota.

Canonical policy v1 hashes version-prefixed compact JSON records with explicit record separators. It includes headers, values, row order, and column order. BOM, equivalent quoting, and CR/CRLF/LF formatting do not change canonical identity; embedded newlines normalize to LF. Empty strings represent missing input structurally. Whitespace, `NA`, and numeric spellings remain literal strings; task/schema inference is deferred to training-only configuration. The result contains no analytical statistics, and its representation omits rows.

Delivered the initial resource recorder and CLI. It samples current-process RSS, process CPU-seconds, and available system memory at start, finish, and approximately every 0.5 seconds. Wall time uses the high-resolution monotonic `perf_counter`; normalized process CPU is `100 * CPU-seconds / wall-seconds / logical CPUs`. Samples do not include descendants or guarantee detection of transient RSS peaks. The command configures BLAS up to T and subordinate OpenMP to one thread, runs serial folds, records actual pools, and prevents overlapping initial probes with an exclusive lock. The future worker owns cross-process admission, cancellation, timeout, and RSS termination.

Executed checks:

- The CSV and benchmark tests initially failed because their production modules did not exist. After implementation, the named suites passed. A fresh reviewer identified excessive allocation before column rejection. `test_delimiter_flood_is_rejected_before_large_cell_list_allocation` reproduced it with an approximately 18.6 MB Python allocation for a 1 MB delimiter input; the streaming guard reduced it below the test's conservative 6 MB ceiling. This is parser allocation evidence, not whole-process RSS evidence.
- `test_recorder_handles_work_shorter_than_a_clock_tick` reproduced a zero-duration measurement on Windows. Switching from Python 3.12's coarse `monotonic` clock to high-resolution monotonic `perf_counter` fixed it.
- The existing equivalent-format and exact-boundary CSV cases exposed inconsistent Windows `stat`/`fstat` ctime semantics after rewrites. File identity checks now use device, inode, size, and mtime consistently; those tests and the replacement-before-open test pass.
- Observed pools exposed both BLAS and OpenMP configured T-wide. An added assertion failed, then passed after OpenMP was restricted to one. Earlier local probe reports remain retained as superseded observations; they are not comparative performance trials.
- Final source verification: `uv run --locked pytest -q` â€” **103 passed in 11.18 seconds, zero skips**. The two Day 2 suites contain 44 tests; the 59 existing tests also pass. `uv lock --check` and `git diff --check` exited 0.

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

## Day 2 commit verification â€” 2026-10-23

The user requested commits with appropriate messages. The existing implementation and its tests were committed in dependency order, followed by the documentation. No implementation or dependency versions changed during packaging.

| Commit | Delivered scope | Fresh staged verification |
|---|---|---|
| `6270779` â€” `feat(data): validate bounded CSV input and stable fingerprints` | Structural CSV parsing, safe validation errors, deterministic raw/canonical identity, and owning tests | Full exported staged tree: **90 passed in 4.01 s, zero skips**; benchmark files were absent from this snapshot. |
| `44be1cb` â€” `feat(benchmark): record initial CPU resource observations` | Monotonic resource sampling, training-only parser/Dummy probe, CLI, and owning tests | Full exported staged tree: **103 passed in 11.34 s, zero skips**. |

Each staged snapshot was exported with `git checkout-index --all` into a newly created temporary directory outside the repository. Tests ran from that directory using `uv run --project <export> --directory <export> --locked pytest -q` with `UV_PROJECT_ENVIRONMENT` pointing to the existing locked checkout environment. This isolates source files from later unstaged changes; it is not a fresh dependency-installation check. The exported snapshots have no Git metadata, so their hardware-inventory test accepts unavailable revision information.

Before each code commit, the staged names, complete diff, and `git diff --cached --check` were reviewed. `git status --short` was checked after each commit. `uv lock --check` also exited 0. Generated benchmark JSON, temporary verification exports, and ignored local planning documents were excluded from the commits. The documentation commit publishes the actual code hashes and evidence; its own identifier is available from Git history. Resource settings remain provisional, and the deferred minor above remains recorded.

## Day 3 â€” 2026-10-26

The user explicitly requested execution of Day 3 with English output. D3a and D3b are verified locally against the existing locked Windows/Python 3.12 environment. No dependencies changed, training ran, dataset downloads occurred, or commits were created. Days 4â€“28 remain unexecuted and G1â€“G4 remain unpassed.

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

The following documentation commit publishes the actual code hash and verification evidence. Ignored architecture/implementation planning documents and runtime artifacts remain local. G1â€“G4 remain unpassed, and Day 4 onward remains unexecuted. The requested push includes existing local history that was absent from `origin/main`; this is progress publication, not a release.

## Day 4 â€” 2026-10-26

The user requested continuation of an in-progress Day 4 implementation. The starting checkout contained modified `ml/contracts.py` and untracked `ml/data/splits.py`/`tests/unit/test_splits.py`; those changes were continued in place. D4a/D4b are now verified locally. No dependencies changed, training or downloads ran, or commits/pushes occurred in this session. Days 5â€“28 remain unexecuted and all four release gates remain unpassed.

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
- Independent read-only review reported **no remaining Critical or Important findings**, reran all 55 split tests, and additionally confirmed binary-search inner sizing agrees with linear search for counts 4â€“34, every valid minority count and four seeds.

Interface decisions and rulings:

- All data remain under parser policy v1. Row IDs are positional and sorted on persistence. Partition hashes cover only their corresponding original records/columns/row IDs and partition name; the separate complete-dataset fingerprint binds the overall source. No raw rows are written to the manifest.
- Use ceiling for the default 20% sizes, matching a fractional split policy, and increase to the documented two-row minimum. Binary inner validation may grow further to the smallest feasible size; persist this expansion. Stable ordering of original label strings prevents positive-label polarity from changing assignments. Changing model seed has no effect on partitions.
- Keep the complete inner-MLP requirements during fold adaptation. Although the plan names 4/3/2 adaptation, two folds may never be the largest feasible choice under the present minimums. Two-fold candidates are supported and directly tested; the selected-five/four/three fixtures and rejection cases preserve all safety requirements. The reviewer agreed that no P0 minimum should be relaxed to manufacture a two-fold selection.
- JSON reload is bounded to 16 MB and rejects duplicate fields. Same-directory publication uses a closed/flushed/fsynced temporary file and a no-clobber hard link; unsupported filesystems fail rather than overwrite. Identical retries succeed, different records conflict, and simulated publication failure leaves no final or temporary file. Private service-owned paths remain required.
- The split module does not replace future SQLite evaluation-scope enforcement. Scope lookup must reuse the original split and prohibit fresh holdout opportunities after feature/metric/seed changes. Checksums are integrity checks, not authentication. Fold-local preprocessing, resource guards, worker lifecycle and one-time finalization remain assigned to later tasks.

This completes the Day 4 profile-partition verification deferred from Day 3, but does not claim G1 completion: Days 5â€“7 remain pending. Existing documented Day 3 subnormal-mean precision and Day 2 initial-probe test limitations remain unchanged.

### Day 4 commit verification

The user subsequently requested committing and pushing the changes. Commit `64a8804` (`feat(data): persist leakage-safe holdout and fold identities`) contains the split contracts, implementation, owning test suite and experiment protocol. These form one reviewable partitioning change; earlier notes describe the initial uncommitted handover.

Before committing, the four exact paths were staged, the staged names and diff were reviewed, and `git diff --cached --check` exited 0. A fresh `git checkout-index --all` export into a new temporary directory was tested with `uv run --project <export> --directory <export> --locked pytest -q`, using the existing locked environment via `UV_PROJECT_ENVIRONMENT`: **205 passed in 31.01 s, zero skips**. This verifies the staged source independently of the unstaged README/evidence changes; it is not a fresh dependency-installation check. `uv lock --check` also exited 0. Post-commit status showed only README and this validation document pending.

The following documentation commit publishes the actual code hash and verification result. Local planning documents remain ignored; runtime files, datasets and generated artifacts are excluded. Day 5 onward remains unexecuted and G1â€“G4 remain unpassed. Pushing this progress does not constitute a release.

## Day 5 â€” 2026-10-26

The user authorized Day 5 implementation, commit and push. Delivered a fresh preprocessing factory and guarded `FoldPreprocessor` in `ml/preprocessing/factory.py`, with owning tests in `tests/unit/test_preprocessing.py` and `tests/unit/test_resource_estimates.py`. No dependencies changed, candidate training or downloads ran. Days 6â€“28 remain unexecuted and G1â€“G4 remain unpassed.

Numeric median imputation/scaling and categorical/boolean missing imputation/one-hot encoding learn only from fitting rows. Empty numeric columns remain zero; constant features and empty categorical columns keep stable finite output. Namespaced categories cannot collide with the missing sentinel. Unknown categories transform without changing fitted width. The feature matrix excludes targets. The wrapper accepts isolated views with matching configuration, records fitting row IDs and creates a fresh transformer on refit; callers remain responsible for selecting persisted fitting identities.

Training-only estimates enforce policy-v1 category/width limits and the 512 MiB preprocessing budget before fit. The live-buffer estimate counts fitting and transform rows, float64 intermediates/copies, float32 cast space and a three-times safety factor. Transform allocation is checked before encoding; realized width/shape and budget are rechecked before explicit dense conversion. Output finiteness and float32 range are checked. See [experiment protocol](EXPERIMENT_PROTOCOL.md#day-5-preprocessing-boundary) for the exact formula, policies and caller obligations. This estimate does not enforce process RSS or cover model allocations; worker enforcement remains later work.

Executed checks in the locked environment:

- Initial test-first run against unimplemented interfaces: **21 failed**, demonstrating absent Day 5 behavior.
- `uv run --locked pytest tests/unit/test_preprocessing.py tests/unit/test_resource_estimates.py -q`: **21 passed in 6.37 s, zero skips**.
- `uv run --locked pytest -q`: **226 passed in 16.74 s, zero skips**, including all 205 previous tests.
- `uv lock --check`: exited 0, resolved 107 packages without lock changes.
- `git diff --check` and final `git diff --cached --check`: exited 0. The staged file list contains only the seven Day 5 source/test/documentation paths.

A separate read-only review found no actionable correctness or leakage issues and independently ran the two new test files: **21 passed, zero skips**. No candidate-score reproducibility, hardware performance, worker resource safety or milestone completion is implied. The local ignored implementation plan marks D5a/D5b complete; it remains outside Git under the existing repository policy. Source, tests, protocol and public validation evidence are included in the Day 5 commit.
