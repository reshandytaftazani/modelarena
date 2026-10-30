"""Immutable holdout reservation, pre-evaluation retry, and one-time evaluation gate."""

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from apps.api.database.repository import ConflictError, JobRecord, JobRepository, NotFoundError, _now
from apps.api.services.retention import find_tombstone
from ml.contracts import FinalResult, FrozenSelection, HoldoutCapability, MetricBundle, MetricValue, TaskType
from ml.tracking.manifest import canonical_bytes, digest_bytes


def compute_candidate_digest(
    candidate_id: str,
    family: str = "",
    split_sha256: str = "",
    config_sha256: str = "",
    primary_metric: str = "",
    primary_score: float | None = 0.0,
    model_seed: int = 42,
) -> str:
    """Compute a deterministic 64-character SHA-256 digest of candidate selection."""
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Candidate ID required")
    payload = [
        "modelarena-candidate-digest-v1",
        candidate_id,
        family,
        split_sha256,
        config_sha256,
        primary_metric,
        round(float(primary_score), 6) if primary_score is not None and math.isfinite(primary_score) else None,
        int(model_seed),
    ]
    return digest_bytes(canonical_bytes(payload))


def reserve(
    repository: JobRepository,
    experiment_id: str,
    candidate_id: str,
    expected_digest: str,
    acknowledge: bool,
    idempotency_key: str | None = None,
) -> tuple[FrozenSelection, JobRecord | None]:
    """Atomically reserve the unique evaluation scope for this candidate."""
    if not acknowledge:
        raise ValueError("Explicit acknowledgement is required before reserving holdout evaluation")

    with repository._connection() as db:
        pending = db.execute(
            "SELECT 1 FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN datasets source ON source.dataset_id=c.dataset_id "
            "JOIN datasets d ON d.canonical_sha256=source.canonical_sha256 "
            "JOIN deletion_intents i ON i.dataset_id=d.dataset_id WHERE e.experiment_id=?",
            (experiment_id,),
        ).fetchone()
    if pending:
        raise ConflictError("Dataset deletion is pending")

    with repository._connection() as db:
        row = db.execute(
            "SELECT c.scope_id, s.canonical_sha256, s.target, s.task "
            "FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN evaluation_scopes s ON s.scope_id=c.scope_id WHERE e.experiment_id=?",
            (experiment_id,),
        ).fetchone()
    if row is None:
        raise NotFoundError("Experiment not found")

    scope_id = row["scope_id"]
    tombstone = find_tombstone(repository, row["canonical_sha256"], row["target"], row["task"])
    if tombstone == "consumed":
        raise ConflictError("Holdout evaluation scope has already been consumed")

    selection = repository.reserve_finalization(scope_id, candidate_id, expected_digest)
    key = idempotency_key or f"finalization-{selection.finalization_id}"
    job = repository.enqueue_finalization(selection.finalization_id, key)
    return selection, job


def get_finalization_status(repository: JobRepository, experiment_id: str) -> dict:
    """Return locked finalization state, selected candidate, and audit trail."""
    with repository._connection() as db:
        row = db.execute(
            "SELECT c.scope_id, s.canonical_sha256, s.target, s.task "
            "FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN evaluation_scopes s ON s.scope_id=c.scope_id WHERE e.experiment_id=?",
            (experiment_id,),
        ).fetchone()
    if row is None:
        raise NotFoundError("Experiment not found")

    scope_id = row["scope_id"]
    tombstone = find_tombstone(repository, row["canonical_sha256"], row["target"], row["task"])
    fin = repository.get_finalization(scope_id)

    if fin is None:
        state = "consumed" if tombstone == "consumed" else "unselected"
        return {
            "scope_id": scope_id,
            "state": state,
            "finalization_id": None,
            "candidate_id": None,
            "candidate_digest": None,
            "job_id": None,
            "audit": [],
            "result": None,
        }

    with repository._connection() as db:
        events = db.execute(
            "SELECT occurred_at, actor, event FROM audit_events WHERE entity_id=? ORDER BY sequence",
            (fin["finalization_id"],),
        ).fetchall()

    return {
        "scope_id": scope_id,
        "state": fin["state"],
        "finalization_id": fin["finalization_id"],
        "experiment_id": fin["experiment_id"],
        "candidate_id": fin["candidate_id"],
        "candidate_digest": fin["candidate_digest"],
        "job_id": fin["refit_job_id"],
        "evaluation_started_at": fin["evaluation_started_at"],
        "audit": [{"occurred_at": e["occurred_at"], "actor": e["actor"], "event": e["event"]} for e in events],
        "result": None,
    }


def retry(repository: JobRepository, experiment_id: str, idempotency_key: str) -> JobRecord:
    """Retry the existing immutable selection after verified pre-evaluation failure."""
    with repository._connection() as db:
        row = db.execute(
            "SELECT f.finalization_id FROM experiments e "
            "JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN finalizations f ON f.scope_id=c.scope_id WHERE e.experiment_id=?",
            (experiment_id,),
        ).fetchone()
    if row is None:
        raise NotFoundError("No finalization reservation found for this experiment")

    return repository.retry_pre_evaluation(row["finalization_id"], idempotency_key)


def start_evaluation(repository: JobRepository, finalization_id: str, expected_digest: str) -> HoldoutCapability:
    """Commit durable evaluation-start and issue the single-use HoldoutCapability."""
    scope_id, token = repository.start_evaluation(finalization_id, expected_digest)
    holdout_path = repository.path.parent / "datasets" / scope_id / "holdout.csv"
    return HoldoutCapability(scope_id=scope_id, holdout_path=str(holdout_path), _auth_token=token)


def issue_holdout_capability(repository: JobRepository, finalization_id: str, expected_digest: str) -> HoldoutCapability:
    return start_evaluation(repository, finalization_id, expected_digest)


def fixture_evaluate_holdout(repository: JobRepository, capability: HoldoutCapability) -> FinalResult:
    """Test-only fixture evaluator consuming HoldoutCapability without opening real CSV."""
    if not isinstance(capability, HoldoutCapability):
        raise TypeError("Valid HoldoutCapability required")

    fin = repository.get_finalization(capability.scope_id)
    if fin is None or fin["state"] != "evaluating" or fin["evaluation_started_at"] is None:
        raise ConflictError("Holdout evaluation must be started durably before evaluation")

    now = _now()
    bundle = MetricBundle(TaskType.REGRESSION, 100, (MetricValue("rmse", 0.85),))
    final_result = FinalResult(
        finalization_id=fin["finalization_id"],
        scope_id=capability.scope_id,
        candidate_id=fin["candidate_id"],
        task=TaskType.REGRESSION,
        primary_metric="rmse",
        metrics=bundle,
        evaluation_seconds=0.05,
        evaluated_at=now,
    )
    repository.record_evaluation_result(fin["finalization_id"], state="evaluated")
    return final_result
