"""Real SQLite contention, durable identity and queue transaction boundaries."""

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from threading import Barrier
from time import perf_counter

import pytest

from apps.api.database.repository import (
    ConflictError, JobRepository, NotFoundError, QueueFullError,
    StorageBusyError, WorkerIdentity,
)
from apps.api.services.jobs import prepare_experiment
from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, RunConfig, TaskType
from ml.data.csv_io import validate_csv


def experiment(repo, tmp_path, *, run=None):
    path = tmp_path / "source.csv"
    path.write_text("x,target\n" + "\n".join(f"{i},{i % 2}" for i in range(100)), encoding="utf-8")
    data = validate_csv(path, ResourceLimits())
    config = DatasetConfig(TaskType.BINARY_CLASSIFICATION, "target",
                           FeatureSchema((FeatureColumn("x", FeatureType.NUMERIC),)), positive_label="1")
    return prepare_experiment(repo, data, config, run or RunConfig())


@pytest.fixture
def setup(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    prepared = experiment(repo, tmp_path)
    return repo, prepared


def test_metadata_and_jobs_survive_reopen_without_raw_records(setup):
    repo, prepared = setup
    job = repo.enqueue_cv(prepared.experiment_id, "request-1")
    reopened = JobRepository(repo.path)
    assert reopened.get_job(job.job_id) == job
    with sqlite3.connect(repo.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("SELECT count(*) FROM datasets").fetchone()[0] == 1
        saved = json.loads(db.execute("SELECT config_json FROM configurations").fetchone()[0])
        assert saved["positive_label"] == "1"
        split = json.loads(db.execute("SELECT manifest_json FROM splits").fetchone()[0])
        assert len(split["train_row_ids"]) == 80 and len(split["holdout_row_ids"]) == 20
        assert db.execute("SELECT count(*) FROM candidates").fetchone()[0] == 0
        assert not {"rows", "targets"} & saved.keys()
        assert db.execute("SELECT event FROM audit_events ORDER BY sequence").fetchall()[-1] == ("job_enqueued",)


def test_idempotency_reuses_job_even_after_reopen_and_full_queue(setup, tmp_path):
    repo, prepared = setup
    first = repo.enqueue_cv(prepared.experiment_id, "same")
    for index in range(9):
        repo.enqueue_cv(prepared.experiment_id, f"other-{index}")
    assert JobRepository(repo.path).enqueue_cv(prepared.experiment_id, "same") == first
    with pytest.raises(QueueFullError):
        repo.enqueue_cv(prepared.experiment_id, "eleventh")
    other = experiment(repo, tmp_path, run=RunConfig(model_seed=43))
    with pytest.raises(ConflictError):
        repo.enqueue_cv(other.experiment_id, "same")


def test_payload_change_conflicts_and_frozen_metadata_cannot_be_rewritten(setup):
    repo, prepared = setup
    with sqlite3.connect(repo.path) as db:
        for table, column in (("experiments", "run_json"), ("configurations", "config_json"),
                              ("splits", "manifest_json")):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                db.execute(f"UPDATE {table} SET {column} = '{{}}'")
    assert repo.enqueue_cv(prepared.experiment_id, "still-valid").state == "queued"


def test_concurrent_duplicate_requests_create_one_job(setup):
    repo, prepared = setup
    barrier = Barrier(2)

    def enqueue(_):
        barrier.wait(timeout=5)
        return JobRepository(repo.path).enqueue_cv(prepared.experiment_id, "concurrent")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(enqueue, range(2)))
    assert results[0].job_id == results[1].job_id
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM audit_events WHERE event='job_enqueued'").fetchone()[0] == 1


@pytest.mark.parametrize("same_owner", [False, True])
def test_two_claimants_yield_one_active_attempt(setup, same_owner):
    repo, prepared = setup
    first = repo.enqueue_cv(prepared.experiment_id, "first")
    repo.enqueue_cv(prepared.experiment_id, "second")
    barrier = Barrier(2)

    def claim(index):
        owner = WorkerIdentity("owner-a" if same_owner or index == 0 else "owner-b", 1)
        barrier.wait(timeout=5)
        return JobRepository(repo.path).claim_next(owner)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))
    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1 and claimed[0].job_id == first.job_id
    assert claimed[0].state == "running" and claimed[0].attempt_id
    assert claimed[0].owner_generation == 1
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM jobs WHERE state='running'").fetchone()[0] == 1
        assert db.execute("SELECT active_job_id FROM worker_lease").fetchone()[0] == first.job_id


def test_claim_failure_rolls_back_lease_job_and_audit(setup):
    repo, prepared = setup
    job = repo.enqueue_cv(prepared.experiment_id, "rollback")
    with sqlite3.connect(repo.path) as db:
        db.execute("""CREATE TRIGGER reject_claim BEFORE INSERT ON audit_events
                      WHEN NEW.event='job_claimed' BEGIN SELECT RAISE(ABORT, 'injected'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        repo.claim_next(WorkerIdentity("owner", 1))
    assert repo.get_job(job.job_id).state == "queued"
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT owner_token, active_job_id FROM worker_lease").fetchone() == (None, None)
        db.execute("DROP TRIGGER reject_claim")
    assert repo.claim_next(WorkerIdentity("owner", 1)).job_id == job.job_id


def test_cancel_is_durable_idempotent_and_running_intent_keeps_slot(setup):
    repo, prepared = setup
    queued = repo.enqueue_cv(prepared.experiment_id, "cancel-before")
    cancelled = repo.request_cancel(queued.job_id)
    assert cancelled.state == "cancelled" and cancelled.cancel_requested
    assert repo.request_cancel(queued.job_id) == cancelled
    assert repo.claim_next(WorkerIdentity("owner", 1)) is None
    running = repo.enqueue_cv(prepared.experiment_id, "cancel-during")
    repo.claim_next(WorkerIdentity("owner", 1))
    requested = repo.request_cancel(running.job_id)
    assert requested.state == "running" and requested.cancel_requested
    repo.enqueue_cv(prepared.experiment_id, "waiting")
    assert repo.claim_next(WorkerIdentity("owner", 1)) is None
    assert JobRepository(repo.path).get_job(running.job_id) == requested


def test_busy_wait_is_bounded_and_does_not_leave_orphan(setup):
    repo, prepared = setup
    bounded = JobRepository(repo.path, busy_timeout_ms=25, begin_retries=1)
    with sqlite3.connect(repo.path) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        started = perf_counter()
        with pytest.raises(StorageBusyError):
            bounded.enqueue_cv(prepared.experiment_id, "busy")
        assert perf_counter() - started < 1.5
        blocker.rollback()
    assert bounded.enqueue_cv(prepared.experiment_id, "busy").state == "queued"
    assert bounded.claim_next(WorkerIdentity("owner", 1)).state == "running"


def test_unknown_ids_injection_and_empty_keys_are_safe(setup):
    repo, prepared = setup
    attack = "'); DROP TABLE jobs; --"
    job = repo.enqueue_cv(prepared.experiment_id, attack)
    assert repo.enqueue_cv(prepared.experiment_id, attack) == job
    for action in (repo.enqueue_cv,):
        with pytest.raises(NotFoundError):
            action("unknown", "new")
    for action in (repo.get_job, repo.request_cancel):
        with pytest.raises(NotFoundError):
            action(attack)
    for key in ("", " ", "x" * 201):
        with pytest.raises(ValueError):
            repo.enqueue_cv(prepared.experiment_id, key)


def test_newer_schema_is_rejected_without_modification(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="schema"):
        JobRepository(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 999


def test_scope_rejects_new_split_seed_without_creating_partial_metadata(setup, tmp_path):
    repo, prepared = setup
    with pytest.raises(ConflictError):
        experiment(repo, tmp_path, run=RunConfig(split_seed=43))
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM experiments").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM splits").fetchone()[0] == 1


def test_stale_generation_cannot_claim(setup):
    repo, prepared = setup
    job = repo.enqueue_cv(prepared.experiment_id, "queued")
    assert repo.claim_next(WorkerIdentity("stale", 2)) is None
    assert repo.get_job(job.job_id).state == "queued"
    assert repo.claim_next(WorkerIdentity("current", 1)).job_id == job.job_id


def test_lower_frozen_queue_limit_is_enforced(setup, tmp_path):
    repo, _ = setup
    prepared = experiment(repo, tmp_path, run=RunConfig(limits=ResourceLimits(queue_size=1)))
    repo.enqueue_cv(prepared.experiment_id, "one")
    with pytest.raises(QueueFullError):
        repo.enqueue_cv(prepared.experiment_id, "two")


def test_cancel_and_claim_race_never_runs_a_cancelled_job(setup):
    repo, prepared = setup
    job = repo.enqueue_cv(prepared.experiment_id, "race")
    barrier = Barrier(2)

    def cancel():
        barrier.wait(timeout=5)
        return repo.request_cancel(job.job_id)

    def claim():
        barrier.wait(timeout=5)
        return repo.claim_next(WorkerIdentity("owner", 1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        cancellation, claiming = pool.submit(cancel), pool.submit(claim)
        cancellation.result()
        claimed = claiming.result()
    final = repo.get_job(job.job_id)
    assert final.cancel_requested
    if claimed is None:
        assert final.state == "cancelled" and final.attempt_id is None
    else:
        assert final.state == "running" and final.attempt_id == claimed.attempt_id


def test_enqueue_failure_rolls_back_job_and_preserves_key(setup):
    repo, prepared = setup
    with sqlite3.connect(repo.path) as db:
        db.execute("""CREATE TRIGGER reject_enqueue BEFORE INSERT ON audit_events
                      WHEN NEW.event='job_enqueued' BEGIN SELECT RAISE(ABORT, 'injected'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        repo.enqueue_cv(prepared.experiment_id, "retry-key")
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        db.execute("DROP TRIGGER reject_enqueue")
    assert repo.enqueue_cv(prepared.experiment_id, "retry-key").state == "queued"


def test_queue_capacity_is_atomic_under_concurrent_requests(setup, tmp_path):
    repo, _ = setup
    prepared = experiment(repo, tmp_path, run=RunConfig(limits=ResourceLimits(queue_size=1)))
    barrier = Barrier(2)

    def enqueue(index):
        barrier.wait(timeout=5)
        try:
            return repo.enqueue_cv(prepared.experiment_id, f"bounded-{index}")
        except QueueFullError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(enqueue, range(2)))
    assert len([result for result in results if result is not None]) == 1


def test_experiment_registration_reuses_identity_and_audit_is_append_only(setup, tmp_path):
    repo, prepared = setup
    assert experiment(repo, tmp_path).experiment_id == prepared.experiment_id
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM audit_events").fetchone()[0] == 1
        for sql in ("UPDATE audit_events SET actor='replacement'", "DELETE FROM audit_events"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute(sql)


@pytest.mark.parametrize("state", ["completed", "failed", "interrupted"])
def test_terminal_job_cancel_is_noop(setup, state):
    repo, prepared = setup
    job = repo.enqueue_cv(prepared.experiment_id, "terminal")
    # Later worker milestones own terminal transitions; simulate their persisted result.
    with sqlite3.connect(repo.path) as db:
        db.execute("UPDATE jobs SET state=? WHERE job_id=?", (state, job.job_id))
    before = repo.get_job(job.job_id)
    assert repo.request_cancel(job.job_id) == before


@pytest.mark.parametrize("kwargs", [
    {"busy_timeout_ms": 0}, {"busy_timeout_ms": 5001}, {"begin_retries": 3},
    {"begin_retries": -1}, {"begin_retries": True},
])
def test_unbounded_or_invalid_wait_policies_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        JobRepository(tmp_path / "invalid.sqlite3", **kwargs)


@pytest.mark.parametrize("token,generation", [("", 1), ("owner", 0), ("owner", True)])
def test_invalid_worker_identities_are_rejected(token, generation):
    with pytest.raises(ValueError):
        WorkerIdentity(token, generation)
