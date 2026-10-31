"""Fast repository-level regressions for the Day 14 review."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from unittest.mock import patch
import os

import psutil

import pytest

from apps.api.database.repository import ConflictError, WorkerIdentity
from apps.api.services.finalization import compute_candidate_digest, reserve, start_evaluation, fixture_evaluate_holdout
from ml.contracts import HoldoutCapability, TaskType
from tests.integration.test_classical_suite import prepared_fixture
from apps.api.services.jobs import prepare_experiment
from ml.data.csv_io import validate_csv


@pytest.fixture
def reservation(tmp_path):
    repo, prepared, run, cv_job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    with repo._transaction() as db:
        config = db.execute("SELECT c.scope_id, s.manifest_sha256 FROM experiments e "
                            "JOIN configurations c ON c.configuration_id=e.configuration_id "
                            "JOIN splits s ON s.split_id=c.split_id WHERE e.experiment_id=?",
                            (prepared.experiment_id,)).fetchone()
        db.execute("INSERT INTO candidates(candidate_id,experiment_id,job_id,attempt_id,family,status,config_sha256) "
                   "VALUES(?,?,?,?,?,'completed',?)", ("candidate", prepared.experiment_id, cv_job.job_id,
                   "cv-attempt", "dummy", "config-hash"))
    digest = compute_candidate_digest("candidate", "dummy", config["manifest_sha256"], "config-hash", "rmse", 1.0, 42)
    bundle = {"candidates": [{"family": "dummy", "eligible": True, "model_seed": 42,
                              "summary": [{"name": "rmse", "mean": 1.0}]}]}
    with patch.object(repo, "read_published_results", return_value=bundle):
        selection, job = reserve(repo, prepared.experiment_id, "candidate", digest, True)
        yield repo, prepared, run, selection, job, bundle


def test_retry_rejects_queued_and_running(reservation):
    repo, _, _, selection, job, _ = reservation
    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "queued-retry")
    # Remove the older CV job from the queue before claiming finalization.
    with repo._transaction() as db:
        db.execute("UPDATE jobs SET state='cancelled' WHERE kind='cv' AND state='queued'")
    assert repo.claim_next(WorkerIdentity("owner", 1), kind="finalization").job_id == job.job_id
    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "running-retry")


def test_concurrent_retries_after_queued_cancellation(reservation):
    repo, _, _, selection, job, _ = reservation
    repo.request_cancel(job.job_id)
    barrier = Barrier(2)
    def attempt(key):
        barrier.wait()
        try:
            return repo.retry_pre_evaluation(selection.finalization_id, key).job_id
        except ConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ("retry-a", "retry-b")))
    assert len([r for r in results if r is not None]) == 1
    winning_key = "retry-a" if results[0] else "retry-b"
    assert repo.retry_pre_evaluation(selection.finalization_id, winning_key).job_id in results


def test_candidate_from_different_experiment_cannot_reserve(reservation, tmp_path):
    repo, prepared, run, _, _, _ = reservation
    # A separate experiment shares the same scope but cannot select this candidate.
    other = prepare_experiment(repo, validate_csv(tmp_path / "input.csv", run.limits),
                               prepared.context.training.config, replace(run, model_seed=43))
    with pytest.raises(ConflictError):
        reserve(repo, other.experiment_id, "candidate", reservation[3].candidate_digest, True)


def test_cross_experiment_candidate_rejected_before_any_scope_lock(tmp_path):
    repo, prepared, run, cv_job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    other = prepare_experiment(repo, validate_csv(tmp_path / "input.csv", run.limits),
                               prepared.context.training.config, replace(run, model_seed=43))
    with repo._transaction() as db:
        db.execute("INSERT INTO candidates(candidate_id,experiment_id,job_id,attempt_id,family,status,config_sha256) "
                   "VALUES(?,?,?,?,?,'completed',?)", ("candidate", prepared.experiment_id, cv_job.job_id,
                   "cv-attempt", "dummy", "config-hash"))
    with pytest.raises(ConflictError):
        reserve(repo, other.experiment_id, "candidate", "a" * 64, True)
    with repo._connection() as db:
        assert db.execute("SELECT count(*) FROM finalizations").fetchone()[0] == 0


@pytest.mark.parametrize("liveness,allowed", [("alive", False), ("unknown", False), ("dead", True)])
def test_retry_requires_verified_child_death(reservation, liveness, allowed):
    repo, _, _, selection, job, _ = reservation
    with repo._transaction() as db:
        db.execute("UPDATE jobs SET state='failed',attempt_id='prior',owner_token='owner',owner_generation=1 WHERE job_id=?", (job.job_id,))
        db.execute("INSERT INTO worker_processes(attempt_id,supervisor_pid,supervisor_created,child_pid,child_created) VALUES('prior',1,1,2,1)")
        db.execute("UPDATE finalizations SET state='retryable_pre_evaluation' WHERE finalization_id=?", (selection.finalization_id,))
    with patch("apps.api.database.repository.process_liveness", return_value=liveness):
        if allowed:
            assert repo.retry_pre_evaluation(selection.finalization_id, "retry").job_id != job.job_id
        else:
            with pytest.raises(ConflictError):
                repo.retry_pre_evaluation(selection.finalization_id, "retry")


def test_queued_job_cannot_authorize_evaluation(reservation):
    repo, _, _, selection, _, _ = reservation
    with pytest.raises(ConflictError):
        start_evaluation(repo, selection.finalization_id, selection.candidate_digest)


def test_forged_capability_cannot_consume_evaluating_scope(reservation):
    repo, _, _, selection, _, _ = reservation
    with repo._transaction() as db:
        db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now' WHERE finalization_id=?",
                   (selection.finalization_id,))
    with pytest.raises((ConflictError, PermissionError)):
        forged = HoldoutCapability(scope_id=selection.scope_id, holdout_path="fake", _auth_token="a" * 64)
        fixture_evaluate_holdout(repo, forged)


def claim_finalization(repo, job):
    with repo._transaction() as db:
        db.execute("UPDATE jobs SET state='cancelled' WHERE kind='cv' AND state='queued'")
    owner = WorkerIdentity("fixture-owner", repo.lease_generation())
    pid = os.getpid()
    created = psutil.Process(pid).create_time()
    claimed = repo.claim_next(owner, kind="finalization", supervisor_process=(pid, created))
    assert claimed.job_id == job.job_id
    repo.record_child(owner, claimed.job_id, claimed.attempt_id, pid, created)
    return owner, claimed


def test_evaluation_is_bound_and_single_use(reservation):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    with pytest.raises(ConflictError):
        start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                         owner=owner, job_id=job.job_id, attempt_id="stale-attempt")
    with pytest.raises(ConflictError):
        start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                         owner=WorkerIdentity("stale", owner.generation), job_id=job.job_id, attempt_id=claimed.attempt_id)
    capability = start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                                  owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)
    values = {name: getattr(capability, name) for name in capability.__dataclass_fields__}
    for field, value in (("scope_id", "wrong"), ("attempt_id", "wrong"),
                         ("candidate_digest", "wrong"), ("_auth_token", "a" * 64)):
        with pytest.raises(ConflictError):
            fixture_evaluate_holdout(repo, HoldoutCapability._issue(**(values | {field: value})))
    assert fixture_evaluate_holdout(repo, capability).scope_id == selection.scope_id
    with pytest.raises(ConflictError):
        fixture_evaluate_holdout(repo, capability)
    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "after-start")


def test_concurrent_evaluation_start_has_one_winner(reservation):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    barrier = Barrier(2)
    def attempt(_):
        barrier.wait()
        try:
            return start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                                    owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)
        except ConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(result is not None for result in results) == 1


def test_capability_rejects_owner_fenced_after_start(reservation):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    capability = start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                                  owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)
    with repo._transaction() as db:
        db.execute("UPDATE worker_lease SET generation=generation+1 WHERE singleton=1")
    with pytest.raises(ConflictError):
        fixture_evaluate_holdout(repo, capability)


def test_failed_attempt_without_child_identity_is_not_retryable(reservation):
    repo, _, _, selection, job, _ = reservation
    with repo._transaction() as db:
        db.execute("UPDATE jobs SET state='failed',attempt_id='missing-child',owner_token='owner',owner_generation=1 WHERE job_id=?", (job.job_id,))
        db.execute("UPDATE finalizations SET state='retryable_pre_evaluation' WHERE finalization_id=?", (selection.finalization_id,))
    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "retry")


def test_same_retry_key_is_concurrently_idempotent(reservation):
    repo, _, _, selection, job, _ = reservation
    repo.request_cancel(job.job_id)
    barrier = Barrier(2)
    def attempt(_):
        barrier.wait()
        return repo.retry_pre_evaluation(selection.finalization_id, "same-key").job_id
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(attempt, range(2)))
    assert ids[0] == ids[1] != job.job_id


def test_changed_frozen_candidate_is_rejected_before_retry(reservation):
    repo, _, _, selection, job, _ = reservation
    repo.request_cancel(job.job_id)
    with repo._transaction() as db:
        db.execute("UPDATE candidates SET config_sha256='changed' WHERE candidate_id=?", (selection.candidate_id,))
    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "retry")


def test_changed_frozen_candidate_is_rejected_before_evaluation_start(reservation):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    with repo._transaction() as db:
        db.execute("UPDATE candidates SET config_sha256='changed' WHERE candidate_id=?", (selection.candidate_id,))
    with pytest.raises(ConflictError):
        start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                         owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)


def test_version_five_store_gets_authorization_table(tmp_path):
    from apps.api.database.repository import JobRepository
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    with repo._transaction() as db:
        db.execute("DROP TABLE evaluation_authorizations")
        db.execute("PRAGMA user_version=5")
    reopened = JobRepository(repo.path)
    with reopened._connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 6
        assert db.execute("SELECT count(*) FROM evaluation_authorizations").fetchone()[0] == 0
