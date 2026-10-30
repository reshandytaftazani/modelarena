"""Real Windows-spawn cancellation and fail-closed restart recovery."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import sqlite3
from time import perf_counter, sleep

import psutil
import pytest

from apps.api.database.repository import ConflictError, JobRepository, WorkerIdentity
from apps.api.workers.supervisor import WorkerSupervisor
from apps.api.workers import monitor
from test_worker_spawn import _queued_job, _supervisor


def wait_for(predicate, seconds=15):
    deadline = perf_counter() + seconds
    while perf_counter() < deadline:
        if predicate():
            return
        sleep(0.02)
    pytest.fail("Timed out waiting for worker state")


def test_queued_cancel_never_spawns(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    assert repo.request_cancel(job.job_id).state == "cancelled"
    worker = _supervisor(repo)
    assert worker.run_once() is None
    assert worker.last_child_pid is None


@pytest.mark.parametrize("cancel_first", [True, False])
def test_cancel_and_completion_have_one_transactional_winner(tmp_path, cancel_first):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    owner = WorkerIdentity("owner", 1)
    claimed = repo.claim_next(owner)
    if cancel_first:
        repo.request_cancel(job.job_id)
    final = repo.finish_job(owner, job.job_id, claimed.attempt_id,
                            state="completed", stage="completed")
    if not cancel_first:
        repo.request_cancel(job.job_id)
    assert final.state == ("cancelled" if cancel_first else "completed")
    assert repo.get_job(job.job_id).state == final.state
    with pytest.raises(ConflictError):
        repo.finish_job(owner, job.job_id, claimed.attempt_id,
                        state="completed", stage="completed")
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM candidates").fetchone()[0] == 0


@pytest.mark.parametrize("options,maximum", [
    ({"work_seconds": 60, "steps": 100}, 4),
    ({"work_seconds": 0, "silent_seconds": 60}, 10),
])
def test_running_cancel_cooperates_or_stops_native_block_within_deadline(tmp_path, options, maximum):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    worker = _supervisor(repo, options=options)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_once)
        wait_for(lambda: repo.get_job(job.job_id).stage == "fitting")
        started = perf_counter()
        repo.request_cancel(job.job_id)
        final = future.result(timeout=12)
    assert perf_counter() - started < maximum
    assert final.state == "cancelled"
    assert worker.last_result is None
    assert not psutil.pid_exists(worker.last_child_pid)
    if options["work_seconds"]:
        assert worker.child_exit_code == 0, "cooperative Event exit must not need termination"


def run_supervisor(path):
    worker = _supervisor(JobRepository(path), options={"silent_seconds": 60})
    worker.run_once()


def test_killed_supervisor_blocks_restart_until_verified_child_is_dead(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path, "first")
    waiting = _queued_job(repo, tmp_path, "waiting")
    parent = multiprocessing.get_context("spawn").Process(target=run_supervisor, args=(repo.path,))
    child = None
    try:
        parent.start()
        wait_for(lambda: repo.get_job(job.job_id).stage == "fitting")
        child = psutil.Process(parent.pid).children()[0]
        with sqlite3.connect(repo.path) as db:
            pid, created = db.execute("SELECT child_pid,child_created FROM worker_processes").fetchone()
        child = psutil.Process(pid)
        assert child.create_time() == created
        parent.terminate()
        parent.join(5)
        restarted = _supervisor(repo, owner_token="replacement")
        assert restarted.run_once() is None
        assert restarted.last_child_pid is None
        assert repo.get_job(waiting.job_id).state == "queued"
        child.terminate()
        child.wait(5)
        assert restarted.run_once().job_id == waiting.job_id
        assert repo.get_job(job.job_id).state == "interrupted"
        assert repo.lease_generation() == 2
    finally:
        if parent.is_alive():
            parent.terminate()
        parent.join(5)
        if child is not None and child.is_running():
            child.kill()
            child.wait(5)


def test_live_supervisor_is_not_replaced_even_after_child_death(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    worker = _supervisor(repo, options={"silent_seconds": 60})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_once)
        wait_for(lambda: repo.get_job(job.job_id).stage == "fitting")
        assert JobRepository(repo.path).get_job(job.job_id).state == "running"
        assert _supervisor(JobRepository(repo.path), owner_token="api-restart").run_once() is None
        psutil.Process(worker.last_child_pid).terminate()
        final = future.result(timeout=10)
    assert final.state == "interrupted"


def test_unknown_ownership_never_releases_slot(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    claimed = repo.claim_next(WorkerIdentity("legacy", 1))
    with sqlite3.connect(repo.path) as db:
        db.execute("UPDATE jobs SET heartbeat_at='2000-01-01' WHERE job_id=?", (job.job_id,))
    assert repo.reconcile_worker() is False
    assert repo.get_job(claimed.job_id).state == "running"
    assert repo.lease_generation() == 1


def test_process_identity_mismatch_is_dead_but_access_denial_is_unknown(monkeypatch):
    pid = os.getpid()
    actual = psutil.Process(pid).create_time()
    assert monitor.process_liveness(pid, actual) == "alive"
    assert monitor.process_liveness(pid, actual + 0.01) == "dead"

    def denied(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    assert monitor.process_liveness(pid, actual) == "unknown"


def test_interrupted_retry_keeps_frozen_experiment_and_fences_old_attempt(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    owner = WorkerIdentity("old", 1)
    claimed = repo.claim_next(owner)
    repo.finish_job(owner, job.job_id, claimed.attempt_id,
                    state="interrupted", stage="interrupted")
    retried = repo.retry_cv(job.job_id, "explicit-retry")
    assert retried.experiment_id == job.experiment_id
    assert retried.job_id != job.job_id
    assert retried.state == "queued"
    assert repo.retry_cv(job.job_id, "explicit-retry").job_id == retried.job_id
    next_attempt = repo.claim_next(WorkerIdentity("new", repo.lease_generation()))
    assert next_attempt.attempt_id != claimed.attempt_id
    with pytest.raises(ConflictError):
        repo.finish_job(owner, job.job_id, claimed.attempt_id,
                        state="completed", stage="completed")


def test_successful_or_running_cv_cannot_be_retried(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    with pytest.raises(ConflictError):
        repo.retry_cv(job.job_id, "retry")
    _supervisor(repo).run_once()
    with pytest.raises(ConflictError):
        repo.retry_cv(job.job_id, "retry")


def test_keyboard_interrupt_joins_child_and_preserves_waiting_queue(tmp_path, monkeypatch):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path, "first")
    waiting = _queued_job(repo, tmp_path, "waiting")
    worker = _supervisor(repo, options={"silent_seconds": 60})

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "_monitor", interrupt)
    final = worker.run_once()
    assert final.state == "interrupted"
    assert repo.get_job(waiting.job_id).state == "queued"
    assert not psutil.pid_exists(worker.last_child_pid)


def test_generation_fence_rejects_old_owner_even_if_token_is_reused(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    owner = WorkerIdentity("owner", 1)
    repo.claim_next(owner)
    with sqlite3.connect(repo.path) as db:
        db.execute("UPDATE worker_lease SET generation=2")
    with pytest.raises(ConflictError):
        repo.record_progress(owner, job.job_id, "fitting")


def test_version_one_metadata_migrates_without_losing_queued_jobs(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    with sqlite3.connect(repo.path) as db:
        db.execute("DROP TABLE worker_processes")
        db.execute("PRAGMA user_version=1")
    migrated = JobRepository(repo.path)
    assert migrated.get_job(job.job_id).state == "queued"
    assert _supervisor(migrated).run_once().state == "completed"
    with sqlite3.connect(repo.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3


def native_fit_child(job_id, attempt_id, limits, workload, options, messages, cancel_event, start_event):
    """Test-owned spawn target: genuinely enter sklearn native fitting, no cancel checks."""
    from apps.api.workers.execute import apply_thread_environment
    apply_thread_environment(limits["compute_threads"])
    assert start_event.wait(10)
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(42)
    x = rng.normal(size=(5000, 10))
    y = x[:, 0] + rng.normal(size=5000)
    with threadpool_limits(limits=1):
        # One real fit first proves native fitting was entered before announcing.
        HistGradientBoostingRegressor(max_iter=1, early_stopping=False).fit(x, y)
        messages.put({"kind": "progress", "stage": "fitting"})
        HistGradientBoostingRegressor(max_iter=100000, early_stopping=False).fit(x, y)


@pytest.mark.parametrize("heartbeat", [0.2, 60.0])
def test_cancel_during_real_native_fit_terminates_within_ten_seconds(tmp_path, monkeypatch, heartbeat):
    from apps.api.workers import execute

    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    worker = _supervisor(repo, heartbeat_seconds=heartbeat,
                         poll_seconds=5 if heartbeat == 60 else 0.05)
    monkeypatch.setattr(execute, "child_main", native_fit_child)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_once)
        wait_for(lambda: repo.get_job(job.job_id).stage == "fitting")
        started = perf_counter()
        repo.request_cancel(job.job_id)
        try:
            final = future.result(timeout=12)
        finally:
            if not future.done() and psutil.pid_exists(worker.last_child_pid):
                psutil.Process(worker.last_child_pid).terminate()
    assert perf_counter() - started < 10
    assert final.state == "cancelled"
    assert worker.last_result is None
    assert not psutil.pid_exists(worker.last_child_pid)


def test_cancelled_attempt_cannot_leave_rankable_candidate_rows(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    owner = WorkerIdentity("owner", 1)
    claimed = repo.claim_next(owner)
    with sqlite3.connect(repo.path) as db:
        db.execute("INSERT INTO candidates(candidate_id,experiment_id,job_id,attempt_id,family,status,config_sha256) "
                   "VALUES('partial',?,?,?,'dummy','completed','digest')",
                   (job.experiment_id, job.job_id, claimed.attempt_id))
    repo.request_cancel(job.job_id)
    repo.finish_job(owner, job.job_id, claimed.attempt_id, state="completed", stage="completed")
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT status FROM candidates").fetchone()[0] == "cancelled"


def test_live_child_prevents_terminal_publication_and_retry(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    worker = _supervisor(repo, options={"silent_seconds": 60})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_once)
        wait_for(lambda: repo.get_job(job.job_id).stage == "fitting")
        attempt = repo.get_job(job.job_id).attempt_id
        try:
            with pytest.raises(ConflictError):
                repo.finish_job(worker.owner, job.job_id, attempt, state="completed", stage="completed")
            with pytest.raises(ConflictError):
                repo.retry_cv(job.job_id, "retry")
        finally:
            repo.request_cancel(job.job_id)
        assert future.result(timeout=12).state == "cancelled"


def test_spawn_registration_crash_gap_fails_closed(tmp_path):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    # A mismatched parent creation time proves the original parent is gone;
    # absent child identity still forbids recovery from guessing no spawn occurred.
    claimed = repo.claim_next(WorkerIdentity("old", 1), supervisor_process=(os.getpid(), 1.0))
    assert repo.reconcile_worker() is False
    assert repo.get_job(job.job_id).attempt_id == claimed.attempt_id
    assert _supervisor(repo).run_once() is None


def test_cancel_arriving_after_result_before_commit_discards_result(tmp_path, monkeypatch):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    worker = _supervisor(repo)
    publish = worker._publish

    def cancel_then_publish(job, attempt, outcome):
        repo.request_cancel(job.job_id)
        return publish(job, attempt, outcome)

    monkeypatch.setattr(worker, "_publish", cancel_then_publish)
    assert worker.run_once().state == "cancelled"
    assert worker.last_result is None


def test_failed_child_identity_observation_never_releases_a_live_child(tmp_path, monkeypatch):
    repo = JobRepository(tmp_path / "jobs.sqlite3")
    job = _queued_job(repo, tmp_path)
    worker = _supervisor(repo, options={"silent_seconds": 60})
    real_creation_time = monitor.process_creation_time

    def cannot_observe_child(pid):
        if pid != os.getpid():
            raise RuntimeError("Not observable")
        return real_creation_time(pid)

    monkeypatch.setattr(monitor, "process_creation_time", cannot_observe_child)
    try:
        final = worker.run_once()
        assert final.state == "running"
        assert _supervisor(repo, owner_token="other").run_once() is None
    finally:
        if worker.last_child_pid and psutil.pid_exists(worker.last_child_pid):
            child = psutil.Process(worker.last_child_pid)
            child.kill()
            child.wait(5)
