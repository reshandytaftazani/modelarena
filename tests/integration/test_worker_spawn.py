"""Real spawned execution: one owned child, durable progress and singleton slot.

The child runs the internal ``probe`` workload because staged-dataset CV
execution arrives with the classical registry (Days 10-12). These tests still
spawn a genuine Windows child process, so process identity, heartbeat, progress
persistence and the single-slot claim are exercised for real. No holdout data,
trained model or generated artifact is involved.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3
import subprocess
import sys
from time import perf_counter, sleep

import pytest

from apps.api.database.repository import ConflictError, JobRepository, WorkerIdentity
from apps.api.services.jobs import prepare_experiment
from apps.api.workers.supervisor import WorkerSupervisor
from ml.contracts import (
    DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, RunConfig,
    TaskType,
)
from ml.data.csv_io import validate_csv


def _queued_job(repo: JobRepository, tmp_path, key: str = "job"):
    path = tmp_path / f"{key}.csv"
    path.write_text("x,target\n" + "\n".join(f"{i},{i % 2}" for i in range(100)), encoding="utf-8")
    data = validate_csv(path, ResourceLimits())
    config = DatasetConfig(
        TaskType.BINARY_CLASSIFICATION, "target",
        FeatureSchema((FeatureColumn("x", FeatureType.NUMERIC),)), positive_label="1",
    )
    prepared = prepare_experiment(repo, data, config, RunConfig())
    return repo.enqueue_cv(prepared.experiment_id, key)


def _supervisor(repo: JobRepository, *, options=None, **kwargs) -> WorkerSupervisor:
    kwargs.setdefault("limits", ResourceLimits())
    kwargs.setdefault("workload", "probe")
    kwargs.setdefault("workload_options", options if options is not None else {"work_seconds": 0.2})
    kwargs.setdefault("heartbeat_seconds", 0.2)
    kwargs.setdefault("sample_seconds", 0.1)
    kwargs.setdefault("poll_seconds", 0.05)
    kwargs.setdefault("owner_token", "supervisor-a")
    return WorkerSupervisor(repo, **kwargs)


def test_spawned_child_completes_job_and_releases_the_slot(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(repo, options={"work_seconds": 0.1, "steps": 3})

    final = supervisor.run_once()

    assert final is not None and final.job_id == job.job_id
    assert final.state == "completed" and final.stage == "completed"
    assert final.owner_token == "supervisor-a" and final.owner_generation == 1
    assert final.safe_error is None
    # The work really happened in another process, not in this one.
    assert supervisor.last_child_pid is not None
    assert supervisor.last_child_pid != os.getpid()
    assert supervisor.last_result["pid"] == supervisor.last_child_pid
    assert supervisor.last_result["parent_pid"] == os.getpid()
    assert supervisor.last_result["steps"] == 3
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT active_job_id FROM worker_lease").fetchone()[0] is None
        assert db.execute("SELECT count(*) FROM jobs WHERE state='running'").fetchone()[0] == 0
        events = [row[0] for row in db.execute("SELECT event FROM audit_events ORDER BY sequence")]
    assert "job_claimed" in events and "job_completed" in events
    assert any(event.startswith("job_telemetry:") for event in events)


def test_empty_queue_does_not_spawn_a_child(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    supervisor = _supervisor(repo)
    assert supervisor.run_once() is None
    assert supervisor.last_child_pid is None


def test_progress_is_persisted_while_the_child_is_still_running(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(repo, options={"work_seconds": 0.6, "steps": 6})

    running_stages = set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(supervisor.run_once)
        while not future.done():
            record = repo.get_job(job.job_id)
            if record.state == "running":
                running_stages.add(record.stage)
            sleep(0.02)
        final = future.result()

    assert final.state == "completed"
    # Progress must be durable *during* execution, not only at the end.
    assert {"starting", "preparing", "fitting"} & running_stages
    assert final.stage == "completed"


def test_heartbeat_continues_while_the_child_is_silent(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(
        repo, options={"work_seconds": 0.0, "silent_seconds": 1.5}, heartbeat_seconds=0.2,
    )

    beats = set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(supervisor.run_once)
        while not future.done():
            beats.add(repo.get_job(job.job_id).heartbeat_at)
            sleep(0.05)
        final = future.result()

    beats.discard(None)
    assert len(beats) >= 3, "heartbeat must not depend on child progress messages"
    assert final.state == "completed"


def test_second_supervisor_cannot_start_a_concurrent_fit(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    _queued_job(repo, tmp_path, "one")
    _queued_job(repo, tmp_path, "two")
    first = _supervisor(repo, owner_token="supervisor-a", options={"work_seconds": 0.8})
    second = _supervisor(repo, owner_token="supervisor-b", options={"work_seconds": 0.8})

    max_running = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(first.run_once), pool.submit(second.run_once)]
        while any(not future.done() for future in futures):
            with sqlite3.connect(repo.path) as db:
                running = db.execute("SELECT count(*) FROM jobs WHERE state='running'").fetchone()[0]
            max_running = max(max_running, running)
            sleep(0.02)
        results = [future.result() for future in futures]

    assert max_running == 1
    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1, "the second supervisor must observe the held slot, not claim it"
    assert claimed[0].state == "completed"
    # The queued job is still available to whichever supervisor runs next.
    assert repo.get_job(_queued_job(repo, tmp_path, "three").job_id).state == "queued"
    assert first.run_once().state == "completed"


def test_host_process_stays_responsive_and_never_fits(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(repo, options={"work_seconds": 1.2})

    latencies = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(supervisor.run_once)
        while not future.done():
            started = perf_counter()
            repo.get_job(job.job_id)
            latencies.append(perf_counter() - started)
            sleep(0.02)
        final = future.result()

    assert final.state == "completed"
    assert len(latencies) >= 5
    assert sum(latencies) / len(latencies) < 0.1
    assert max(latencies) < 0.5
    assert supervisor.last_child_pid != os.getpid()


def test_stale_worker_identity_cannot_write_worker_state(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    claimed = repo.claim_next(WorkerIdentity("owner", 1))
    assert claimed is not None and claimed.job_id == job.job_id
    intruder = WorkerIdentity("intruder", 1)

    with pytest.raises(ConflictError):
        repo.heartbeat(intruder, claimed.job_id)
    with pytest.raises(ConflictError):
        repo.record_progress(intruder, claimed.job_id, "fitting")
    with pytest.raises(ConflictError):
        repo.finish_job(intruder, claimed.job_id, claimed.attempt_id,
                        state="completed", stage="completed")
    with pytest.raises(ConflictError):
        repo.finish_job(WorkerIdentity("owner", 1), claimed.job_id, "other-attempt",
                        state="completed", stage="completed")
    assert repo.get_job(job.job_id).state == "running"


def test_worker_state_rejects_unsafe_stage_and_state_values(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    claimed = repo.claim_next(WorkerIdentity("owner", 1))
    assert claimed is not None and claimed.job_id == job.job_id
    owner = WorkerIdentity("owner", 1)
    for stage in ("", "Fitting", "row 1 is 3.5", "x" * 65, "a b"):
        with pytest.raises(ValueError):
            repo.record_progress(owner, job.job_id, stage)
    for state in ("running", "queued", "done", ""):
        with pytest.raises(ValueError):
            repo.finish_job(owner, job.job_id, claimed.attempt_id, state=state, stage="completed")


def test_running_cancel_is_honoured_instead_of_publishing_completed(tmp_path):
    """A persisted cancel request must not end as a silent 'completed' result.

    Day 9 owns cooperative cancellation, grace periods and the cancel/result
    race in full. Day 8 must still never publish success for work the user has
    already asked to abandon.
    """
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(repo, options={"work_seconds": 20.0, "steps": 2})

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(supervisor.run_once)
        while repo.get_job(job.job_id).state != "running":
            sleep(0.02)
        repo.request_cancel(job.job_id)
        final = future.result()
    elapsed = perf_counter() - started

    assert final.state == "cancelled"
    assert final.stage == "cancelled"
    assert final.safe_error == "cancelled"
    assert final.cancel_requested
    assert elapsed < 15, "a cancelled child must not run to completion"
    assert supervisor.last_result is None
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT active_job_id FROM worker_lease").fetchone()[0] is None
        events = [row[0] for row in db.execute("SELECT event FROM audit_events ORDER BY sequence")]
    assert "job_cancelled" in events
    assert "job_completed" not in events


def test_unrenderable_telemetry_does_not_strand_a_running_job(tmp_path, monkeypatch):
    """Telemetry is diagnostics; a bad sample must not block the queue forever.

    If rendering raised out of publication, the job would stay ``running`` and
    hold the singleton slot, so every later claim would fail.
    """
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(repo, options={"work_seconds": 0.0, "report_pools": False})
    monkeypatch.setattr(
        supervisor, "_telemetry", lambda: {"peak_rss_bytes": "not-a-number"},
    )

    final = supervisor.run_once()

    assert final.state == "completed"
    assert final.stage == "completed"
    assert repo.get_job(job.job_id).state == "completed"
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT active_job_id FROM worker_lease").fetchone()[0] is None
        events = [row[0] for row in db.execute("SELECT event FROM audit_events ORDER BY sequence")]
    assert "job_completed" in events
    assert not any(event.startswith("job_telemetry:") for event in events)


def test_execution_module_is_import_safe_for_spawn(tmp_path):
    """The child entrypoint must not import numeric libraries at module scope."""
    code = (
        "import sys\n"
        "import psutil\n"
        "import apps.api.workers.execute as execute\n"
        "assert callable(execute.child_main)\n"
        "for banned in ('numpy', 'torch', 'sklearn', 'pandas'):\n"
        "    assert banned not in sys.modules, f'{banned} imported at module scope'\n"
        "assert psutil.Process().children() == [], 'importing spawned a child'\n"
        "print('import-safe')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=os.getcwd(), timeout=120)
    assert result.returncode == 0, result.stderr
    assert "import-safe" in result.stdout
