"""Child resource telemetry, thread-pool scoping and safe failure persistence.

These tests spawn real Windows children. They never load user data, so "no raw
rows" is proven by injecting a secret string into a deliberate workload failure
and asserting the persisted record cannot echo it.
"""

import json
import logging
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from apps.api.database.repository import JobRepository
from apps.api.services.jobs import prepare_experiment
from apps.api.workers import monitor
from apps.api.workers.supervisor import WorkerSupervisor
from ml.contracts import (
    DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, RunConfig,
    TaskType,
)
from ml.data.csv_io import validate_csv


def _queued_job(repo, tmp_path, key="job"):
    path = tmp_path / f"{key}.csv"
    path.write_text("x,target\n" + "\n".join(f"{i},{i % 2}" for i in range(100)), encoding="utf-8")
    data = validate_csv(path, ResourceLimits())
    config = DatasetConfig(
        TaskType.BINARY_CLASSIFICATION, "target",
        FeatureSchema((FeatureColumn("x", FeatureType.NUMERIC),)), positive_label="1",
    )
    prepared = prepare_experiment(repo, data, config, RunConfig())
    return repo.enqueue_cv(prepared.experiment_id, key)


def _supervisor(repo, *, limits=None, options=None):
    return WorkerSupervisor(
        repo, limits=limits or ResourceLimits(), workload="probe",
        workload_options=options or {"work_seconds": 0.2},
        heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05,
        owner_token="supervisor-a",
    )


def test_production_soft_cap_stays_at_four_gibibytes():
    limits = ResourceLimits()
    assert limits.child_rss_bytes == 4 * 1024**3
    assert limits.sample_interval_seconds == 0.5
    assert limits.system_reserve_bytes == 4 * 1024**3
    with pytest.raises(ValueError):
        ResourceLimits(child_rss_bytes=8 * 1024**3)


def test_one_busy_logical_processor_is_reported_as_a_fraction_of_capacity():
    assert monitor.normalized_cpu_percent(1.0, 1.0, 12) == pytest.approx(100 / 12)
    assert monitor.normalized_cpu_percent(12.0, 1.0, 12) == pytest.approx(100.0)
    assert monitor.normalized_cpu_percent(0.0, 2.0, 12) == 0.0
    with pytest.raises(ValueError):
        monitor.normalized_cpu_percent(-1.0, 1.0, 12)


def test_benchmark_module_reuses_the_single_cpu_definition():
    from scripts.benchmark import normalized_cpu_percent

    assert normalized_cpu_percent is monitor.normalized_cpu_percent


def test_memory_pressure_requires_a_continuous_reserve_breach():
    tracker = monitor.MemoryPressureTracker(reserve_bytes=1000, seconds=10)
    assert tracker.update(500, now=0.0) is False
    assert tracker.update(500, now=9.0) is False
    assert tracker.update(1000, now=9.5) is False, "recovery must reset the window"
    assert tracker.update(500, now=10.0) is False
    assert tracker.update(500, now=19.9) is False
    assert tracker.update(500, now=20.0) is True
    assert tracker.update(500, now=21.0) is True
    assert tracker.update(2000, now=22.0) is False
    assert tracker.update(500, now=23.0) is False


@pytest.mark.parametrize("threads", [2, 4, 6])
def test_thread_settings_apply_in_real_spawned_children(tmp_path, threads):
    repo = JobRepository(tmp_path / f"metadata-{threads}.sqlite3")
    _queued_job(repo, tmp_path, f"threads-{threads}")
    supervisor = _supervisor(
        repo, limits=ResourceLimits(compute_threads=threads),
        options={"work_seconds": 0.1, "steps": 2, "report_pools": True},
    )

    final = supervisor.run_once()

    assert final.state == "completed"
    result = supervisor.last_result
    assert result["compute_threads"] == threads
    assert result["torch_intra_threads"] == threads
    assert result["torch_interop_threads"] == 1
    stages = result["threadpools"]
    # Each stage is measured separately, so a nested T-wide pair cannot hide.
    assert set(stages) == {"blas", "torch"}
    assert stages["blas"], "a real BLAS pool must be observable inside the child"
    assert result["wide_layers"] == {"blas": 1, "torch": 1}
    blas_wide = [p for p in stages["blas"] if p["num_threads"] > 1]
    assert [p["num_threads"] for p in blas_wide] == [threads]
    assert blas_wide[0]["user_api"] == "blas"
    assert all(p["num_threads"] == 1 for p in stages["blas"] if p["user_api"] == "openmp")
    torch_wide = [p for p in stages["torch"] if p["num_threads"] > 1]
    assert [p["num_threads"] for p in torch_wide] == [threads]
    assert all(p["num_threads"] == 1 for p in stages["torch"] if p["user_api"] == "blas")
    assert result["pid"] != result["parent_pid"]


def test_child_resources_are_sampled_and_reported(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    _queued_job(repo, tmp_path)
    supervisor = _supervisor(repo, options={"work_seconds": 0.6, "steps": 4})

    final = supervisor.run_once()

    assert final.state == "completed"
    resources = supervisor.last_resources
    assert resources["sample_count"] >= 3
    assert resources["peak_rss_bytes"] > 0
    assert resources["minimum_available_bytes"] > 0
    assert resources["wall_seconds"] >= 0.6
    assert resources["cpu_seconds"] >= 0
    assert resources["sample_interval_seconds"] == pytest.approx(0.1)
    assert 0 <= resources["normalized_cpu_percent"] <= 100
    assert resources["logical_cores"] >= 1
    assert "raw" not in json.dumps(resources).lower()


def test_reduced_test_only_rss_threshold_fails_the_job_safely(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(
        repo, limits=ResourceLimits(child_rss_bytes=256 * 1024**2),
        options={"work_seconds": 20.0, "steps": 1, "allocate_mb": 512, "report_pools": False},
    )

    final = supervisor.run_once()

    assert final.state == "failed"
    assert final.safe_error == "memory_limit_exceeded"
    assert final.stage == "failed"
    assert supervisor.last_resources["peak_rss_bytes"] > 256 * 1024**2
    assert supervisor.child_exit_code is not None, "a breached child must be terminated"
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT active_job_id FROM worker_lease").fetchone()[0] is None
        assert db.execute("SELECT count(*) FROM candidates").fetchone()[0] == 0
        events = [row[0] for row in db.execute("SELECT event FROM audit_events ORDER BY sequence")]
    assert "job_failed" in events


def test_wall_time_breach_is_persisted_as_a_timeout(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(
        repo, limits=ResourceLimits(cv_job_seconds=1),
        options={"work_seconds": 30.0, "steps": 2, "report_pools": False},
    )

    started = time.perf_counter()
    final = supervisor.run_once()
    elapsed = time.perf_counter() - started

    assert final.state == "failed"
    assert final.safe_error == "timeout"
    assert elapsed < 20, "a wall-clock breach must terminate the owned child"
    assert supervisor.child_exit_code is not None
    assert repo.get_job(job.job_id).safe_error == "timeout"


def test_child_crash_is_persisted_without_partial_results(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    supervisor = _supervisor(
        repo, options={"work_seconds": 0.0, "crash_exit_code": 7, "report_pools": False},
    )

    final = supervisor.run_once()

    assert final.state == "interrupted"
    assert final.safe_error == "child_exited"
    assert supervisor.child_exit_code == 7
    assert supervisor.last_result is None
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM candidates").fetchone()[0] == 0
    assert repo.get_job(job.job_id).state == "interrupted"


def test_workload_failure_never_echoes_raw_exception_text(tmp_path, caplog):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    job = _queued_job(repo, tmp_path)
    secret = "PATIENT-ROW-1999-5.4321"
    supervisor = _supervisor(
        repo, options={"work_seconds": 0.0, "raise_message": secret, "report_pools": False},
    )

    with caplog.at_level(logging.DEBUG):
        final = supervisor.run_once()

    assert final.state == "failed"
    assert final.safe_error == "workload_failed"
    with sqlite3.connect(repo.path) as db:
        rows = db.execute("SELECT * FROM jobs").fetchall()
        audit = db.execute("SELECT actor, entity_id, event FROM audit_events").fetchall()
    persisted = json.dumps([list(row) for row in rows] + [list(row) for row in audit])
    assert secret not in persisted
    assert secret not in caplog.text
    assert secret not in json.dumps(supervisor.last_resources)


def test_sampler_lifecycle_is_explicit_and_survives_child_exit(tmp_path):
    script = tmp_path / "short_child.py"
    script.write_text(
        "import time\n"
        "time.sleep(0.4)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen([sys.executable, str(script)], cwd=str(tmp_path))
    try:
        creation_time = monitor.process_creation_time(process.pid)
        sampler = monitor.ChildSampler(process.pid, creation_time=creation_time)
        with pytest.raises(RuntimeError):
            sampler.summary()  # Never started: no summary exists yet.
        with pytest.raises(RuntimeError):
            sampler.sample()
        sampler.start()
        time.sleep(0.25)
        sampler.sample()
        process.wait(timeout=60)
        time.sleep(0.2)
        sampler.sample()  # A dead child ends sampling but keeps prior observations.
        sampler.stop()
        summary = sampler.summary()
    finally:
        process.kill()
    assert summary["sample_count"] >= 2
    assert summary["peak_rss_bytes"] > 0
    assert summary["ended_before_stop"] is True
    with pytest.raises(RuntimeError):
        sampler.sample()
    with pytest.raises(RuntimeError):
        monitor.ChildSampler(process.pid, creation_time=creation_time + 60.0)


def test_child_identity_mismatch_blocks_termination(tmp_path):
    script = tmp_path / "long_child.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(script)], cwd=str(tmp_path))
    try:
        time.sleep(0.5)
        assert monitor.terminate_owned_child(process.pid, creation_time=0.0) is False
        assert process.poll() is None, "an unverified PID must never be terminated"
        assert monitor.terminate_owned_child(
            process.pid, creation_time=monitor.process_creation_time(process.pid), timeout=5.0,
        ) is True
        process.wait(timeout=10)
    finally:
        process.kill()


def test_probe_options_are_validated_before_spawning(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    with pytest.raises(ValueError):
        _supervisor(repo, options={"work_seconds": -1})
    with pytest.raises(ValueError):
        _supervisor(repo, options={"unknown_option": 1})
    with pytest.raises(ValueError):
        WorkerSupervisor(repo, limits=ResourceLimits(), workload="../../evil",
                         workload_options={}, owner_token="supervisor-a")
