"""Local tracking and fail-closed publication on the CV worker boundary."""

import json
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from apps.api.database.repository import JobRepository
from apps.api.workers.supervisor import WorkerSupervisor
from ml.contracts import TaskType
from tests.integration.test_classical_suite import prepared_fixture


def worker_for(repo, prepared, run):
    return WorkerSupervisor(repo, workload="classical", classical_input=(prepared, run),
                            heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)


def test_real_local_tracking_and_durable_diagnostics(tmp_path):
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    worker = worker_for(repo, prepared, run)
    assert worker.run_once().state == "completed"
    results = JobRepository(repo.path).read_published_results(job.job_id)
    assert len(results["candidates"]) == 5
    assert results["candidates"][0]["diagnostics"]["partition"] == "out_of_fold_training"
    assert results["manifest"]["dataset"]["canonical_sha256"] == prepared.context.split.dataset_sha256
    assert results["manifest"]["split"]["manifest_sha256"] == prepared.context.split.manifest_sha256
    assert results["manifest"]["code"]["revision"]
    assert results["manifest"]["artifact_sha256"]["results.json"]
    assert results["manifest"]["observed_threadpools"]
    client = MlflowClient(tracking_uri="sqlite:///" + (tmp_path / "mlflow.sqlite3").as_posix())
    experiments = client.search_experiments()
    runs = client.search_runs([e.experiment_id for e in experiments])
    own = [r for r in runs if r.data.tags.get("modelarena.attempt_id") == repo.get_job(job.job_id).attempt_id]
    assert len(own) == 31  # parent + five candidates + five folds each
    assert all(r.info.status == "FINISHED" for r in own)
    assert sum(r.data.tags.get("modelarena.kind") == "fold" for r in own) == 25
    assert all(r.data.tags.get("modelarena.job_id") == job.job_id for r in own)
    parent = next(r for r in own if r.data.tags["modelarena.kind"] == "attempt")
    assert {a.path for a in client.list_artifacts(parent.info.run_id)} == {"manifest.json", "results.json"}


def test_tracking_failure_cannot_complete_or_publish(tmp_path, monkeypatch):
    from ml.tracking import mlflow_store
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    monkeypatch.setattr(mlflow_store, "track_attempt", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    worker = worker_for(repo, prepared, run)
    assert worker.run_once().state == "failed"
    assert worker.last_result is None
    with pytest.raises(LookupError):
        repo.read_published_results(job.job_id)


def test_disk_write_failure_cannot_publish(tmp_path, monkeypatch):
    from ml.tracking import publication
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    monkeypatch.setattr(publication, "_write", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert worker_for(repo, prepared, run).run_once().state == "failed"
    with repo._connection() as db:
        assert db.execute("SELECT count(*) FROM candidates WHERE job_id=?", (job.job_id,)).fetchone()[0] == 0


def test_tampered_private_bundle_is_never_read(tmp_path):
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    assert worker_for(repo, prepared, run).run_once().state == "completed"
    with repo._connection() as db:
        path = Path(db.execute("SELECT bundle_path FROM candidates WHERE job_id=? LIMIT 1", (job.job_id,)).fetchone()[0])
    (path / "results.json").write_text(json.dumps({"candidates": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        repo.read_published_results(job.job_id)


def test_database_failure_rolls_back_candidates_and_identifies_orphans(tmp_path):
    from ml.tracking.reconcile import reconcile_orphans
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    with repo._connection() as db:
        db.execute("CREATE TRIGGER inject_failure BEFORE INSERT ON candidates BEGIN SELECT RAISE(ABORT, 'disk error'); END")
    worker = worker_for(repo, prepared, run)
    assert worker.run_once().state == "failed"
    with repo._connection() as db:
        assert db.execute("SELECT count(*) FROM candidates WHERE job_id=?", (job.job_id,)).fetchone()[0] == 0
    with pytest.raises(LookupError):
        repo.read_published_results(job.job_id)
    client = MlflowClient(tracking_uri="sqlite:///" + (tmp_path / "mlflow.sqlite3").as_posix())
    experiment = client.get_experiment_by_name("modelarena-" + prepared.experiment_id)
    unfinished = client.create_run(experiment.experiment_id, tags={"modelarena.attempt_id": repo.get_job(job.job_id).attempt_id})
    report = reconcile_orphans(repo)
    assert len(report["bundle_paths"]) == 1 and len(report["mlflow_run_ids"]) == 32
    assert reconcile_orphans(repo) == report
    assert all(client.get_run(run_id).data.tags["modelarena.orphaned"] == "true" for run_id in report["mlflow_run_ids"])
    assert client.get_run(unfinished.info.run_id).info.status == "KILLED"


def test_cancel_during_publication_wins_before_database_commit(tmp_path, monkeypatch):
    from ml.tracking import publication
    from ml.tracking.reconcile import reconcile_orphans
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    original = publication.stage_attempt
    def stage_then_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        repo.request_cancel(job.job_id)
        return result
    monkeypatch.setattr(publication, "stage_attempt", stage_then_cancel)
    worker = worker_for(repo, prepared, run)
    assert worker.run_once().state == "cancelled"
    assert worker.last_result is None
    with pytest.raises(LookupError):
        repo.read_published_results(job.job_id)
    assert len(reconcile_orphans(repo)["mlflow_run_ids"]) == 31


def test_missing_metrics_are_not_logged_as_nan(tmp_path):
    from ml.tracking.mlflow_store import track_attempt
    from types import SimpleNamespace
    job = SimpleNamespace(experiment_id="test-experiment", job_id="test-job", attempt_id="test-attempt")
    candidates = [{"family": "dummy", "model_seed": 42, "primary_metric": "f1",
                   "summary": [{"name": "f1", "mean": None, "std": None, "reason": "undefined"}],
                   "folds": [{"fold_index": 0, "status": "completed", "fit_seconds": 0.0,
                              "score_seconds": 0.0, "metrics": {"metrics": [
                                  {"name": "f1", "value": None, "reason": "undefined"}]}}]}]
    ids, client = track_attempt(tmp_path, job, candidates)
    fold = ids["candidates"]["dummy"]["folds"]["0"]
    assert "f1" not in client.get_run(fold).data.metrics
    assert "f1_mean" not in client.get_run(ids["candidates"]["dummy"]["run_id"]).data.metrics
