"""CV-only leaderboard over the checksummed published worker bundle."""

from contextlib import contextmanager
import sqlite3

import pytest
from fastapi.testclient import TestClient

from apps.api.database.repository import JobRepository, NotFoundError
from apps.api.main import create_app
from apps.api.services.leaderboard import get_leaderboard
from apps.api.workers.supervisor import WorkerSupervisor
from ml.contracts import TaskType
from tests.integration.test_classical_suite import prepared_fixture


class PoisonRepository(JobRepository):
    @contextmanager
    def _connection(self):
        with super()._connection() as db:
            def authorize(action, table, _column, _database, _source):
                if action == sqlite3.SQLITE_READ and table == "holdout_results":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(authorize)
            yield db


def test_cv_leaderboard_reads_only_published_results(tmp_path):
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    before = get_leaderboard(repo, prepared.experiment_id)
    assert before.ranked == () and before.unranked == ()
    assert before.job_id is None

    worker = WorkerSupervisor(repo, workload="classical", classical_input=(prepared, run),
                              heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
    assert worker.run_once().state == "completed"
    with repo._connection() as db:
        db.execute("CREATE TABLE holdout_results(secret TEXT)")
    result = get_leaderboard(PoisonRepository(repo.path), prepared.experiment_id)
    bundle = repo.read_published_results(job.job_id)
    assert result.job_id == job.job_id
    assert result.primary_metric == "rmse" and result.direction == "minimize"
    assert [entry.family for entry in result.ranked] == [entry["family"] for entry in bundle["comparison"]["ranked"]]
    assert len(result.ranked) + len(result.unranked) == 5
    assert len({entry.candidate_id for entry in (*result.ranked, *result.unranked)}) == 5
    assert all(entry.candidate_id and entry.mean is not None for entry in result.ranked)
    assert all(entry.mean is None for entry in result.unranked)
    app = create_app(PoisonRepository(repo.path), session_token="a" * 43)
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": "a" * 43}) as client:
        response = client.get(f"/api/v1/experiments/{prepared.experiment_id}/leaderboard")
        assert response.status_code == 200
        assert response.json()["job_id"] == job.job_id
        assert [entry["candidate_id"] for entry in response.json()["ranked"]] == [entry.candidate_id for entry in result.ranked]


def test_unknown_experiment_is_not_an_empty_leaderboard(tmp_path):
    with pytest.raises(NotFoundError):
        get_leaderboard(JobRepository(tmp_path / "metadata.sqlite3"), "unknown")


def test_failed_or_incomplete_jobs_do_not_supply_results(tmp_path):
    repo, prepared, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    repo.request_cancel(job.job_id)
    result = get_leaderboard(repo, prepared.experiment_id)
    assert result.job_id is None and result.ranked == ()


def test_leaderboard_never_reads_a_holdout_table(tmp_path):
    repo, prepared, _, _ = prepared_fixture(tmp_path, TaskType.REGRESSION)
    with repo._connection() as db:
        db.execute("CREATE TABLE holdout_results(secret TEXT)")

    result = get_leaderboard(PoisonRepository(repo.path), prepared.experiment_id)
    assert result.partition == "cross_validation_training"


def test_rejects_inconsistent_comparison_metadata(tmp_path, monkeypatch):
    repo, prepared, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    with repo._connection() as db:
        db.execute("UPDATE jobs SET state='completed' WHERE job_id=?", (job.job_id,))
        db.execute("INSERT INTO candidates(candidate_id,experiment_id,job_id,attempt_id,family,status,config_sha256) "
                   "VALUES(?,?,?,?,?,?,?)", ("candidate", prepared.experiment_id, job.job_id, "attempt", "dummy", "failed", "hash"))
    monkeypatch.setattr(repo, "read_published_results", lambda _: {"comparison": {"ranked": [
        {"family": "dummy", "mean": 1.0, "std": 0.0, "mean_fit_seconds": 1.0, "baseline_delta": 0.0, "tied": False}],
        "unranked": []}})
    with pytest.raises(ValueError, match="metadata"):
        get_leaderboard(repo, prepared.experiment_id)
