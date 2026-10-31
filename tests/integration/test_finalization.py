"""Finalization dispatch safety regressions; fixture data only."""

from unittest.mock import patch
import multiprocessing
import os

import psutil

import pytest

from apps.api.database.repository import ConflictError, JobRepository, WorkerIdentity
from apps.api.services.finalization import compute_candidate_digest, get_finalization_status, reserve
from apps.api.services.retention import _key, _scope_hmac
from apps.api.workers.supervisor import WorkerSupervisor
from ml.contracts import TaskType
from tests.integration.test_classical_suite import prepared_fixture
from tests.integration.test_review_finalization_regressions import claim_finalization, reservation


def test_probe_worker_must_not_claim_finalization(reservation):
    repo, _, _, _, job, _ = reservation
    # The older CV job is not part of this dispatch check.
    with repo._transaction() as db:
        db.execute("UPDATE jobs SET state='cancelled' WHERE kind='cv' AND state='queued'")
    worker = WorkerSupervisor(repo, workload="probe")
    assert worker.run_once() is None
    assert repo.get_job(job.job_id).state == "queued"


@pytest.mark.parametrize("publication", [None, {}])
def test_generic_completion_cannot_publish_a_finalization_without_a_bundle(reservation, publication):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    with patch("apps.api.database.repository.process_liveness", return_value="dead"):
        with pytest.raises(ConflictError):
            repo.finish_job(owner, job.job_id, claimed.attempt_id,
                            state="completed", stage="completed", publication=publication)
    assert repo.get_job(job.job_id).state == "running"
    assert repo.get_finalization(selection.scope_id)["state"] == "refitting"


def test_generic_finalization_completion_respects_prior_cancellation(reservation):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    repo.request_cancel(job.job_id)
    with patch("apps.api.database.repository.process_liveness", return_value="dead"):
        result = repo.finish_job(owner, job.job_id, claimed.attempt_id,
                                 state="completed", stage="completed")
    assert result.state == "cancelled"
    assert repo.get_finalization(selection.scope_id)["state"] == "retryable_pre_evaluation"


@pytest.mark.parametrize("state", ["reserved", "consumed"])
@pytest.mark.parametrize("entrypoint", ["service", "repository"])
def test_deleted_scope_tombstone_blocks_reservation(tmp_path, state, entrypoint):
    repo, prepared, _, cv_job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    with repo._transaction() as db:
        scope = db.execute(
            "SELECT s.scope_id,s.canonical_sha256,s.target,s.task,sp.manifest_sha256 "
            "FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN evaluation_scopes s ON s.scope_id=c.scope_id JOIN splits sp ON sp.split_id=c.split_id "
            "WHERE e.experiment_id=?", (prepared.experiment_id,),
        ).fetchone()
        db.execute("INSERT INTO candidates(candidate_id,experiment_id,job_id,attempt_id,family,status,config_sha256) "
                   "VALUES(?,?,?,?,?,'completed',?)", ("candidate", prepared.experiment_id, cv_job.job_id,
                   "cv-attempt", "dummy", "config-hash"))
    key = _key(repo, create=True)
    identifier = _scope_hmac(key, scope["canonical_sha256"], scope["target"], scope["task"])
    with repo._transaction() as db:
        db.execute("INSERT INTO scope_tombstones(scope_hmac,version,state) VALUES(?,1,?)", (identifier, state))
    digest = compute_candidate_digest("candidate", "dummy", scope["manifest_sha256"], "config-hash", "rmse", 1.0, 42)
    bundle = {"candidates": [{"family": "dummy", "eligible": True, "model_seed": 42,
                              "summary": [{"name": "rmse", "mean": 1.0}]}]}
    with patch.object(repo, "read_published_results", return_value=bundle):
        with pytest.raises(ConflictError):
            if entrypoint == "service":
                reserve(repo, prepared.experiment_id, "candidate", digest, True)
            else:
                repo.reserve_finalization(scope["scope_id"], "candidate", digest,
                                          experiment_id=prepared.experiment_id)
    assert repo.get_finalization(scope["scope_id"]) is None


@pytest.mark.parametrize("state", ["reserved", "consumed"])
def test_deleted_scope_status_preserves_tombstone_lock(tmp_path, state):
    repo, prepared, _, _ = prepared_fixture(tmp_path, TaskType.REGRESSION)
    with repo._connection() as db:
        scope = db.execute(
            "SELECT s.canonical_sha256,s.target,s.task FROM experiments e "
            "JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN evaluation_scopes s ON s.scope_id=c.scope_id WHERE e.experiment_id=?",
            (prepared.experiment_id,),
        ).fetchone()
    key = _key(repo, create=True)
    identifier = _scope_hmac(key, scope["canonical_sha256"], scope["target"], scope["task"])
    with repo._transaction() as db:
        db.execute("INSERT INTO scope_tombstones(scope_hmac,version,state) VALUES(?,1,?)", (identifier, state))

    status = get_finalization_status(repo, prepared.experiment_id)
    assert status["state"] == state
    assert status["finalization_id"] is None
    assert status["result"] is None


@pytest.mark.parametrize("state", ["reserved", "consumed"])
def test_orphaned_scope_marker_cannot_be_reselected(tmp_path, state):
    repo, prepared, _, cv_job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    with repo._transaction() as db:
        scope = db.execute(
            "SELECT c.scope_id,s.manifest_sha256 FROM experiments e "
            "JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN splits s ON s.split_id=c.split_id WHERE e.experiment_id=?",
            (prepared.experiment_id,),
        ).fetchone()
        db.execute("INSERT INTO candidates(candidate_id,experiment_id,job_id,attempt_id,family,status,config_sha256) "
                   "VALUES(?,?,?,?,?,'completed',?)", ("candidate", prepared.experiment_id, cv_job.job_id,
                   "cv-attempt", "dummy", "config-hash"))
        db.execute("INSERT INTO scope_lock_markers(scope_id,state) VALUES(?,?)", (scope["scope_id"], state))
    digest = compute_candidate_digest("candidate", "dummy", scope["manifest_sha256"], "config-hash", "rmse", 1.0, 42)
    bundle = {"candidates": [{"family": "dummy", "eligible": True, "model_seed": 42,
                              "summary": [{"name": "rmse", "mean": 1.0}]}]}
    with patch.object(repo, "read_published_results", return_value=bundle):
        with pytest.raises(ConflictError):
            reserve(repo, prepared.experiment_id, "candidate", digest, True)
    assert repo.get_finalization(scope["scope_id"]) is None
    assert get_finalization_status(repo, prepared.experiment_id)["state"] == state


@pytest.mark.parametrize("task", list(TaskType))
def test_stored_refit_input_never_opens_source_or_holdout(tmp_path, task, monkeypatch):
    from apps.api.services import workflow
    from pathlib import Path

    repo, prepared, run, _ = prepared_fixture(tmp_path, task)
    monkeypatch.setattr(workflow, "load_dataset", lambda *args: pytest.fail("source reopened"))
    original = Path.open
    def open_spy(path, *args, **kwargs):
        if path.name in ("holdout.json", "source.csv", "input.csv"):
            pytest.fail("refit input opened source or holdout")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_spy)
    loaded, loaded_run = workflow.load_refit_experiment(repo, prepared.experiment_id)
    assert loaded.context == prepared.context
    assert loaded_run == run
    assert set(loaded.context.training.row_ids).isdisjoint(loaded.context.split.holdout_row_ids)


def test_stored_training_input_tampering_fails_closed(tmp_path):
    from apps.api.services.workflow import load_refit_experiment

    repo, prepared, _, _ = prepared_fixture(tmp_path, TaskType.REGRESSION)
    with repo._connection() as db:
        row = db.execute(
            "SELECT c.scope_id,c.split_id FROM experiments e JOIN configurations c "
            "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
            (prepared.experiment_id,),
        ).fetchone()
    partition = repo.path.parent / "datasets" / row["scope_id"] / row["split_id"] / "train.json"
    partition.write_bytes(partition.read_bytes().replace(b'group-0', b'group-X', 1))
    with pytest.raises((ValueError, ConflictError)):
        load_refit_experiment(repo, prepared.experiment_id)


def test_partition_preparation_cannot_recreate_files_after_deletion(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from apps.api.services import partitions
    from apps.api.services.retention import request_deletion
    from ml.data.csv_io import validate_csv

    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    repo.request_cancel(job.job_id)
    data = validate_csv(tmp_path / "input.csv", run.limits)
    with repo._connection() as db:
        row = db.execute("SELECT c.dataset_id,c.scope_id FROM experiments e JOIN configurations c "
                         "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
                         (prepared.experiment_id,)).fetchone()
    resolved, release = Event(), Event()
    original = partitions._folder

    def pause_after_check(*args, **kwargs):
        folder = original(*args, **kwargs)
        resolved.set()
        assert release.wait(15)
        return folder

    monkeypatch.setattr(partitions, "_folder", pause_after_check)
    with ThreadPoolExecutor(max_workers=2) as executor:
        writing = executor.submit(partitions.save_training_partition, repo, prepared.experiment_id,
                                  data, prepared.context.split)
        try:
            assert resolved.wait(15)
            deleting = executor.submit(request_deletion, repo, row["dataset_id"])
            # In the broken implementation deletion finishes while preparation is
            # paused, then preparation recreates the supposedly removed directory.
            from concurrent.futures import TimeoutError
            try:
                deleting.result(timeout=1)
            except TimeoutError:
                pass
        finally:
            release.set()
        writing.result(timeout=20)
        assert deleting.result(timeout=20)["state"] == "deletion_pending"
    assert not (repo.path.parent / "datasets" / row["scope_id"]).exists()


def test_partition_preparation_rejects_existing_deletion_intent(tmp_path):
    from apps.api.services.partitions import save_training_partition
    from apps.api.services.retention import request_deletion
    from ml.data.csv_io import validate_csv

    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    repo.request_cancel(job.job_id)
    data = validate_csv(tmp_path / "input.csv", run.limits)
    with repo._connection() as db:
        row = db.execute("SELECT c.dataset_id,c.scope_id FROM experiments e JOIN configurations c "
                         "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
                         (prepared.experiment_id,)).fetchone()
    request_deletion(repo, row["dataset_id"])
    with pytest.raises(ConflictError, match="deletion"):
        save_training_partition(repo, prepared.experiment_id, data, prepared.context.split)
    assert not (repo.path.parent / "datasets" / row["scope_id"]).exists()


def test_real_holdout_partition_requires_durable_start_and_is_single_use(reservation):
    from apps.api.services.finalization import start_evaluation
    from ml.evaluation.finalize import read_holdout

    repo, prepared, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    with pytest.raises((TypeError, PermissionError)):
        read_holdout(repo, None)
    capability = start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                                  owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)
    view = read_holdout(repo, capability)
    assert view.row_ids == prepared.context.split.holdout_row_ids
    assert set(view.row_ids).isdisjoint(prepared.context.training.row_ids)
    assert view.targets == tuple(float(i * 1.5) for i in view.row_ids)
    assert repo.get_finalization(selection.scope_id)["state"] == "evaluating"
    with pytest.raises(ConflictError):
        read_holdout(repo, capability)


def test_real_holdout_reader_rejects_fenced_attempt_before_open(reservation, monkeypatch):
    from apps.api.services.finalization import start_evaluation
    from ml.evaluation.finalize import read_holdout

    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    capability = start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                                  owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)
    with repo._transaction() as db:
        db.execute("UPDATE worker_lease SET generation=generation+1 WHERE singleton=1")
    from pathlib import Path
    original = Path.open
    def open_spy(path, *args, **kwargs):
        if path.name == "holdout.json":
            pytest.fail("fenced attempt opened holdout")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_spy)
    with pytest.raises(ConflictError):
        read_holdout(repo, capability)


def test_holdout_corruption_after_start_consumes_authorization(reservation):
    from apps.api.services.finalization import start_evaluation
    from ml.evaluation.finalize import read_holdout

    repo, prepared, _, selection, job, _ = reservation
    with repo._connection() as db:
        split_id = db.execute("SELECT c.split_id FROM experiments e JOIN configurations c "
                              "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
                              (prepared.experiment_id,)).fetchone()[0]
    path = repo.path.parent / "datasets" / selection.scope_id / split_id / "holdout.json"
    path.write_bytes(path.read_bytes().replace(b'group-0', b'group-X', 1))
    owner, claimed = claim_finalization(repo, job)
    capability = start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                                  owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)
    with pytest.raises(ValueError, match="checksum"):
        read_holdout(repo, capability)
    assert repo.get_finalization(selection.scope_id)["state"] == "evaluating"
    with pytest.raises(ConflictError):
        read_holdout(repo, capability)
    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "after-corruption")


def _spawned_holdout_reader(db_path, incoming, outgoing):
    from ml.evaluation.finalize import read_holdout
    capability = incoming.get(timeout=30)
    view = read_holdout(JobRepository(db_path), capability)
    outgoing.put((os.getpid(), view.row_ids, view.targets))


def test_spawned_child_reads_actual_holdout_after_start(reservation):
    from apps.api.services.finalization import start_evaluation

    repo, prepared, _, selection, job, _ = reservation
    with repo._transaction() as db:
        db.execute("UPDATE jobs SET state='cancelled' WHERE kind='cv' AND state='queued'")
    owner = WorkerIdentity("spawned-reader", repo.lease_generation())
    claimed = repo.claim_next(owner, kind="finalization",
                              supervisor_process=(os.getpid(), psutil.Process().create_time()))
    ctx = multiprocessing.get_context("spawn")
    incoming, outgoing = ctx.Queue(), ctx.Queue()
    child = ctx.Process(target=_spawned_holdout_reader, args=(repo.path, incoming, outgoing))
    child.start()
    try:
        repo.record_child(owner, job.job_id, claimed.attempt_id, child.pid,
                          psutil.Process(child.pid).create_time())
        capability = start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                                      owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)
        incoming.put(capability)
        pid, ids, targets = outgoing.get(timeout=40)
        child.join(timeout=10)
        assert child.exitcode == 0 and pid == child.pid and pid != os.getpid()
        assert ids == prepared.context.split.holdout_row_ids
        assert targets == tuple(float(i * 1.5) for i in ids)
        with repo._connection() as db:
            assert db.execute("SELECT used FROM evaluation_authorizations WHERE finalization_id=?",
                              (selection.finalization_id,)).fetchone()[0] == 1
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
        incoming.close()
        outgoing.close()


def test_equivalent_upload_deletion_removes_shared_partitions(tmp_path):
    from fastapi.testclient import TestClient
    from apps.api.main import create_app
    from tests.integration.test_retention import TOKEN, prepared

    repo = JobRepository(tmp_path / "metadata.sqlite3")
    with TestClient(create_app(repo, session_token=TOKEN), base_url="http://localhost:8000",
                    headers={"X-ModelArena-Token": TOKEN}) as client:
        original, experiment_id, csv = prepared(client)
        equivalent = client.post('/api/v1/datasets', files={'file': ('b.csv', csv.replace(b'\n', b'\r\n'))})
        assert equivalent.status_code == 201
        other = equivalent.json()['dataset_id']
        assert other != original
        with repo._connection() as db:
            scope = db.execute("SELECT c.scope_id FROM experiments e JOIN configurations c "
                               "ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
                               (experiment_id,)).fetchone()[0]
        folder = tmp_path / 'datasets' / scope
        assert folder.exists()
        response = client.request('DELETE', f'/api/v1/datasets/{other}', json={'acknowledge': True})
        assert response.status_code == 202
        assert not response.json()['file_cleanup_failed']
        assert not folder.exists()


def _upload_equivalent_fixture(repo, tmp_path, limits):
    from apps.api.services.datasets import publish_upload, upload_root

    # Exercise the actual upload path with distinct bytes and the same rows.
    staging = upload_root(repo) / 'equivalent.csv'
    original = (tmp_path / 'input.csv').read_bytes()
    normalized = original.replace(b'\r\n', b'\n')
    content = normalized if normalized != original else normalized.replace(b'\n', b'\r\n')
    staging.write_bytes(content)
    result = publish_upload(repo, staging, limits)
    staging.unlink(missing_ok=True)
    with repo._connection() as db:
        count = db.execute('SELECT count(*) FROM datasets WHERE canonical_sha256=?',
                           (result['canonical_sha256'],)).fetchone()[0]
    assert count == 2
    return result['dataset_id']


def test_retry_after_equivalent_dataset_deletion_is_rejected(reservation, tmp_path):
    from apps.api.services.retention import request_deletion

    repo, _, run, selection, job, _ = reservation
    other = _upload_equivalent_fixture(repo, tmp_path, run.limits)
    repo.request_cancel(job.job_id)
    with repo._connection() as db:
        cv = db.execute("SELECT job_id FROM jobs WHERE kind='cv'").fetchone()[0]
    repo.request_cancel(cv)
    assert request_deletion(repo, other)['state'] == 'deletion_pending'
    with pytest.raises(ConflictError, match='deletion'):
        repo.retry_pre_evaluation(selection.finalization_id, 'after-equivalent-deletion')
    with pytest.raises(ConflictError, match='deletion'):
        repo.enqueue_finalization(selection.finalization_id, 'after-equivalent-deletion')


def test_equivalent_upload_deletion_serializes_with_partition_preparation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event
    from apps.api.services import partitions
    from apps.api.services.retention import request_deletion
    from ml.data.csv_io import validate_csv

    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    repo.request_cancel(job.job_id)
    data = validate_csv(tmp_path / 'input.csv', run.limits)
    other = _upload_equivalent_fixture(repo, tmp_path, run.limits)
    with repo._connection() as db:
        scope = db.execute('SELECT scope_id FROM evaluation_scopes WHERE canonical_sha256=?',
                           (data.canonical_sha256,)).fetchone()[0]
    entered, release, deletion_started = Event(), Event(), Event()
    original = partitions._folder

    def pause(*args, **kwargs):
        folder = original(*args, **kwargs)
        entered.set()
        assert release.wait(15)
        return folder

    def delete_equivalent():
        deletion_started.set()
        return request_deletion(repo, other)

    monkeypatch.setattr(partitions, '_folder', pause)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(partitions.save_training_partition, repo, prepared.experiment_id, data, prepared.context.split)
        try:
            assert entered.wait(15)
            deleting = pool.submit(delete_equivalent)
            assert deletion_started.wait(15)
            with pytest.raises(TimeoutError):
                deleting.result(timeout=1)
        finally:
            release.set()
        writer.result(timeout=20)
        assert deleting.result(timeout=20)['state'] == 'deletion_pending'
    assert not (tmp_path / 'datasets' / scope).exists()
    with pytest.raises(ConflictError, match='deletion'):
        partitions.save_training_partition(repo, prepared.experiment_id, data, prepared.context.split)
    assert not (tmp_path / 'datasets' / scope).exists()


def _run_fixture_cv(tmp_path, task=TaskType.REGRESSION, family="dummy"):
    import shutil
    from ml.contracts import ResourceLimits, RunConfig
    from ml.data.csv_io import validate_csv
    from apps.api.services.datasets import upload_root
    from apps.api.services.jobs import prepare_experiment
    from apps.api.workers.supervisor import WorkerSupervisor

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

    worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, limits=limits)
    completed_cv = worker.run_once()
    assert completed_cv is not None and completed_cv.state == "completed"
    with repo._connection() as db:
        cands = db.execute("SELECT candidate_id, family, config_sha256 FROM candidates WHERE job_id=? AND family=?",
                           (job.job_id, family)).fetchall()
        assert len(cands) == 1
        cand_id, cand_family, config_sha = cands[0]
        scope = db.execute("SELECT c.scope_id, s.manifest_sha256 FROM experiments e "
                           "JOIN configurations c ON c.configuration_id=e.configuration_id "
                           "JOIN splits s ON s.split_id=c.split_id WHERE e.experiment_id=?",
                           (prepared.experiment_id,)).fetchone()
    bundle = repo.read_published_results(job.job_id)
    cand_data = next(c for c in bundle["candidates"] if c["family"] == cand_family)
    metric_name = "f1" if task == TaskType.BINARY_CLASSIFICATION else "rmse"
    score = next(s["mean"] for s in cand_data["summary"] if s["name"] == metric_name)
    digest = compute_candidate_digest(cand_id, cand_family, scope["manifest_sha256"], config_sha,
                                      metric_name, score, cand_data.get("model_seed", 42))
    return repo, prepared, run, cand_id, digest, scope["scope_id"]


def test_production_finalization_worker_evaluates_and_publishes_once_with_repeated_retrieval(tmp_path):
    from fastapi.testclient import TestClient
    from apps.api.main import create_app
    from tests.integration.test_retention import TOKEN

    repo, prepared, run, cand_id, digest, scope_id = _run_fixture_cv(tmp_path)
    selection, job = reserve(repo, prepared.experiment_id, cand_id, digest, acknowledge=True)
    assert job is not None and job.kind == "finalization"
    assert repo.get_finalization(scope_id)["state"] == "refitting"

    worker = WorkerSupervisor(repo, workload="finalization", load_from_storage=True, limits=run.limits)
    finished = worker.run_once()
    assert finished is not None and finished.state == "completed"
    assert worker.last_child_pid is not None and worker.last_child_pid != os.getpid()

    status = get_finalization_status(repo, prepared.experiment_id)
    assert status["state"] == "evaluated"
    assert status["result"] is not None
    assert "metrics" in status["result"]
    assert status["result"]["metrics"]["task"] == "regression"

    with repo._connection() as db:
        used = db.execute("SELECT used FROM evaluation_authorizations WHERE finalization_id=?",
                          (selection.finalization_id,)).fetchone()[0]
    assert used == 1

    client = TestClient(create_app(repo, session_token=TOKEN), base_url="http://localhost:8000",
                        headers={"X-ModelArena-Token": TOKEN})
    with client:
        res1 = client.get(f"/api/v1/experiments/{prepared.experiment_id}/finalization")
        assert res1.status_code == 200
        assert res1.json()["state"] == "evaluated"
        assert res1.json()["result"] == status["result"]

        res2 = client.post(f"/api/v1/experiments/{prepared.experiment_id}/finalization",
                           json={"candidate_id": cand_id, "expected_digest": digest, "acknowledge": True})
        assert res2.status_code in (200, 202)
        assert res2.json()["state"] == "evaluated"
        assert res2.json()["result"] == status["result"]

    with repo._connection() as db:
        used_after = db.execute("SELECT used FROM evaluation_authorizations WHERE finalization_id=?",
                                (selection.finalization_id,)).fetchone()[0]
    assert used_after == 1


def test_production_finalization_pre_start_failure_permits_identical_retry(tmp_path):
    repo, prepared, run, cand_id, digest, scope_id = _run_fixture_cv(tmp_path)
    selection, job = reserve(repo, prepared.experiment_id, cand_id, digest, acknowledge=True)

    worker = WorkerSupervisor(repo, workload="finalization", load_from_storage=True, limits=run.limits,
                              options={"raise_before_evaluation": "simulated refit child failure"})
    failed_job = worker.run_once()
    assert failed_job.state in ("failed", "interrupted")

    fin = repo.get_finalization(scope_id)
    assert fin["evaluation_started_at"] is None
    assert fin["state"] == "retryable_pre_evaluation"

    with pytest.raises(ConflictError):
        repo.reserve_finalization(scope_id, cand_id, "0" * 64, experiment_id=prepared.experiment_id)

    retried_job = repo.retry_pre_evaluation(selection.finalization_id, "retry-1")
    assert retried_job.state == "queued"

    worker2 = WorkerSupervisor(repo, workload="finalization", load_from_storage=True, limits=run.limits)
    completed = worker2.run_once()
    assert completed is not None and completed.state == "completed"
    assert repo.get_finalization(scope_id)["state"] == "evaluated"


def test_production_finalization_post_start_crash_remains_locked(tmp_path):
    repo, prepared, run, cand_id, digest, scope_id = _run_fixture_cv(tmp_path)
    selection, job = reserve(repo, prepared.experiment_id, cand_id, digest, acknowledge=True)

    worker = WorkerSupervisor(repo, workload="finalization", load_from_storage=True, limits=run.limits,
                              options={"crash_after_evaluation_start": 42})
    outcome = worker.run_once()
    assert outcome.state in ("failed", "interrupted")

    fin = repo.get_finalization(scope_id)
    assert fin["evaluation_started_at"] is not None
    assert fin["state"] == "locked_failed"

    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "post-crash-retry")


def test_production_finalization_stale_attempt_fencing(tmp_path):
    from apps.api.services.finalization import start_evaluation

    repo, prepared, run, cand_id, digest, scope_id = _run_fixture_cv(tmp_path)
    selection, job = reserve(repo, prepared.experiment_id, cand_id, digest, acknowledge=True)

    owner = WorkerIdentity("owner-stale", repo.lease_generation())
    claimed = repo.claim_next(owner, supervisor_process=(os.getpid(), psutil.Process().create_time()), kind="finalization")
    assert claimed is not None

    with repo._transaction() as db:
        db.execute("UPDATE worker_lease SET generation=generation+1 WHERE singleton=1")

    with pytest.raises(ConflictError):
        start_evaluation(repo, selection.finalization_id, selection.candidate_digest,
                         owner=owner, job_id=job.job_id, attempt_id=claimed.attempt_id)


def test_production_finalization_complete_staged_bundle_recovery_without_recomputation(tmp_path, monkeypatch):
    repo, prepared, run, cand_id, digest, scope_id = _run_fixture_cv(tmp_path)
    selection, job = reserve(repo, prepared.experiment_id, cand_id, digest, acknowledge=True)

    def crash_before_publish(*args, **kwargs):
        raise RuntimeError("simulated crash after staging")
    monkeypatch.setattr(repo, "publish_finalization", crash_before_publish)

    worker = WorkerSupervisor(repo, workload="finalization", load_from_storage=True, limits=run.limits)
    with pytest.raises(RuntimeError):
        worker.run_once()

    monkeypatch.undo()
    active_job = repo.get_job(job.job_id)
    assert active_job.attempt_id is not None
    with repo._transaction() as db:
        db.execute("UPDATE worker_processes SET supervisor_created=0.0 WHERE attempt_id=?", (active_job.attempt_id,))
    assert repo.reconcile_worker() is True

    fin = repo.get_finalization(scope_id)
    assert fin["state"] == "evaluated"
    assert fin["result_checksum"] is not None
    assert repo.get_job(job.job_id).state == "completed"


def test_reservation_durably_binds_evidence_and_budget_enforcing_identity(tmp_path):
    repo, prepared, run, cand_id, digest, scope_id = _run_fixture_cv(tmp_path)
    selection, job = reserve(repo, prepared.experiment_id, cand_id, digest, acknowledge=True)

    fin = repo.get_finalization(scope_id)
    assert fin["evidence_sha256"] is not None
    assert "plan_json" in fin and fin["plan_json"] is not None

    repo.request_cancel(job.job_id)
    with repo._transaction() as db:
        db.execute("UPDATE jobs SET state='cancelled' WHERE job_id=?", (job.job_id,))
        db.execute("UPDATE finalizations SET state='retryable_pre_evaluation' WHERE finalization_id=?",
                   (selection.finalization_id,))

    with repo._transaction() as db:
        db.execute("UPDATE candidates SET config_sha256='tampered' WHERE candidate_id=?", (cand_id,))
    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "retry-tampered")
