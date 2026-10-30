"""Integration tests for the immutable holdout reservation and finalization gate."""

from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3
from threading import Barrier
from time import sleep

import pytest
from fastapi.testclient import TestClient

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError, WorkerIdentity
from apps.api.main import create_app
from apps.api.workers.supervisor import WorkerSupervisor
from apps.api.services.finalization import (
    compute_candidate_digest,
    fixture_evaluate_holdout,
    get_finalization_status,
    issue_holdout_capability,
    reserve,
    retry,
    start_evaluation,
)
from ml.contracts import FrozenSelection, HoldoutCapability, TaskType


TOKEN = "a" * 43


def run_cv_experiment(tmp_path, task=TaskType.REGRESSION):
    """Set up and run a full classical CV suite to produce eligible candidates."""
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    app = create_app(repo, session_token=TOKEN)
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        csv = ("x,category,target\n" + "\n".join(
            f"{i},group-{i % 3},{i * 2}" if task == TaskType.REGRESSION
            else f"{i},group-{i % 3},{'yes' if i % 2 else 'no'}"
            for i in range(100)
        )).encode("utf-8")
        dataset_res = client.post("/api/v1/datasets", files={"file": ("input.csv", csv)})
        assert dataset_res.status_code == 201
        dataset_id = dataset_res.json()["dataset_id"]

        config_payload = {
            "task": str(task),
            "target": "target",
            "features": [
                {"name": "x", "dtype": "numeric"},
                {"name": "category", "dtype": "categorical"},
            ],
            "positive_label": "yes" if task == TaskType.BINARY_CLASSIFICATION else None,
        }
        config_res = client.post(f"/api/v1/datasets/{dataset_id}/configurations", json=config_payload)
        assert config_res.status_code == 201
        config_id = config_res.json()["configuration_id"]

        exp_res = client.post("/api/v1/experiments", json={"configuration_id": config_id})
        assert exp_res.status_code == 201
        experiment_id = exp_res.json()["experiment_id"]

        job_res = client.post(f"/api/v1/experiments/{experiment_id}/jobs", json={"idempotency_key": "cv-job"})
        assert job_res.status_code == 202
        job_id = job_res.json()["job_id"]

        # Run worker
        worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
        job_record = worker.run_once()
        assert job_record.state == "completed"

        # Read leaderboard
        leaderboard_res = client.get(f"/api/v1/experiments/{experiment_id}/leaderboard")
        assert leaderboard_res.status_code == 200
        leaderboard = leaderboard_res.json()
        assert len(leaderboard["ranked"]) > 0

        # Scope ID
        with repo._connection() as db:
            scope_id = db.execute("SELECT scope_id FROM configurations WHERE configuration_id=?", (config_id,)).fetchone()[0]

        return repo, app, dataset_id, config_id, experiment_id, scope_id, leaderboard


def candidate_digest_from_client(client, candidate_id, primary_score=None):
    cand_details = client.get(f"/api/v1/candidates/{candidate_id}").json()
    return compute_candidate_digest(
        candidate_id=cand_details["candidate_id"],
        family=cand_details["family"],
        split_sha256=cand_details["split_sha256"],
        config_sha256=cand_details.get("config_sha256", "default_config"),
        primary_metric=cand_details["primary_metric"],
        primary_score=primary_score if primary_score is not None else cand_details["summary"][0]["mean"],
        model_seed=cand_details["model_seed"],
    ), cand_details


def test_concurrent_contenders_yields_one_selection(tmp_path):
    repo, app, _, _, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    ranked = leaderboard["ranked"]
    assert len(ranked) >= 2
    cand1 = ranked[0]
    cand2 = ranked[1]

    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        digest1, _ = candidate_digest_from_client(client, cand1["candidate_id"], cand1["mean"])
        digest2, _ = candidate_digest_from_client(client, cand2["candidate_id"], cand2["mean"])

        barrier = Barrier(2)
        results = []

        def contender(candidate_id, digest):
            barrier.wait()
            res = client.post(
                f"/api/v1/experiments/{experiment_id}/finalization",
                json={"candidate_id": candidate_id, "expected_digest": digest, "acknowledge": True},
            )
            results.append(res.status_code)

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(contender, cand1["candidate_id"], digest1)
            f2 = executor.submit(contender, cand2["candidate_id"], digest2)
            f1.result()
            f2.result()

        # Exactly one selection wins (202), and the other fails with 409
        assert sorted(results) == [202, 409]


def test_duplicate_reservation_is_idempotent(tmp_path):
    repo, app, _, _, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    cand = leaderboard["ranked"][0]
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        digest, _ = candidate_digest_from_client(client, cand["candidate_id"], cand["mean"])
        res1 = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
        )
        assert res1.status_code == 202
        body1 = res1.json()

        res2 = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
        )
        assert res2.status_code == 202
        body2 = res2.json()

        assert body1["finalization_id"] == body2["finalization_id"]
        assert body1["candidate_id"] == body2["candidate_id"]


def test_changed_candidate_or_digest_denied(tmp_path):
    repo, app, _, _, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    cand1 = leaderboard["ranked"][0]
    cand2 = leaderboard["ranked"][1]

    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        digest1, _ = candidate_digest_from_client(client, cand1["candidate_id"], cand1["mean"])
        digest2, _ = candidate_digest_from_client(client, cand2["candidate_id"], cand2["mean"])

        # Mismatched digest is rejected
        bad_digest = "0" * 64
        res_bad = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand1["candidate_id"], "expected_digest": bad_digest, "acknowledge": True},
        )
        assert res_bad.status_code == 409

        # Valid reservation
        res_ok = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand1["candidate_id"], "expected_digest": digest1, "acknowledge": True},
        )
        assert res_ok.status_code == 202

        # Changed candidate after reservation is denied with 409
        res_changed = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand2["candidate_id"], "expected_digest": digest2, "acknowledge": True},
        )
        assert res_changed.status_code == 409


def test_unrankable_or_ineligible_candidate_rejected(tmp_path):
    repo, app, _, _, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    # Insert an unrankable / failed candidate into candidates table
    bad_cand_id = "00000000-0000-0000-0000-000000000099"
    with repo._connection() as db:
        job_id = db.execute("SELECT job_id FROM jobs WHERE experiment_id=?", (experiment_id,)).fetchone()[0]
        db.execute(
            "INSERT INTO candidates(candidate_id,experiment_id,job_id,attempt_id,family,status,config_sha256,bundle_path,bundle_sha256,mlflow_run_id) "
            "VALUES(?,?,?,?,?,'failed','hash','path','bundle_hash','run_id')",
            (bad_cand_id, experiment_id, job_id, "attempt-1", "dummy"),
        )

    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        res = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": bad_cand_id, "expected_digest": "a" * 64, "acknowledge": True},
        )
        assert res.status_code in (409, 422)


def test_missing_acknowledgement_rejected(tmp_path):
    repo, app, _, _, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    cand = leaderboard["ranked"][0]
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        res = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand["candidate_id"], "expected_digest": "a" * 64, "acknowledge": False},
        )
        assert res.status_code == 422


def test_scope_lock_denies_competing_experiment_same_scope(tmp_path):
    repo, app, dataset_id, config_id, experiment_id1, scope_id, leaderboard = run_cv_experiment(tmp_path)
    cand = leaderboard["ranked"][0]

    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        digest, _ = candidate_digest_from_client(client, cand["candidate_id"], cand["mean"])
        # Reserve on experiment 1
        res1 = client.post(
            f"/api/v1/experiments/{experiment_id1}/finalization",
            json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
        )
        assert res1.status_code == 202

        # Create experiment 2 with different model seed on the same configuration (same evaluation scope)
        exp2_res = client.post("/api/v1/experiments", json={"configuration_id": config_id, "model_seed": 999})
        assert exp2_res.status_code == 201
        experiment_id2 = exp2_res.json()["experiment_id"]

        # Run CV for experiment 2
        client.post(f"/api/v1/experiments/{experiment_id2}/jobs", json={"idempotency_key": "cv-job-2"})
        worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
        worker.run_once()

        lb2 = client.get(f"/api/v1/experiments/{experiment_id2}/leaderboard").json()
        cand2 = lb2["ranked"][0]
        digest2, _ = candidate_digest_from_client(client, cand2["candidate_id"], cand2["mean"])

        # Attempt to reserve finalization on experiment 2 is denied with 409
        res2 = client.post(
            f"/api/v1/experiments/{experiment_id2}/finalization",
            json={"candidate_id": cand2["candidate_id"], "expected_digest": digest2, "acknowledge": True},
        )
        assert res2.status_code == 409


def test_pre_start_failure_retry_requires_dead_child(tmp_path):
    repo, app, _, _, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    cand = leaderboard["ranked"][0]

    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        digest, _ = candidate_digest_from_client(client, cand["candidate_id"], cand["mean"])
        res = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
        )
        assert res.status_code == 202
        finalization_id = res.json()["finalization_id"]

        # Simulate a claimed job with live child process PID (current process is alive)
        with repo._transaction() as db:
            row = db.execute("SELECT refit_job_id FROM finalizations WHERE finalization_id=?", (finalization_id,)).fetchone()
            job_id = row[0]
            attempt_id = "attempt-pre-start-1"
            db.execute("UPDATE jobs SET state='running', attempt_id=?, owner_token='worker-1', owner_generation=1 WHERE job_id=?", (attempt_id, job_id))
            db.execute("INSERT INTO worker_processes(attempt_id, supervisor_pid, supervisor_created, child_pid, child_created) VALUES(?, ?, ?, ?, ?)",
                       (attempt_id, os.getpid(), 1000.0, os.getpid(), 1000.0))

        # Retry should fail with 409 because child is still alive
        retry_fail = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization/retry",
            json={"idempotency_key": "retry-1"},
        )
        assert retry_fail.status_code == 409

        # Now mark the child as dead (PID 99999999 or dead status and failed job)
        with repo._transaction() as db:
            db.execute("UPDATE worker_processes SET child_pid=99999999, child_created=1.0 WHERE attempt_id=?", (attempt_id,))
            db.execute("UPDATE jobs SET state='failed' WHERE job_id=?", (job_id,))
            db.execute("UPDATE finalizations SET state='retryable_pre_evaluation' WHERE finalization_id=?", (finalization_id,))

        # Retry now succeeds!
        retry_ok = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization/retry",
            json={"idempotency_key": "retry-1"},
        )
        assert retry_ok.status_code == 202
        new_job_id = retry_ok.json()["job_id"]
        assert new_job_id != job_id


def test_durable_evaluation_start_issues_capability_and_makes_retry_impossible(tmp_path):
    repo, app, _, _, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    cand = leaderboard["ranked"][0]

    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        digest, _ = candidate_digest_from_client(client, cand["candidate_id"], cand["mean"])
        res = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
        )
        assert res.status_code == 202
        finalization_id = res.json()["finalization_id"]

        # Capability cannot be constructed with empty or invalid token
        with pytest.raises(PermissionError):
            HoldoutCapability(scope_id=scope_id, holdout_path="path/to/holdout.csv", _auth_token="")

        # Start evaluation durably
        capability = start_evaluation(repo, finalization_id, expected_digest=digest)
        assert isinstance(capability, HoldoutCapability)
        assert capability.scope_id == scope_id

        # Database state is now 'evaluating' and scope_lock_markers is 'consumed'
        with repo._connection() as db:
            fin = db.execute("SELECT state, evaluation_started_at FROM finalizations WHERE finalization_id=?", (finalization_id,)).fetchone()
            assert fin["state"] == "evaluating"
            assert fin["evaluation_started_at"] is not None

            marker = db.execute("SELECT state FROM scope_lock_markers WHERE scope_id=?", (scope_id,)).fetchone()
            assert marker["state"] == "consumed"

        # Any retry is now IMPOSSIBLE and must return 409
        retry_blocked = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization/retry",
            json={"idempotency_key": "retry-after-start"},
        )
        assert retry_blocked.status_code == 409

        # Even if child crashes or fails post-start, retry is permanently locked
        with repo._transaction() as db:
            db.execute("UPDATE finalizations SET state='locked_failed' WHERE finalization_id=?", (finalization_id,))

        retry_locked = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization/retry",
            json={"idempotency_key": "retry-after-crash"},
        )
        assert retry_locked.status_code == 409

        # Cannot start evaluation a second time
        with pytest.raises(ConflictError):
            start_evaluation(repo, finalization_id, expected_digest=digest)


def test_fixture_evaluator_and_poison_holdout_isolation(tmp_path):
    repo, app, _, _, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    cand = leaderboard["ranked"][0]

    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        digest, _ = candidate_digest_from_client(client, cand["candidate_id"], cand["mean"])
        res = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand["candidate_id"], "expected_digest": digest, "acknowledge": True},
        )
        finalization_id = res.json()["finalization_id"]

        capability = start_evaluation(repo, finalization_id, expected_digest=digest)

        # Fixture evaluation succeeds with valid capability
        final_result = fixture_evaluate_holdout(repo, capability)
        assert final_result.scope_id == scope_id
        assert final_result.candidate_id == cand["candidate_id"]

        # GET /api/v1/experiments/{id}/finalization reports evaluated state and metrics
        get_res = client.get(f"/api/v1/experiments/{experiment_id}/finalization")
        assert get_res.status_code == 200
        get_body = get_res.json()
        assert get_body["state"] == "evaluated"
        assert get_body["candidate_id"] == cand["candidate_id"]
        assert "evaluation_started" in [a["event"] for a in get_body["audit"]]


def test_consumed_tombstone_prevents_new_reservation(tmp_path):
    repo, app, dataset_id, config_id, experiment_id, scope_id, leaderboard = run_cv_experiment(tmp_path)
    cand = leaderboard["ranked"][0]

    # Insert a tombstone for this scope with state 'consumed'
    from apps.api.services.retention import _key, _scope_hmac
    key = _key(repo, create=True)
    with repo._connection() as db:
        scope = db.execute("SELECT canonical_sha256, target, task FROM evaluation_scopes WHERE scope_id=?", (scope_id,)).fetchone()
    hmac_digest = _scope_hmac(key, scope["canonical_sha256"], scope["target"], scope["task"])
    with repo._transaction() as db:
        db.execute("INSERT INTO scope_tombstones(scope_hmac, version, state) VALUES(?, 1, 'consumed')", (hmac_digest,))

    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        res = client.post(
            f"/api/v1/experiments/{experiment_id}/finalization",
            json={"candidate_id": cand["candidate_id"], "expected_digest": "a" * 64, "acknowledge": True},
        )
        assert res.status_code == 409
