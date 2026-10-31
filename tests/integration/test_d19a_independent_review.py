"""Safety regressions found during the independent D19a review."""

import json

import pytest

from apps.api.database.repository import ConflictError
from apps.api.workers.supervisor import WorkerSupervisor, _Outcome
from tests.integration.test_finalization import _run_fixture_cv
from tests.integration.test_review_finalization_regressions import claim_finalization, reservation
from apps.api.services.finalization import reserve
from ml.evaluation.finalize import stage_final_bundle


def test_stale_finalization_supervisor_cannot_release_new_lease(reservation):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    worker = WorkerSupervisor(repo, workload="finalization", limits=reservation[2].limits)
    worker.owner = owner
    with repo._transaction() as db:
        db.execute("UPDATE worker_lease SET generation=generation+1, owner_token='new-owner' WHERE singleton=1")
    with pytest.raises(ConflictError):
        worker._publish(claimed, claimed.attempt_id, _Outcome("failed", "failed", "child_exited"))
    with repo._connection() as db:
        lease = db.execute("SELECT owner_token,active_job_id FROM worker_lease WHERE singleton=1").fetchone()
    assert tuple(lease) == ("new-owner", job.job_id)
    assert repo.get_finalization(selection.scope_id)["state"] == "refitting"


def test_retry_rejects_altered_stored_plan(tmp_path):
    repo, prepared, _, candidate, digest, scope = _run_fixture_cv(tmp_path)
    selection, job = reserve(repo, prepared.experiment_id, candidate, digest, acknowledge=True)
    repo.request_cancel(job.job_id)
    with repo._transaction() as db:
        plan = json.loads(db.execute("SELECT plan_json FROM finalizations WHERE scope_id=?", (scope,)).fetchone()[0])
        plan["selection"]["resource_limits"]["compute_threads"] = 999
        db.execute("UPDATE finalizations SET plan_json=? WHERE scope_id=?", (json.dumps(plan), scope))
    with pytest.raises(ConflictError):
        repo.retry_pre_evaluation(selection.finalization_id, "altered-plan")


def test_cancel_before_publication_durably_locks(reservation):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    with repo._transaction() as db:
        db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now',attempt_id=? WHERE finalization_id=?",
                   (claimed.attempt_id, selection.finalization_id))
    repo.request_cancel(job.job_id)
    result = repo.publish_finalization(owner, job.job_id, claimed.attempt_id, selection.finalization_id,
                                       "unused", "unused")
    assert result.state == "cancelled"
    assert repo.get_finalization(selection.scope_id)["state"] == "locked_failed"
    assert repo.reconcile_worker()


def _valid_bundle_payload(selection):
    return {
        "finalization_id": selection.finalization_id,
        "scope_id": selection.scope_id,
        "candidate_id": selection.candidate_id,
        "candidate_digest": selection.candidate_digest,
        "task": selection.task.value,
        "primary_metric": selection.primary_metric,
        "evaluation_seconds": 0.05,
        "evaluated_at": "2026-10-31T00:00:00+00:00",
        "metrics": {"task": selection.task.value, "metrics": []},
    }


def test_recovery_in_live_supervisor_releases_lease(reservation):
    repo, _, run, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    with repo._transaction() as db:
        db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now',attempt_id=? WHERE finalization_id=?",
                   (claimed.attempt_id, selection.finalization_id))
    stage_final_bundle(repo.path.parent, selection.finalization_id, selection.scope_id,
                       claimed.attempt_id, selection.candidate_id, _valid_bundle_payload(selection))
    worker = WorkerSupervisor(repo, workload="finalization", limits=run.limits)
    worker.owner = owner
    result = worker._publish(claimed, claimed.attempt_id, _Outcome("failed", "failed", "child_exited"))
    assert result.state == "completed"
    with repo._connection() as db:
        assert db.execute("SELECT active_job_id FROM worker_lease WHERE singleton=1").fetchone()[0] is None


def test_production_worker_claims_finalization(reservation, monkeypatch):
    repo, _, run, _, job, _ = reservation
    with repo._transaction() as db:
        db.execute("UPDATE jobs SET state='cancelled' WHERE kind='cv' AND state='queued'")
    worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, claim_all=True, limits=run.limits)
    monkeypatch.setattr(worker, "_execute", lambda claimed: claimed)
    assert worker.run_once().job_id == job.job_id


def test_normal_stored_worker_executes_finalization_in_spawned_child(tmp_path):
    repo, prepared, run, candidate, digest, scope = _run_fixture_cv(tmp_path)
    selection, job = reserve(repo, prepared.experiment_id, candidate, digest, acknowledge=True)
    worker = WorkerSupervisor(repo, workload="classical", load_from_storage=True, claim_all=True, limits=run.limits)
    assert worker.run_once().state == "completed"
    assert worker.last_child_pid is not None
    assert repo.get_finalization(scope)["state"] == "evaluated"
    with repo._connection() as db:
        reads = db.execute("SELECT count(*) FROM audit_events WHERE entity_id=? AND event='holdout_read_authorized'",
                           (selection.finalization_id,)).fetchone()[0]
    assert reads == 1


def test_recovery_rejects_bundle_with_wrong_candidate(reservation):
    repo, _, _, selection, job, _ = reservation
    _, claimed = claim_finalization(repo, job)
    with repo._transaction() as db:
        db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now',attempt_id=? WHERE finalization_id=?",
                   (claimed.attempt_id, selection.finalization_id))
    stage_final_bundle(repo.path.parent, selection.finalization_id, selection.scope_id,
                       claimed.attempt_id, "wrong-candidate", {"ok": True})
    with repo._transaction() as db:
        fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (selection.finalization_id,)).fetchone()
        assert not repo._try_recover_staged_final_bundle(db, fin, claimed.attempt_id, "now")
    assert repo.get_finalization(selection.scope_id)["state"] == "evaluating"


def test_recovery_rejects_bundle_with_invalid_result_payload(reservation):
    repo, _, _, selection, job, _ = reservation
    _, claimed = claim_finalization(repo, job)
    with repo._transaction() as db:
        db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now',attempt_id=? WHERE finalization_id=?",
                   (claimed.attempt_id, selection.finalization_id))
    stage_final_bundle(repo.path.parent, selection.finalization_id, selection.scope_id,
                       claimed.attempt_id, selection.candidate_id, {"invalid_payload": True})
    with repo._transaction() as db:
        fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (selection.finalization_id,)).fetchone()
        assert not repo._try_recover_staged_final_bundle(db, fin, claimed.attempt_id, "now")
    assert repo.get_finalization(selection.scope_id)["state"] == "evaluating"


def test_publish_finalization_rejects_invalid_result_payload(reservation):
    repo, _, _, selection, job, _ = reservation
    owner, claimed = claim_finalization(repo, job)
    with repo._transaction() as db:
        db.execute("UPDATE finalizations SET state='evaluating',evaluation_started_at='now',attempt_id=? WHERE finalization_id=?",
                   (claimed.attempt_id, selection.finalization_id))
    bundle_path, bundle_sha = stage_final_bundle(
        repo.path.parent, selection.finalization_id, selection.scope_id,
        claimed.attempt_id, selection.candidate_id, {"corrupted_schema": 123},
    )
    with pytest.raises(ValueError, match="payload failed schema or identity validation"):
        repo.publish_finalization(
            owner, job.job_id, claimed.attempt_id, selection.finalization_id,
            str(bundle_path), bundle_sha,
        )
