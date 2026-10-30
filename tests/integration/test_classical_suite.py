"""Classical suite integration with frozen, training-only input."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from time import monotonic, sleep

import pytest

from apps.api.database.repository import JobRepository
from apps.api.services.jobs import prepare_experiment
from apps.api.workers.supervisor import WorkerSupervisor
from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, RunConfig, TaskType
from ml.data.csv_io import validate_csv
from ml.models.classical import classical_families
from ml.training.classical_suite import SuiteCancelled, evaluate_classical_suite


def prepared_fixture(tmp_path, task):
    path = tmp_path / "input.csv"
    path.write_text("x,category,target\n" + "\n".join(
        f"{i},group-{i % 3},{'yes' if i % 2 else 'no'}" if task == TaskType.BINARY_CLASSIFICATION
        else f"{i},group-{i % 3},{i * 1.5}" for i in range(100)), encoding="utf-8")
    config = DatasetConfig(task, "target", FeatureSchema((
        FeatureColumn("x", FeatureType.NUMERIC), FeatureColumn("category", FeatureType.CATEGORICAL),
    )), positive_label="yes" if task == TaskType.BINARY_CLASSIFICATION else None)
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    run = RunConfig()
    prepared = prepare_experiment(repo, validate_csv(path, run.limits), config, run)
    job = repo.enqueue_cv(prepared.experiment_id, "suite")
    return repo, prepared, run, job


@pytest.mark.parametrize("task", list(TaskType))
def test_spawned_worker_runs_four_families_on_frozen_training_folds(tmp_path, task):
    repo, prepared, run, job = prepared_fixture(tmp_path, task)
    worker = WorkerSupervisor(repo, workload="classical", classical_input=(prepared, run),
                              heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
    final = worker.run_once()
    assert final.job_id == job.job_id and final.state == "completed"
    assert worker.last_child_pid != os.getpid() and worker.last_result["pid"] == worker.last_child_pid
    candidates = worker.last_result["candidates"]
    assert tuple(c["family"] for c in candidates) == classical_families(task)
    for candidate in candidates:
        assert candidate["eligible"] and candidate["split_sha256"] == prepared.context.split.manifest_sha256
        assert len(candidate["folds"]) == 5
        for fold, frozen in zip(candidate["folds"], prepared.context.split.folds, strict=True):
            assert fold["status"] == "completed" and tuple(fold["validation_row_ids"]) == frozen.validation_row_ids
            assert set(fold["validation_row_ids"]).isdisjoint(prepared.context.split.holdout_row_ids)
            assert fold["fit_seconds"] >= 0 and fold["score_seconds"] >= 0


def test_suite_cancels_between_candidates_without_returning_partial_results(tmp_path):
    _, prepared, run, _ = prepared_fixture(tmp_path, TaskType.REGRESSION)
    event = threading.Event()
    seen = []
    def after_candidate(result):
        seen.append(result.family)
        event.set()
    with pytest.raises(SuiteCancelled):
        evaluate_classical_suite(prepared.context, run, cancelled=event.is_set, on_candidate=after_candidate)
    assert seen == ["dummy"]


def test_failed_family_is_visible_and_ineligible_without_losing_other_families(tmp_path, monkeypatch):
    from ml.training import cv
    _, prepared, run, _ = prepared_fixture(tmp_path, TaskType.REGRESSION)
    original = cv.make_classical
    def fail_fit(task, family, seed):
        model = original(task, family, seed)
        if family == "ridge":
            model.fit = lambda *args: (_ for _ in ()).throw(ValueError("private data"))
        return model
    monkeypatch.setattr(cv, "make_classical", fail_fit)
    results = evaluate_classical_suite(prepared.context, run)
    assert len(results) == 4
    failed = results[1]
    assert failed.family == "ridge" and not failed.eligible
    assert all(f.status == "failed" and f.error == "fold_execution_failed" for f in failed.folds)
    assert all(result.eligible for result in (results[0], *results[2:]))


def test_worker_rejects_input_for_a_different_frozen_experiment(tmp_path):
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    worker = WorkerSupervisor(repo, workload="classical", classical_input=(prepared, replace(run, model_seed=43)))
    assert worker.run_once().state == "failed"
    assert worker.last_child_pid is None and worker.last_result is None
    assert repo.get_job(job.job_id).safe_error == "supervisor_error"


def test_all_families_fit_fresh_preprocessors_only_on_frozen_fit_rows(tmp_path, monkeypatch):
    from ml.preprocessing.factory import FoldPreprocessor
    _, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    context = prepared.context
    original = FoldPreprocessor.fit
    seen = []
    def inspect(self, view):
        frozen = context.split.folds[len(seen) % len(context.split.folds)]
        assert view.row_ids == frozen.fit_row_ids
        assert set(view.row_ids).isdisjoint(context.split.holdout_row_ids)
        seen.append(self)
        return original(self, view)
    monkeypatch.setattr(FoldPreprocessor, "fit", inspect)
    results = evaluate_classical_suite(context, run)
    assert all(result.eligible for result in results)
    assert len(seen) == 20 and len({id(instance) for instance in seen}) == 20


def test_running_classical_job_blocks_another_worker_and_can_be_cancelled(tmp_path):
    repo, prepared, run, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    second = repo.enqueue_cv(prepared.experiment_id, "second")
    first = WorkerSupervisor(repo, workload="classical", classical_input=(prepared, run),
                             heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
    rival = WorkerSupervisor(repo, workload="probe", workload_options={"work_seconds": 0.1},
                             owner_token="rival")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first.run_once)
        deadline = monotonic() + 30
        while repo.get_job(job.job_id).stage != "fitting" and monotonic() < deadline:
            assert not future.done(), "Classical worker finished before reaching fitting"
            sleep(0.02)
        assert repo.get_job(job.job_id).stage == "fitting"
        assert rival.run_once() is None and rival.last_child_pid is None
        assert repo.get_job(second.job_id).state == "queued"
        repo.request_cancel(job.job_id)
        assert future.result(timeout=15).state == "cancelled"
    assert first.last_result is None
