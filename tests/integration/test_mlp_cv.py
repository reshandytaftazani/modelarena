"""Acceptance tests for both-task MLP CV and curve artifact publication (Day 17).

Verifies D17a and D17b:
- Unified binary/regression registry with MLP.
- 5-fold CV on feasible fixtures for both tasks with frozen fold identities.
- Guarded float32 conversion and numerically stable sigmoid probabilities.
- Zero holdout access and poison-holdout proof.
- Classical candidate comparison, undefined metric handling and failure policies.
- MLflow/manifest persistence: per-fold curves, best epochs, runtime, caps, inner-fit counts.
- Separate curves per fold (no cross-fold concoctions), finite chart JSON and failure diagnostics.
"""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from time import perf_counter, sleep
from types import SimpleNamespace

from mlflow.tracking import MlflowClient
import numpy as np
import psutil
import pytest
import torch

from apps.api.database.repository import WorkerIdentity

from ml.contracts import (
    EpochLoss,
    MetricValue,
    ResourceLimits,
    RunConfig,
    TaskType,
    TrainingContext,
)
from ml.models import classical_families, model_families
from ml.preprocessing.factory import FoldPreprocessor
from ml.tracking.publication import stage_attempt
from ml.training.classical_suite import SuiteCancelled
from ml.training.cv import compare_candidates, evaluate_candidate, summarize_candidate
from tests.integration.test_baseline_cv import context_for
from tests.integration.test_classical_suite import prepared_fixture
from apps.api.workers.supervisor import WorkerSupervisor


@pytest.mark.parametrize("task", list(TaskType))
def test_spawned_worker_publishes_mlp_with_frozen_epoch_cap(tmp_path, task):
    run = RunConfig(limits=ResourceLimits(compute_threads=2, max_epochs=2))
    repo, prepared, _, job = prepared_fixture(tmp_path, task, run)
    worker = WorkerSupervisor(repo, workload="classical", classical_input=(prepared, run),
                              limits=run.limits, heartbeat_seconds=0.2, sample_seconds=0.1,
                              poll_seconds=0.05)
    assert worker.run_once().state == "completed"
    results = repo.read_published_results(job.job_id)
    candidate = next(c for c in results["candidates"] if c["family"] == "mlp")
    assert candidate["eligible"]
    assert [f["validation_row_ids"] for f in candidate["folds"]] == [
        list(f.validation_row_ids) for f in prepared.context.split.folds]
    assert all(f["epochs_trained"] == len(f["loss_curve"]) == 2 for f in candidate["folds"])
    manifest_candidate = next(c for c in results["manifest"]["candidates"] if c["family"] == "mlp")
    assert manifest_candidate["model_parameters"]["max_epochs"] == 2
    client = MlflowClient(tracking_uri="sqlite:///" + (tmp_path / "mlflow.sqlite3").as_posix())
    run_id = results["manifest"]["mlflow_run_ids"]["candidates"]["mlp"]["run_id"]
    assert client.get_run(run_id).data.params["max_epochs"] == "2"


def test_completed_mlp_fold_requires_coherent_finite_curve(tmp_path):
    context = context_for(tmp_path, TaskType.REGRESSION)
    candidate = evaluate_candidate(context, RunConfig(family="mlp", limits=ResourceLimits(max_epochs=2)))
    fold = candidate.folds[0]
    assert fold.epochs_trained == 2
    with pytest.raises(ValueError):
        replace(fold, loss_curve=fold.loss_curve[:1])
    with pytest.raises(ValueError):
        replace(fold, loss_curve=(EpochLoss(1, 1.0, None), EpochLoss(3, 1.0, 1.0)))


def test_mlp_positive_label_mapping_and_fold_scores(tmp_path):
    from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType
    from ml.data.csv_io import validate_csv
    from ml.data.validation import configure_dataset
    from ml.data.splits import create_split, training_from_manifest
    from ml.evaluation.metrics import score_predictions

    path = tmp_path / "labels.csv"
    path.write_text("x,target\n" + "\n".join(f"{i},{'a' if i % 2 else 'z'}" for i in range(100)))
    config = DatasetConfig(TaskType.BINARY_CLASSIFICATION, "target",
                           FeatureSchema((FeatureColumn("x", FeatureType.NUMERIC),)), positive_label="a")
    data = validate_csv(path, ResourceLimits())
    split = create_split(data, config, RunConfig())
    context = TrainingContext(training_from_manifest(configure_dataset(data, config), split), split)
    candidate = evaluate_candidate(context, RunConfig(family="mlp", limits=ResourceLimits(max_epochs=2)))
    assert context.training.label_mapping == (("z", 0), ("a", 1))
    for fold in candidate.folds:
        points = [p for p in candidate.diagnostics.points if p.fold_index == fold.fold_index]
        assert all(p.truth == p.row_id % 2 for p in points)
        expected = score_predictions(TaskType.BINARY_CLASSIFICATION, [p.truth for p in points],
                                     [p.prediction for p in points], [p.probability for p in points])
        assert expected == fold.metrics


def test_inner_fit_is_exact_data_used_for_gradient_loader(tmp_path, monkeypatch):
    from ml.training import mlp_loop
    from ml.training.mlp_loop import subset_view
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION)
    fold = context.split.folds[0]
    original = mlp_loop.create_dataloader
    seen = []
    def capture(features, targets, **kwargs):
        seen.append((features.copy(), targets.copy(), kwargs["shuffle"]))
        return original(features, targets, **kwargs)
    monkeypatch.setattr(mlp_loop, "create_dataloader", capture)
    result = mlp_loop.train_mlp_fold(context, 0, RunConfig(family="mlp", limits=ResourceLimits(max_epochs=2)))
    fit = subset_view(context.training, fold.inner.fit_row_ids)
    val = subset_view(context.training, fold.inner.validation_row_ids)
    assert len(seen) == 2 and seen[0][2] and not seen[1][2]
    np.testing.assert_array_equal(seen[0][0], result.preprocessor.transform(fit, dtype=np.float32))
    np.testing.assert_array_equal(seen[0][1], np.asarray(fit.targets, dtype=np.float32))
    np.testing.assert_array_equal(seen[1][0], result.preprocessor.transform(val, dtype=np.float32))
    np.testing.assert_array_equal(seen[1][1], np.asarray(val.targets, dtype=np.float32))


def test_scoring_failure_retains_completed_fit_curve_but_cannot_rank(tmp_path, monkeypatch):
    context = context_for(tmp_path, TaskType.REGRESSION)
    validation_ids = set(context.split.folds[0].validation_row_ids)
    original = FoldPreprocessor.transform
    def fail_outer(self, view, **kwargs):
        if set(view.row_ids) == validation_ids:
            raise ValueError("private scoring failure")
        return original(self, view, **kwargs)
    monkeypatch.setattr(FoldPreprocessor, "transform", fail_outer)
    candidate = evaluate_candidate(context, RunConfig(family="mlp", limits=ResourceLimits(max_epochs=2)))
    fold = candidate.folds[0]
    assert not candidate.eligible and fold.status == "failed" and fold.error == "fold_execution_failed"
    assert fold.epochs_trained == len(fold.loss_curve) == 2
    assert fold.inner_fit_row_count == len(context.split.folds[0].inner.fit_row_ids)


def test_mlp_metric_tracking_failure_blocks_publication(tmp_path, monkeypatch):
    run = RunConfig(limits=ResourceLimits(compute_threads=2, max_epochs=2))
    repo, prepared, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION, run)
    original = MlflowClient.log_metric
    def fail_curve(self, run_id, key, value, *args, **kwargs):
        if key == "train_loss":
            raise OSError("simulated tracking failure")
        return original(self, run_id, key, value, *args, **kwargs)
    monkeypatch.setattr(MlflowClient, "log_metric", fail_curve)
    worker = WorkerSupervisor(repo, workload="classical", classical_input=(prepared, run), limits=run.limits,
                              heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
    assert worker.run_once().state == "failed"
    assert worker.last_result is None
    with pytest.raises(LookupError):
        repo.read_published_results(job.job_id)
    with repo._connection() as db:
        assert db.execute("SELECT count(*) FROM candidates WHERE job_id=?", (job.job_id,)).fetchone()[0] == 0


def test_overdue_mlp_fold_is_not_ranked(tmp_path, monkeypatch):
    from ml.training import cv
    context = context_for(tmp_path, TaskType.REGRESSION)
    original = cv.train_mlp_fold
    def overdue(*args, **kwargs):
        return replace(original(*args, **kwargs), fit_seconds=2.0)
    monkeypatch.setattr(cv, "train_mlp_fold", overdue)
    candidate = evaluate_candidate(context, RunConfig(family="mlp", limits=ResourceLimits(max_epochs=1, mlp_fold_seconds=1)))
    assert not candidate.eligible
    assert all(f.status == "timeout" and f.metrics is None for f in candidate.folds)


# --- D17a: Registry & CV Scoring Tests ---


def test_model_registry_includes_mlp_for_both_tasks():
    """Verify MLP is registered in the unified registry alongside classical models."""
    bin_families = model_families(TaskType.BINARY_CLASSIFICATION)
    reg_families = model_families(TaskType.REGRESSION)

    assert "mlp" in bin_families
    assert "mlp" in reg_families

    # Classical families remain unaffected
    assert bin_families == ("dummy", "logistic", "random_forest", "hist_gradient_boosting", "mlp")
    assert reg_families == ("dummy", "ridge", "random_forest", "hist_gradient_boosting", "mlp")
    assert classical_families(TaskType.BINARY_CLASSIFICATION) == ("dummy", "logistic", "random_forest", "hist_gradient_boosting")
    assert classical_families(TaskType.REGRESSION) == ("dummy", "ridge", "random_forest", "hist_gradient_boosting")

    with pytest.raises(ValueError, match="Unsupported task"):
        model_families("unknown_task")  # type: ignore[arg-type]


def test_evaluate_candidate_rejects_unsupported_family_and_invalid_seed(tmp_path):
    """Verify strict validation of family name and model seed."""
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION)

    with pytest.raises(ValueError, match="Unsupported task or model family"):
        evaluate_candidate(context, RunConfig(family="nonexistent_model"))

    with pytest.raises(ValueError, match="unsigned 32-bit"):
        evaluate_candidate(context, RunConfig(family="mlp", model_seed=-1))


def test_binary_classification_mlp_cv_five_folds_and_sigmoid_scoring(tmp_path):
    """Verify 5-fold binary classification MLP CV, sigmoid probabilities, and OOF diagnostics."""
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION, positives=30)
    config = RunConfig(family="mlp", model_seed=42)

    result = evaluate_candidate(context, config)

    assert result.complete is True
    assert result.eligible is True
    assert result.family == "mlp"
    assert len(result.folds) == 5
    assert result.expected_fold_indices == (0, 1, 2, 3, 4)

    # Fold validation rows must strictly match frozen manifest partitions
    for fold, frozen in zip(result.folds, context.split.folds, strict=True):
        assert fold.status == "completed"
        assert fold.validation_row_ids == frozen.validation_row_ids
        assert set(fold.validation_row_ids).isdisjoint(context.split.holdout_row_ids)
        assert fold.metrics is not None
        assert fold.fit_seconds > 0.0
        assert fold.score_seconds >= 0.0

        # D17b fold attributes
        assert fold.best_epoch is not None and 1 <= fold.best_epoch <= 100
        assert fold.epochs_trained is not None and 1 <= fold.epochs_trained <= 100
        assert fold.best_epoch <= fold.epochs_trained
        assert fold.inner_fit_row_count == len(frozen.inner.fit_row_ids)
        assert fold.loss_curve is not None
        assert len(fold.loss_curve) == fold.epochs_trained
        for pt in fold.loss_curve:
            assert isinstance(pt, EpochLoss)
            assert math.isfinite(pt.train_loss)
            assert pt.val_loss is not None and math.isfinite(pt.val_loss)

    # Diagnostic points and probabilities via sigmoid
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.partition == "out_of_fold_training"
    assert len(diagnostics.points) == len(context.split.train_row_ids)
    for pt in diagnostics.points:
        assert pt.probability is not None
        assert 0.0 <= pt.probability <= 1.0
        assert pt.prediction in (0, 1)
        assert pt.prediction == int(pt.probability >= 0.5)

    # Binary metrics must all be valid and finite
    metrics = {m.name: m.value for m in diagnostics.metrics.metrics}
    for name in ("f1", "precision", "recall", "roc_auc", "average_precision"):
        assert name in metrics
        assert metrics[name] is not None
        assert math.isfinite(metrics[name])


def test_regression_mlp_cv_five_folds_and_raw_predictions(tmp_path):
    """Verify 5-fold regression MLP CV, raw predictions, and finite diagnostics."""
    context = context_for(tmp_path, TaskType.REGRESSION)
    config = RunConfig(family="mlp", model_seed=42)

    result = evaluate_candidate(context, config)

    assert result.complete is True
    assert result.eligible is True
    assert result.family == "mlp"
    assert len(result.folds) == 5

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.partition == "out_of_fold_training"
    assert len(diagnostics.points) == len(context.split.train_row_ids)
    for pt in diagnostics.points:
        assert pt.probability is None
        assert pt.residual is not None
        assert math.isfinite(pt.residual)
        assert pt.residual == pytest.approx(pt.truth - pt.prediction)

    metrics = {m.name: m.value for m in diagnostics.metrics.metrics}
    for name in ("mae", "rmse", "r2"):
        assert name in metrics
        assert metrics[name] is not None
        assert math.isfinite(metrics[name])


def test_guarded_float32_conversion_on_outer_validation(tmp_path, monkeypatch):
    """Verify that outer validation transform is requested as guarded np.float32."""
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION)
    transform_dtypes = []
    orig_transform = FoldPreprocessor.transform

    def spy_transform(self, data, *, dtype=np.float64):
        transform_dtypes.append(dtype)
        return orig_transform(self, data, dtype=dtype)

    monkeypatch.setattr(FoldPreprocessor, "transform", spy_transform)
    result = evaluate_candidate(context, RunConfig(family="mlp"))

    assert result.complete is True
    # Inner validation transforms and outer validation transforms must use np.float32
    assert all(d == np.float32 for d in transform_dtypes)
    # At least 5 inner validation transforms + 5 outer validation transforms = 10
    assert len(transform_dtypes) >= 10


def test_no_holdout_access_during_mlp_cv(tmp_path, monkeypatch):
    """Verify that neither preprocessing fit, inner splits, nor evaluation touch holdout data."""
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION)
    holdout_set = set(context.split.holdout_row_ids)

    fitted_ids = []
    transformed_ids = []
    orig_fit = FoldPreprocessor.fit
    orig_transform = FoldPreprocessor.transform

    def spy_fit(self, view):
        fitted_ids.append(set(view.row_ids))
        return orig_fit(self, view)

    monkeypatch.setattr(FoldPreprocessor, "fit", spy_fit)
    def spy_transform(self, view, **kwargs):
        transformed_ids.append(tuple(view.row_ids))
        return orig_transform(self, view, **kwargs)
    monkeypatch.setattr(FoldPreprocessor, "transform", spy_transform)
    result = evaluate_candidate(context, RunConfig(family="mlp"))

    assert result.complete is True
    # Verify no holdout row ID ever touched fit
    assert fitted_ids == [set(fold.inner.fit_row_ids) for fold in context.split.folds]
    assert all(seen.isdisjoint(holdout_set) for seen in fitted_ids)
    assert len(transformed_ids) == 3 * len(context.split.folds)
    for index, frozen in enumerate(context.split.folds):
        assert transformed_ids[3 * index:3 * index + 3] == [
            frozen.inner.fit_row_ids, frozen.inner.validation_row_ids, frozen.validation_row_ids]
        assert all(set(ids).isdisjoint(holdout_set) for ids in transformed_ids[3 * index:3 * index + 3])


def test_failed_fold_policy_matches_classical(tmp_path, monkeypatch):
    """Verify that a fold execution failure makes the candidate ineligible and unranked."""
    context = context_for(tmp_path, TaskType.REGRESSION)
    calls = 0
    orig_fit = FoldPreprocessor.fit

    def fail_on_fold_2(self, view):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated training breakdown")
        return orig_fit(self, view)

    monkeypatch.setattr(FoldPreprocessor, "fit", fail_on_fold_2)
    result = evaluate_candidate(context, RunConfig(family="mlp"))

    assert result.complete is False
    assert result.eligible is False
    assert result.folds[1].status == "failed"
    assert result.folds[1].error == "fold_execution_failed"
    assert result.folds[1].metrics is None

    # Summarize candidate must record incomplete_folds
    summary = summarize_candidate(result)
    assert any(s.reason == "incomplete_folds" for s in summary)

    # compare_candidates must relegate the candidate to unranked
    baseline = evaluate_candidate(context, RunConfig(family="dummy"))
    comparison = compare_candidates(context, (baseline, result))
    assert [entry.candidate.family for entry in comparison.ranked] == ["dummy"]
    assert comparison.unranked == (result,)


def test_undefined_primary_policy_matches_classical(tmp_path):
    """Verify that undefined primary metric in a completed fold invalidates ranking."""
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION)
    result = evaluate_candidate(context, RunConfig(family="mlp"))
    assert result.eligible is True

    # Mutate fold 0 primary metric to undefined
    fold0 = result.folds[0]
    mutated_bundle = replace(fold0.metrics, metrics=(
        MetricValue("f1", None, "undefined_zero_division"),
    ) + fold0.metrics.metrics[1:])
    mutated_candidate = replace(result, folds=(replace(fold0, metrics=mutated_bundle),) + result.folds[1:])

    assert mutated_candidate.eligible is False
    summary = summarize_candidate(mutated_candidate)
    f1_summary = next(s for s in summary if s.name == "f1")
    assert f1_summary.reason == "undefined_in_one_or_more_folds"
    assert f1_summary.mean is None

    baseline = evaluate_candidate(context, RunConfig(family="dummy"))
    comparison = compare_candidates(context, (baseline, mutated_candidate))
    assert comparison.unranked == (mutated_candidate,)


def test_comparison_ranks_mlp_alongside_classical(tmp_path):
    """Verify that complete MLP candidate ranks against classical baseline and models."""
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION)
    baseline = evaluate_candidate(context, RunConfig(family="dummy"))
    mlp_candidate = evaluate_candidate(context, RunConfig(family="mlp"))

    comparison = compare_candidates(context, (baseline, mlp_candidate))

    assert len(comparison.ranked) == 2
    assert not comparison.unranked
    assert {e.candidate.family for e in comparison.ranked} == {"dummy", "mlp"}

    mlp_entry = next(e for e in comparison.ranked if e.candidate.family == "mlp")
    assert mlp_entry.baseline_delta is not None
    assert math.isfinite(mlp_entry.mean)
    assert math.isfinite(mlp_entry.std)
    assert mlp_entry.mean_fit_seconds > 0.0


def test_cancellation_during_mlp_cv_raises_suite_cancelled(tmp_path):
    """Verify RunControl cancellation raises SuiteCancelled without partial results."""
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION)

    with pytest.raises(SuiteCancelled):
        evaluate_candidate(context, RunConfig(family="mlp"), cancelled=lambda: True)


# --- D17b: Tracking & Artifact Lineage Tests ---


def test_persistence_to_mlflow_and_manifest(tmp_path):
    """Verify D17b artifact persistence: per-fold curves, best epoch, caps, inner fit count."""
    context = context_for(tmp_path, TaskType.BINARY_CLASSIFICATION, positives=30)
    run = RunConfig(family="mlp", limits=ResourceLimits(compute_threads=2))
    candidate = evaluate_candidate(context, run)

    # Form tracking payload
    payload = {
        "pid": 12345,
        "workload": "classical",
        "candidates": [{**asdict(candidate), "eligible": candidate.eligible}],
        "comparison": {
            "ranked": [{"family": "mlp", "mean": candidate.summary[0].mean, "std": candidate.summary[0].std,
                        "mean_fit_seconds": 0.5, "baseline_delta": 0.1, "tied": False}],
            "unranked": [],
        },
        "observed_threadpools": [],
    }

    job = SimpleNamespace(
        job_id="test-job-id",
        attempt_id="test-attempt-id",
        experiment_id="test-exp-id",
    )
    dataset = {
        "canonical_sha256": context.split.dataset_sha256,
        "row_count": context.split.row_count,
    }

    staged = stage_attempt(tmp_path, job, context, run, dataset, payload, telemetry=None)

    bundle_dir = Path(staged["path"])
    results_path = bundle_dir / "results.json"
    manifest_path = bundle_dir / "manifest.json"

    assert results_path.is_file()
    assert manifest_path.is_file()

    # Verify finite chart JSON in results.json
    results_content = json.loads(results_path.read_text(encoding="utf-8"))
    c_out = results_content["candidates"][0]
    assert c_out["family"] == "mlp"
    assert len(c_out["folds"]) == 5
    for fold in c_out["folds"]:
        assert fold["best_epoch"] is not None
        assert fold["epochs_trained"] is not None
        assert fold["inner_fit_row_count"] > 0
        assert fold["fit_seconds"] > 0.0
        assert "loss_curve" in fold
        curve = fold["loss_curve"]
        assert len(curve) == fold["epochs_trained"] <= 100
        for pt in curve:
            assert isinstance(pt["epoch"], int)
            assert math.isfinite(pt["train_loss"])
            assert pt["val_loss"] is not None and math.isfinite(pt["val_loss"])

    # Verify manifest.json contains separate curves, caps, provenance and checksums
    manifest_content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_content["schema_version"] == 1
    assert manifest_content["scope"] == "training_cv_only"
    assert manifest_content["code"]["revision"]
    assert manifest_content["artifact_sha256"]["results.json"] == staged["results_sha256"]

    manifest_c = manifest_content["candidates"][0]
    assert manifest_c["family"] == "mlp"
    # Model parameters must reflect default caps/architecture
    params = manifest_c["model_parameters"]
    assert params["max_epochs"] == 100
    assert params["patience"] == 10
    assert params["lr"] == 1e-3
    assert params["batch_size"] == 32
    assert params["hidden_dims"] == [64, 32]

    # Verify per-fold summaries preserved in manifest
    assert len(manifest_c["folds"]) == 5
    for f_idx, fold in enumerate(manifest_c["folds"]):
        assert fold["fold_index"] == f_idx
        assert fold["best_epoch"] is not None
        assert fold["epochs_trained"] <= 100
        assert fold["inner_fit_row_count"] == len(context.split.folds[f_idx].inner.fit_row_ids)
        assert len(fold["loss_curve"]) == fold["epochs_trained"]

    # Verify MLflow tracking runs and separate per-fold loss curves
    client = MlflowClient(tracking_uri="sqlite:///" + (tmp_path / "mlflow.sqlite3").as_posix())
    experiments = client.search_experiments()
    runs = client.search_runs([e.experiment_id for e in experiments])

    own_runs = [r for r in runs if r.data.tags.get("modelarena.attempt_id") == "test-attempt-id"]
    # 1 parent + 1 candidate + 5 folds = 7 runs
    assert len(own_runs) == 7

    candidate_run = next(r for r in own_runs if r.data.tags.get("modelarena.kind") == "candidate")
    assert candidate_run.data.params["max_epochs"] == "100"
    assert candidate_run.data.params["patience"] == "10"
    assert candidate_run.data.params["lr"] == "0.001"
    assert candidate_run.data.params["hidden_dims"] == "64,32"

    fold_runs = [r for r in own_runs if r.data.tags.get("modelarena.kind") == "fold"]
    assert len(fold_runs) == 5

    for f_run in fold_runs:
        f_idx = int(f_run.data.tags["modelarena.fold_index"])
        assert "fit_seconds" in f_run.data.metrics
        assert "score_seconds" in f_run.data.metrics
        assert "best_epoch" in f_run.data.metrics
        assert "inner_fit_row_count" in f_run.data.metrics
        assert "epochs_trained" in f_run.data.metrics

        # Verify per-fold loss curve metric history
        train_history = client.get_metric_history(f_run.info.run_id, "train_loss")
        val_history = client.get_metric_history(f_run.info.run_id, "val_loss")

        assert len(train_history) == int(f_run.data.metrics["epochs_trained"])
        assert len(val_history) == int(f_run.data.metrics["epochs_trained"])
        assert len(train_history) <= 100

        # Steps must match epoch numbers (1, 2, ..., N)
        for expected_step, m in enumerate(train_history, start=1):
            assert m.step == expected_step
            assert math.isfinite(m.value)
        for expected_step, m in enumerate(val_history, start=1):
            assert m.step == expected_step
            assert math.isfinite(m.value)
        expected_curve = c_out["folds"][f_idx]["loss_curve"]
        assert manifest_c["folds"][f_idx]["loss_curve"] == expected_curve
        assert [m.value for m in train_history] == [p["train_loss"] for p in expected_curve]
        assert [m.value for m in val_history] == [p["val_loss"] for p in expected_curve]


def test_supervisor_hard_watchdog_terminates_stuck_mlp_child(tmp_path, monkeypatch):
    """Supervisor enforces MLP per-fold deadline without relying on child to return."""
    run = RunConfig(limits=ResourceLimits(compute_threads=2, max_epochs=2, mlp_fold_seconds=1, cv_job_seconds=30))
    repo, prepared, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION, run)
    monkeypatch.setenv("_MODELARENA_TEST_MLP_BLOCK_SECONDS", "6.0")
    worker = WorkerSupervisor(
        repo, workload="classical", classical_input=(prepared, run),
        limits=run.limits, heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05,
    )
    from concurrent.futures import ThreadPoolExecutor
    mlp_start_time = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_once)
        while not future.done():
            prog = repo.get_work_progress(job.job_id)
            if prog and prog.get("family") == "mlp" and mlp_start_time is None:
                mlp_start_time = perf_counter()
            sleep(0.02)
        final = future.result(timeout=30)
        mlp_duration = (perf_counter() - mlp_start_time) if mlp_start_time is not None else 0.0

    # 1. Supervisor must interrupt the child during the MLP fold well before the 6.0s native block completes
    assert mlp_start_time is not None, "MLP family must have been reached and tracked"
    assert 0.7 <= mlp_duration < 3.5, f"Supervisor timing out of expected range; took {mlp_duration:.2f}s"
    # 2. Owned child is stopped and dead
    assert worker.last_child_pid is not None
    assert worker.last_child_pid != os.getpid()
    assert not psutil.pid_exists(worker.last_child_pid)
    # 3. Honest terminal state
    assert final.state == "failed"
    assert final.stage == "failed"
    assert final.safe_error == "timeout"
    # 4. Fail-closed: no eligible candidate or partial result published
    assert worker.last_result is None
    with pytest.raises(LookupError):
        repo.read_published_results(job.job_id)
    with repo._connection() as db:
        assert db.execute("SELECT count(*) FROM candidates WHERE job_id=?", (job.job_id,)).fetchone()[0] == 0
    # 5. Worker slot can be recovered safely
    recovered = repo.claim_next(WorkerIdentity("supervisor-recovery", repo.lease_generation()))
    assert repo.get_job(job.job_id).state == "failed"


def test_mlp_fold_deadline_versus_completion_race(tmp_path):
    """An overdue fold payload arriving at completion must not claim success."""
    import multiprocessing
    run = RunConfig(limits=ResourceLimits(compute_threads=2, max_epochs=2, mlp_fold_seconds=1, cv_job_seconds=30))
    repo, prepared, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION, run)
    worker = WorkerSupervisor(
        repo, workload="classical", classical_input=(prepared, run),
        limits=run.limits, heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05,
    )
    claimed = repo.claim_next(worker.owner)
    assert claimed is not None
    # Simulate a result payload returning with an overdue MLP fold
    overdue_result = {
        "pid": 99999, "workload": "classical",
        "candidates": [{
            "family": "mlp", "model_seed": 42, "split_sha256": "fake",
            "primary_metric": "rmse", "direction": "minimize",
            "eligible": True,
            "folds": [{
                "fold_index": 0, "validation_row_ids": [1], "status": "completed",
                "metrics": None, "fit_seconds": 2.5, "score_seconds": 0.1,
            }],
        }],
    }
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    q.put({"kind": "progress", "stage": "fitting", "family": "mlp", "fold_index": 0,
           "finished_units": 20, "total_units": 25, "fold_finished": False})
    q.put({"kind": "progress", "stage": "fitting", "family": "mlp", "fold_index": 0,
           "finished_units": 21, "total_units": 25, "fold_finished": True})
    q.put({"kind": "result", "payload": overdue_result})

    class DummyProcess:
        pid = 12345
        exitcode = 0
        def is_alive(self):
            return False
        def join(self, timeout=None):
            pass

    outcome = worker._monitor(DummyProcess(), q, claimed, None)
    assert outcome.state == "failed"
    assert outcome.code == "timeout"
    assert worker.last_result is None

    final = worker._publish(claimed, claimed.attempt_id, outcome)
    assert final.state == "failed"
    assert final.safe_error == "timeout"
    with pytest.raises(LookupError):
        repo.read_published_results(claimed.job_id)
    with repo._connection() as db:
        assert db.execute("SELECT count(*) FROM candidates WHERE job_id=?", (claimed.job_id,)).fetchone()[0] == 0


def test_mlp_fold_deadline_versus_cancellation_race(tmp_path, monkeypatch):
    """Cancellation requested during an active MLP fold stops the child and marks cancelled."""
    run = RunConfig(limits=ResourceLimits(compute_threads=2, max_epochs=2, mlp_fold_seconds=10, cv_job_seconds=30))
    repo, prepared, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION, run)
    monkeypatch.setenv("_MODELARENA_TEST_MLP_BLOCK_SECONDS", "8.0")
    worker = WorkerSupervisor(
        repo, workload="classical", classical_input=(prepared, run),
        limits=run.limits, heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05,
    )
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_once)
        # Wait until job enters mlp fitting stage
        while not future.done():
            prog = repo.get_work_progress(job.job_id)
            if prog and prog.get("family") == "mlp":
                break
            sleep(0.02)
        # Request cancel while child is in fold
        repo.request_cancel(job.job_id)
        final = future.result(timeout=15)

    assert final.state == "cancelled"
    assert final.stage == "cancelled"
    assert worker.last_result is None
    assert worker.last_child_pid is not None
    assert not psutil.pid_exists(worker.last_child_pid)
    with pytest.raises(LookupError):
        repo.read_published_results(job.job_id)
    with repo._connection() as db:
        assert db.execute("SELECT count(*) FROM candidates WHERE job_id=?", (job.job_id,)).fetchone()[0] == 0
    # Verify slot is recovered
    recovered = repo.claim_next(WorkerIdentity("supervisor-cancel-recovery", repo.lease_generation()))
    assert repo.get_job(job.job_id).state == "cancelled"

