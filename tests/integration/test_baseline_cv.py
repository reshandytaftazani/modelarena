"""Real frozen-fold baselines, leakage sentinels and incomplete-result gates."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from ml.contracts import (
    DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits,
    RunConfig, TaskType, TrainingContext,
)
from ml.data.csv_io import validate_csv
from ml.data.splits import create_split, training_from_manifest
from ml.data.validation import configure_dataset
from ml.models.classical import make_classical
from ml.preprocessing.factory import FoldPreprocessor
from ml.training.cv import evaluate_candidate, summarize_candidate


def context_for(tmp_path, task=TaskType.BINARY_CLASSIFICATION, positives=20):
    path = tmp_path / "fixture.csv"
    labels = ["yes" if i < positives else "no" for i in range(100)] if task == TaskType.BINARY_CLASSIFICATION else list(range(100))
    path.write_text("x,category,target\n" + "\n".join(
        f"{i},category-{i % 3},{label}" for i, label in enumerate(labels)
    ), encoding="utf-8")
    config = DatasetConfig(task, "target", FeatureSchema((
        FeatureColumn("x", FeatureType.NUMERIC), FeatureColumn("category", FeatureType.CATEGORICAL),
    )), positive_label="yes" if task == TaskType.BINARY_CLASSIFICATION else None)
    data = validate_csv(path, ResourceLimits())
    manifest = create_split(data, config, RunConfig())
    configured = configure_dataset(data, config)

    class PoisonHoldout(tuple):
        def __getitem__(self, index):
            if index in manifest.holdout_row_ids:
                raise AssertionError("CV attempted holdout access")
            return super().__getitem__(index)

    guarded = replace(configured, rows=PoisonHoldout(configured.rows), targets=PoisonHoldout(configured.targets))
    training = training_from_manifest(guarded, manifest)
    return TrainingContext(training, manifest)


@pytest.mark.parametrize("task", list(TaskType))
def test_baseline_uses_every_frozen_fold_and_is_reproducible(tmp_path, task):
    context = context_for(tmp_path, task)
    first = evaluate_candidate(context, RunConfig())
    second = evaluate_candidate(context, RunConfig())
    assert first.eligible and second.eligible
    assert first.split_sha256 == context.split.manifest_sha256
    assert tuple(f.fold_index for f in first.folds) == (0, 1, 2, 3, 4)
    for actual, frozen, repeated in zip(first.folds, context.split.folds, second.folds, strict=True):
        assert actual.status == "completed"
        assert actual.validation_row_ids == frozen.validation_row_ids
        assert actual.metrics == repeated.metrics
        assert actual.fit_seconds >= 0 and actual.score_seconds >= 0
        assert actual.metrics.sample_count == len(frozen.validation_row_ids)
    assert first.total_seconds >= sum(f.fit_seconds + f.score_seconds for f in first.folds)
    assert first.summary == second.summary
    json.dumps(asdict(first), allow_nan=False)


def test_minority_positive_dummy_remains_comparable(tmp_path):
    result = evaluate_candidate(context_for(tmp_path), RunConfig())
    assert result.primary_metric == "f1"
    assert result.direction == "maximize"
    assert result.eligible
    f1 = next(metric for metric in result.summary if metric.name == "f1")
    assert (f1.mean, f1.std, f1.valid_folds) == (0, 0, 5)
    for fold in result.folds:
        assert fold.metrics.get("precision").reason == "zero_denominator"
        assert fold.metrics.get("f1").value == 0


def test_fresh_preprocessing_sees_only_outer_fit_rows(tmp_path, monkeypatch):
    context = context_for(tmp_path)
    original = FoldPreprocessor.fit
    fitted = []

    def inspect(self, data):
        fitted.append(self)
        index = len(fitted) - 1
        assert data.row_ids == context.split.folds[index].fit_row_ids
        assert not set(data.row_ids) & set(context.split.holdout_row_ids)
        result = original(self, data)
        mean = self.transformer_.named_transformers_["numeric"].named_steps["scaler"].mean_[0]
        assert mean == pytest.approx(np.mean(data.row_ids))
        return result

    monkeypatch.setattr(FoldPreprocessor, "fit", inspect)
    result = evaluate_candidate(context, RunConfig())
    assert result.eligible and len(fitted) == 5
    assert len({id(instance) for instance in fitted}) == 5


def test_regression_mean_baseline_and_population_std(tmp_path):
    context = context_for(tmp_path, TaskType.REGRESSION)
    result = evaluate_candidate(context, RunConfig())
    assert result.primary_metric == "rmse" and result.direction == "minimize"
    fold_scores = []
    for fold, frozen in zip(result.folds, context.split.folds, strict=True):
        # The fixture target equals global row ID: independent of model code.
        mean = sum(frozen.fit_row_ids) / len(frozen.fit_row_ids)
        expected = (sum((value - mean)**2 for value in frozen.validation_row_ids) / len(frozen.validation_row_ids))**.5
        assert fold.metrics.get("rmse").value == pytest.approx(expected)
        fold_scores.append(expected)
    aggregate = next(metric for metric in result.summary if metric.name == "rmse")
    assert aggregate.mean == pytest.approx(np.mean(fold_scores))
    assert aggregate.std == pytest.approx(np.std(fold_scores, ddof=0))
    assert aggregate.std != pytest.approx(np.std(fold_scores, ddof=1))


def test_failed_fold_is_recorded_without_partial_ranking(tmp_path, monkeypatch):
    context = context_for(tmp_path)
    original = FoldPreprocessor.fit
    calls = 0

    def fail_once(self, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("private row/path must not be exposed")
        return original(self, data)

    monkeypatch.setattr(FoldPreprocessor, "fit", fail_once)
    result = evaluate_candidate(context, RunConfig())
    assert not result.eligible and len(result.folds) == calls == 5
    assert result.folds[1].status == "failed"
    assert result.folds[1].error == "fold_execution_failed"
    assert result.folds[1].metrics is None
    assert all(item.mean is None and item.std is None and item.reason for item in result.summary)
    assert "private row/path" not in json.dumps(asdict(result))


@pytest.mark.parametrize("change", ["missing", "duplicate", "undefined"])
def test_partial_or_undefined_primary_cannot_be_eligible(tmp_path, change):
    result = evaluate_candidate(context_for(tmp_path), RunConfig())
    folds = result.folds
    if change == "missing":
        folds = folds[:-1]
    elif change == "duplicate":
        folds = folds[:-1] + (folds[0],)
    else:
        from ml.contracts import MetricValue
        metrics = folds[0].metrics
        metrics = replace(metrics, metrics=(MetricValue("f1", None, "test_undefined"),) + metrics.metrics[1:])
        folds = (replace(folds[0], metrics=metrics),) + folds[1:]
    partial = replace(result, folds=folds)
    assert not partial.eligible
    summary = summarize_candidate(partial)
    primary = next(metric for metric in summary if metric.name == "f1")
    assert primary.mean is None and primary.std is None and primary.reason


@pytest.mark.parametrize("task,primary", [
    (TaskType.BINARY_CLASSIFICATION, "average_precision"), (TaskType.REGRESSION, "mae"),
])
def test_primary_metric_is_preselected(tmp_path, task, primary):
    result = evaluate_candidate(context_for(tmp_path, task), RunConfig(primary_metric=primary))
    assert result.eligible and result.primary_metric == primary


def test_mismatched_context_or_run_is_rejected(tmp_path):
    context = context_for(tmp_path)
    for bad_context in (
        replace(context, training=replace(context.training, row_ids=tuple(reversed(context.training.row_ids)))),
        replace(context, split=replace(context.split, split_seed=43)),
        replace(context, training=replace(context.training, config=replace(context.training.config, positive_label="no"))),
    ):
        with pytest.raises(ValueError):
            evaluate_candidate(bad_context, RunConfig())
    for bad_run in (RunConfig(split_seed=43), RunConfig(primary_metric="mae"), RunConfig(family="arbitrary")):
        with pytest.raises(ValueError):
            evaluate_candidate(context, bad_run)


def test_dummy_factory_is_task_specific_and_rejects_unknown_inputs():
    classifier = make_classical(TaskType.BINARY_CLASSIFICATION, "dummy", 42)
    regressor = make_classical(TaskType.REGRESSION, "dummy", 42)
    assert isinstance(classifier, DummyClassifier) and classifier.strategy == "prior"
    assert isinstance(regressor, DummyRegressor) and regressor.strategy == "mean"
    for task, family, seed in (("regression", "dummy", 42), (TaskType.REGRESSION, "unknown", 42),
                                (TaskType.REGRESSION, "dummy", -1)):
        with pytest.raises(ValueError):
            make_classical(task, family, seed)


def test_memory_guard_failure_never_publishes_scores(tmp_path):
    result = evaluate_candidate(context_for(tmp_path), RunConfig(limits=ResourceLimits(preprocessing_bytes=1)))
    assert not result.eligible
    assert len(result.folds) == 5
    assert all(f.status == "failed" and f.metrics is None for f in result.folds)
