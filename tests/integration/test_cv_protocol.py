"""Frozen-fold comparison and out-of-fold diagnostics, never holdout selection."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from ml.contracts import MetricValue, ResourceLimits, RunConfig, TaskType
from ml.training.cv import compare_candidates, evaluate_candidate
from tests.integration.test_baseline_cv import context_for


@pytest.mark.parametrize("task,family", [(TaskType.BINARY_CLASSIFICATION, "logistic"),
                                      (TaskType.REGRESSION, "ridge")])
def test_comparison_uses_complete_fold_means_and_direction(tmp_path, task, family):
    context = context_for(tmp_path, task)
    baseline = evaluate_candidate(context, RunConfig())
    candidate = evaluate_candidate(context, RunConfig(family=family))
    comparison = compare_candidates(context, (baseline, candidate))
    assert {entry.candidate.family for entry in comparison.ranked} == {"dummy", family}
    values = [entry.mean for entry in comparison.ranked]
    assert values == sorted(values, reverse=task == TaskType.BINARY_CLASSIFICATION)
    for entry in comparison.ranked:
        scores = [fold.metrics.get(candidate.primary_metric).value for fold in entry.candidate.folds]
        assert entry.mean == pytest.approx(np.mean(scores))
        assert entry.std == pytest.approx(np.std(scores, ddof=0))
        base = next(x.mean for x in comparison.ranked if x.candidate.family == "dummy")
        expected = entry.mean - base if candidate.direction == "maximize" else base - entry.mean
        assert entry.baseline_delta == pytest.approx(expected)
        assert entry.mean_fit_seconds == pytest.approx(np.mean([f.fit_seconds for f in entry.candidate.folds]))


@pytest.mark.parametrize("mutation", ["split", "fold_rows", "metric", "seed", "limits", "direction"])
def test_incompatible_candidates_rejected(tmp_path, mutation):
    context = context_for(tmp_path, TaskType.REGRESSION)
    baseline = evaluate_candidate(context, RunConfig())
    other = evaluate_candidate(context, RunConfig(family="ridge"))
    if mutation == "split":
        other = replace(other, split_sha256="0" * 64)
    elif mutation == "fold_rows":
        other = replace(other, folds=(replace(other.folds[0], validation_row_ids=(999,)),) + other.folds[1:])
    elif mutation == "metric":
        other = evaluate_candidate(context, RunConfig(family="ridge", primary_metric="mae"))
    elif mutation == "seed":
        other = evaluate_candidate(context, RunConfig(family="ridge", model_seed=43))
    elif mutation == "limits":
        other = evaluate_candidate(context, RunConfig(family="ridge", limits=ResourceLimits(max_diagnostic_points=3)))
    else:
        other = replace(other, direction="maximize")
    with pytest.raises(ValueError):
        compare_candidates(context, (baseline, other))


@pytest.mark.parametrize("mutation", ["missing", "timeout", "undefined", "duplicate"])
def test_partial_timeout_or_undefined_never_ranks(tmp_path, mutation):
    context = context_for(tmp_path)
    baseline = evaluate_candidate(context, RunConfig())
    other = evaluate_candidate(context, RunConfig(family="logistic"))
    if mutation == "missing":
        other = replace(other, folds=other.folds[:-1])
    elif mutation == "duplicate":
        other = replace(other, folds=other.folds[:-1] + (other.folds[0],))
    elif mutation == "timeout":
        other = replace(other, folds=(replace(other.folds[0], status="timeout", metrics=None),) + other.folds[1:])
    else:
        fold = other.folds[0]
        metrics = replace(fold.metrics, metrics=(MetricValue("f1", None, "undefined"),) + fold.metrics.metrics[1:])
        other = replace(other, folds=(replace(fold, metrics=metrics),) + other.folds[1:])
    comparison = compare_candidates(context, (baseline, other))
    assert [entry.candidate.family for entry in comparison.ranked] == ["dummy"]
    assert comparison.unranked == (other,)


@pytest.mark.parametrize("task", list(TaskType))
def test_oof_lineage_diagnostics_bounded_and_deterministic(tmp_path, task, monkeypatch):
    context = context_for(tmp_path, task)
    from ml.preprocessing.factory import FoldPreprocessor
    original = FoldPreprocessor.fit
    fitted = []
    def spy(self, view):
        fitted.append(set(view.row_ids))
        return original(self, view)
    monkeypatch.setattr(FoldPreprocessor, "fit", spy)
    config = RunConfig(limits=ResourceLimits(max_diagnostic_points=7))
    result = evaluate_candidate(context, config)
    repeated = evaluate_candidate(context, config)
    diagnostics = result.diagnostics
    assert diagnostics == repeated.diagnostics
    assert diagnostics.partition == "out_of_fold_training"
    assert diagnostics.sample_count == len(context.training.row_ids)
    assert len(diagnostics.points) == 7
    for point in diagnostics.points:
        frozen = context.split.folds[point.fold_index]
        assert point.row_id in frozen.validation_row_ids
        assert point.row_id not in fitted[point.fold_index]
        assert point.row_id not in context.split.holdout_row_ids
        if task == TaskType.REGRESSION:
            assert point.residual == pytest.approx(point.truth - point.prediction)
    assert diagnostics.metrics.sample_count == len(context.training.row_ids)
    if task == TaskType.BINARY_CLASSIFICATION:
        assert sum(map(sum, diagnostics.metrics.confusion_matrix)) == len(context.training.row_ids)
        assert len(diagnostics.roc_curve) <= 7 and len(diagnostics.pr_curve) <= 7
    else:
        assert not diagnostics.roc_curve and not diagnostics.pr_curve
    json.dumps(asdict(result), allow_nan=False)


def test_failed_candidate_has_no_misleading_oof_diagnostics(tmp_path, monkeypatch):
    context = context_for(tmp_path)
    from ml.preprocessing.factory import FoldPreprocessor
    original = FoldPreprocessor.fit
    calls = 0
    def fail_once(self, view):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("private")
        return original(self, view)
    monkeypatch.setattr(FoldPreprocessor, "fit", fail_once)
    result = evaluate_candidate(context, RunConfig())
    assert not result.eligible and result.diagnostics is None


def test_near_ties_use_mean_fit_time_and_baseline_is_optional(tmp_path):
    context = context_for(tmp_path, TaskType.REGRESSION)
    baseline = evaluate_candidate(context, RunConfig())
    other = evaluate_candidate(context, RunConfig(family="ridge"))
    adjusted = []
    for base_fold, fold in zip(baseline.folds, other.folds, strict=True):
        metrics = tuple(replace(metric, value=base_fold.metrics.get("rmse").value + 5e-9)
                        if metric.name == "rmse" else metric for metric in fold.metrics.metrics)
        adjusted.append(replace(fold, metrics=replace(fold.metrics, metrics=metrics), fit_seconds=0))
    other = replace(other, folds=tuple(adjusted))
    baseline = replace(baseline, folds=tuple(replace(fold, fit_seconds=1) for fold in baseline.folds))
    comparison = compare_candidates(context, (baseline, other))
    assert [entry.candidate.family for entry in comparison.ranked] == ["ridge", "dummy"]
    assert all(entry.tied for entry in comparison.ranked)
    assert comparison.ranked[0].baseline_delta < 0
    without_dummy = compare_candidates(context, (other,))
    assert without_dummy.ranked[0].baseline_delta is None


def test_oof_pooled_rmse_is_labeled_and_not_used_for_ranking(tmp_path):
    context = context_for(tmp_path, TaskType.REGRESSION)
    candidate = evaluate_candidate(context, RunConfig())
    entry = compare_candidates(context, (candidate,)).ranked[0]
    pooled = candidate.diagnostics.metrics.get("rmse").value
    assert pooled != pytest.approx(entry.mean, abs=1e-6)
    assert entry.mean == pytest.approx(np.mean([fold.metrics.get("rmse").value for fold in candidate.folds]))


def test_display_cap_one_and_undefined_baseline(tmp_path):
    context = context_for(tmp_path)
    run = RunConfig(limits=ResourceLimits(max_diagnostic_points=1))
    baseline = evaluate_candidate(context, run)
    candidate = evaluate_candidate(context, replace(run, family="logistic"))
    assert len(candidate.diagnostics.points) == len(candidate.diagnostics.roc_curve) == len(candidate.diagnostics.pr_curve) == 1
    fold = baseline.folds[0]
    metrics = replace(fold.metrics, metrics=(MetricValue("f1", None, "undefined"),) + fold.metrics.metrics[1:])
    baseline = replace(baseline, folds=(replace(fold, metrics=metrics),) + baseline.folds[1:])
    comparison = compare_candidates(context, (baseline, candidate))
    assert comparison.ranked[0].baseline_delta is None
    assert comparison.unranked == (baseline,)
