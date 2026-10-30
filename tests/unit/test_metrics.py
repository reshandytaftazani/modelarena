"""Hand-derived metric references and explicit undefined/invalid cases."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from ml.contracts import TaskType
from ml.evaluation.metrics import metric_direction, score_predictions


BINARY = TaskType.BINARY_CLASSIFICATION
REGRESSION = TaskType.REGRESSION


def test_binary_counts_and_fixed_label_order():
    result = score_predictions(BINARY, [1, 0, 1, 0, 1], [1, 1, 0, 0, 1], [.9, .8, .2, .1, .7])
    assert result.confusion_matrix == ((1, 1), (1, 2))
    assert result.get("precision").value == pytest.approx(2 / 3)
    assert result.get("recall").value == pytest.approx(2 / 3)
    assert result.get("f1").value == pytest.approx(2 / 3)


def test_ap_is_not_trapezoidal_pr_area_and_auc_uses_positive_one():
    result = score_predictions(BINARY, [0, 1, 1, 0], [0, 1, 0, 1], [.1, .8, .4, .6])
    assert result.get("average_precision").value == pytest.approx(5 / 6)
    assert result.get("average_precision").value != pytest.approx(19 / 24)
    assert result.get("roc_auc").value == pytest.approx(.75)


def test_no_positive_predictions_keeps_zero_f1_comparable():
    result = score_predictions(BINARY, [0, 1, 0, 1], [0, 0, 0, 0], [.2] * 4)
    assert result.get("f1").value == 0
    assert result.get("recall").value == 0
    assert result.get("precision").value is None
    assert result.get("precision").reason
    assert result.get("roc_auc").value == .5
    assert result.get("average_precision").value == .5


@pytest.mark.parametrize("truth,prediction,null_names", [
    ([0, 0], [0, 0], ("precision", "recall", "f1", "roc_auc", "average_precision")),
    ([0, 0], [1, 0], ("recall", "roc_auc", "average_precision")),
    ([1, 1], [1, 1], ("roc_auc", "average_precision")),
])
def test_undefined_metrics_have_reasons(truth, prediction, null_names):
    scores = [.8 if p else .1 for p in prediction]
    result = score_predictions(BINARY, truth, prediction, scores)
    for metric in result.metrics:
        assert (metric.value is None) == (metric.name in null_names)
        assert bool(metric.reason) == (metric.value is None)
    json.dumps(asdict(result), allow_nan=False)


def test_threshold_is_fixed_at_half_including_ties():
    result = score_predictions(BINARY, [0, 1], [0, 1], [.499, .5])
    assert result.get("f1").value == 1
    with pytest.raises(ValueError, match="threshold"):
        score_predictions(BINARY, [0, 1], [0, 0], [.499, .5])


def test_missing_probabilities_do_not_invent_ranking_metrics():
    result = score_predictions(BINARY, [0, 1], [0, 1], None)
    assert result.get("f1").value == 1
    assert result.get("roc_auc").value is None
    assert result.get("average_precision").reason


def test_regression_reference_values():
    result = score_predictions(REGRESSION, [1, 2, 4], [2, 2, 2], None)
    assert result.confusion_matrix is None
    assert result.get("mae").value == 1
    assert result.get("rmse").value == pytest.approx(np.sqrt(5 / 3))
    assert result.get("r2").value == pytest.approx(-1 / 14)


@pytest.mark.parametrize("truth,prediction", [([2, 2], [2, 2]), ([2, 2], [1, 3]), ([2], [1])])
def test_constant_or_small_r2_is_null_not_sklearn_force_finite(truth, prediction):
    result = score_predictions(REGRESSION, truth, prediction, None)
    assert result.get("r2").value is None
    assert result.get("r2").reason
    assert result.get("mae").value is not None


@pytest.mark.parametrize("task,name,direction", [
    (BINARY, "f1", "maximize"), (BINARY, "precision", "maximize"),
    (BINARY, "recall", "maximize"), (BINARY, "roc_auc", "maximize"),
    (BINARY, "average_precision", "maximize"), (REGRESSION, "r2", "maximize"),
    (REGRESSION, "mae", "minimize"), (REGRESSION, "rmse", "minimize"),
])
def test_metric_directions(task, name, direction):
    assert metric_direction(task, name) == direction


@pytest.mark.parametrize("truth,prediction,score", [
    ([], [], []), ([0, 1], [0], [.1, .9]), ([[0, 1]], [0, 1], [.1, .9]),
    ([0, 2], [0, 1], [.1, .9]), ([0, 1], [0, 2], [.1, .9]),
    ([0, 1], [0, 1], [.1, float("nan")]), ([0, 1], [0, 1], [.1, float("inf")]),
    ([0, 1], [0, 1], [-.1, .9]), ([0, 1], [0, 1], [.1, 1.1]),
    ([0, 1], [0, 1], [[.1, .9], [.9, .1]]), ([0, 1], [0, 1], [.1]),
    ([float("nan"), 1], [0, 1], [.1, .9]),
])
def test_invalid_binary_vectors_fail(truth, prediction, score):
    with pytest.raises(ValueError):
        score_predictions(BINARY, truth, prediction, score)


@pytest.mark.parametrize("truth,prediction,score", [
    ([1, float("inf")], [1, 2], None), ([1, 2], [1, float("nan")], None),
    ([1, 2], [1, 2], [.1, .2]), ([], [], None),
])
def test_invalid_regression_vectors_fail(truth, prediction, score):
    with pytest.raises(ValueError):
        score_predictions(REGRESSION, truth, prediction, score)


def test_unknown_task_and_metric_are_rejected():
    with pytest.raises(ValueError):
        score_predictions("regression", [1, 2], [1, 2], None)
    with pytest.raises(ValueError):
        metric_direction(BINARY, "rmse")
