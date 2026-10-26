"""Finite, explicitly nullable task metrics; positive label is always encoded 1."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.contracts import MetricBundle, MetricValue, TaskType


def metric_direction(task: TaskType, name: str) -> str:
    if not isinstance(task, TaskType):
        raise ValueError("Explicit task required")
    names = (("f1", "precision", "recall", "roc_auc", "average_precision")
             if task == TaskType.BINARY_CLASSIFICATION else ("mae", "rmse", "r2"))
    if name not in names:
        raise ValueError("Metric does not belong to task")
    return "minimize" if name in ("mae", "rmse") else "maximize"


def _vector(values: ArrayLike, name: str) -> NDArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite numeric vector") from None
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _ratio(name: str, numerator: int, denominator: int) -> MetricValue:
    if not denominator:
        return MetricValue(name, None, "zero_denominator")
    return MetricValue(name, numerator / denominator)


def score_predictions(task: TaskType, truth: ArrayLike, prediction: ArrayLike,
                      score: ArrayLike | None = None) -> MetricBundle:
    """Binary score is P(encoded label 1), not a logit or a two-column matrix.

    Predictions, when probabilities are supplied, must use score >= 0.5.
    Undefined secondary metrics remain null without poisoning other metrics.
    Invalid/nonfinite inputs or unrepresentable arithmetic fail, not rank.
    """
    if not isinstance(task, TaskType):
        raise ValueError("Explicit task required")
    y = _vector(truth, "Truth")
    predicted = _vector(prediction, "Prediction")
    if y.shape != predicted.shape:
        raise ValueError("Truth and prediction must align")
    if task == TaskType.BINARY_CLASSIFICATION:
        if not np.isin(y, (0, 1)).all() or not np.isin(predicted, (0, 1)).all():
            raise ValueError("Binary labels must be encoded as 0 and 1")
        probabilities = None if score is None else _vector(score, "Probability")
        if probabilities is not None:
            if probabilities.shape != y.shape or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError("Probabilities must align and lie in [0, 1]")
            if not np.array_equal(predicted, probabilities >= .5):
                raise ValueError("Predictions must use fixed probability threshold 0.5")
        tp = int(np.sum((y == 1) & (predicted == 1)))
        fp = int(np.sum((y == 0) & (predicted == 1)))
        fn = int(np.sum((y == 1) & (predicted == 0)))
        tn = int(np.sum((y == 0) & (predicted == 0)))
        metrics = [_ratio("f1", 2 * tp, 2 * tp + fp + fn),
                   _ratio("precision", tp, tp + fp), _ratio("recall", tp, tp + fn)]
        reason = ("probabilities_unavailable" if probabilities is None else
                  "both_labels_required" if len(np.unique(y)) != 2 else None)
        for name, function in (("roc_auc", roc_auc_score), ("average_precision", average_precision_score)):
            metrics.append(MetricValue(name, None, reason) if reason else
                           MetricValue(name, float(function(y, probabilities))))
        return MetricBundle(task, len(y), tuple(metrics), ((tn, fp), (fn, tp)))

    if score is not None:
        raise ValueError("Regression does not accept probability scores")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            residual = y - predicted
            mae = float(np.mean(np.abs(residual)))
            rmse = float(np.sqrt(np.mean(residual**2)))
            if len(y) < 2:
                r2 = MetricValue("r2", None, "at_least_two_samples_required")
            elif np.all(y == y[0]):
                r2 = MetricValue("r2", None, "constant_target")
            else:
                r2 = MetricValue("r2", float(1 - np.sum(residual**2) / np.sum((y - np.mean(y))**2)))
    except FloatingPointError:
        raise ValueError("Regression metric arithmetic exceeds finite range") from None
    metrics = (MetricValue("mae", mae), MetricValue("rmse", rmse), r2)
    if any(metric.value is not None and not math.isfinite(metric.value) for metric in metrics):
        raise ValueError("Metric output must be finite")
    return MetricBundle(task, len(y), metrics)
