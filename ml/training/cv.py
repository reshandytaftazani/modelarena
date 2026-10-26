"""Serial Dummy CV on persisted identities; never imports a holdout reader.

This is an ordinary ML function for the future execution child, not an API
request handler. Worker cancellation, hard native-fit deadlines and durable
publication are later milestones. Resource guards apply in every fold now.
"""

from dataclasses import replace
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from ml.contracts import (
    CandidateResult, FoldResult, MetricSummary, RunConfig, TaskType,
    TrainingContext, TrainingView,
)
from ml.data.splits import _config_digest, verify_manifest
from ml.evaluation.metrics import metric_direction, score_predictions
from ml.models.classical import make_classical
from ml.preprocessing.factory import FoldPreprocessor


def _subset(view: TrainingView, ids: tuple[int, ...], positions: dict[int, int]) -> TrainingView:
    return TrainingView(view.config, ids, tuple(view.rows[positions[i]] for i in ids),
                        tuple(view.targets[positions[i]] for i in ids), view.label_mapping)


def summarize_candidate(result: CandidateResult) -> tuple[MetricSummary, ...]:
    """Never average just the surviving folds. Undefined secondary stays null."""
    names = (("f1", "precision", "recall", "roc_auc", "average_precision")
             if result.task == TaskType.BINARY_CLASSIFICATION else ("mae", "rmse", "r2"))
    summary = []
    for name in names:
        values = []
        for fold in result.folds:
            if fold.status == "completed" and fold.metrics is not None:
                try:
                    value = fold.metrics.get(name).value
                except KeyError:
                    continue
                if value is not None and np.isfinite(value):
                    values.append(value)
        reason = ("incomplete_folds" if not result.complete else
                  "undefined_in_one_or_more_folds" if len(values) != len(result.expected_fold_indices) else None)
        summary.append(MetricSummary(name, None, None, len(values), reason) if reason else
                       MetricSummary(name, float(np.mean(values)), float(np.std(values, ddof=0)), len(values)))
    return tuple(summary)


def evaluate_candidate(context: TrainingContext, config: RunConfig) -> CandidateResult:
    """Fit a fresh guarded preprocessor + Dummy on each complete outer-fit set."""
    started = perf_counter()
    if not isinstance(context, TrainingContext) or not isinstance(config, RunConfig):
        raise TypeError("TrainingContext and RunConfig required")
    view, split = context.training, context.split
    if (not verify_manifest(split) or view.row_ids != split.train_row_ids
            or _config_digest(view.config) != split.config_sha256 or config.split_seed != split.split_seed):
        raise ValueError("Training context does not match the frozen split/configuration")
    task = view.config.task
    primary = config.primary_metric or ("f1" if task == TaskType.BINARY_CLASSIFICATION else "rmse")
    direction = metric_direction(task, primary)
    # Validate family/task/seed before beginning any fold work.
    make_classical(task, config.family, config.model_seed)
    positions = {row_id: index for index, row_id in enumerate(view.row_ids)}
    folds = []
    with threadpool_limits(limits=config.limits.compute_threads, user_api="blas"), \
            threadpool_limits(limits=1, user_api="openmp"):
        for frozen in split.folds:
            fit_seconds = score_seconds = 0.0
            stage_started = perf_counter()
            scoring = False
            try:
                fit = _subset(view, frozen.fit_row_ids, positions)
                validation = _subset(view, frozen.validation_row_ids, positions)
                preprocessor = FoldPreprocessor(config.limits)
                encoded_fit = preprocessor.fit_transform(fit)
                model = make_classical(task, config.family, config.model_seed)
                model.fit(encoded_fit, np.asarray(fit.targets))
                fit_seconds = perf_counter() - stage_started
                scoring = True
                stage_started = perf_counter()
                encoded_validation = preprocessor.transform(validation)
                if task == TaskType.BINARY_CLASSIFICATION:
                    positive_column = np.flatnonzero(model.classes_ == 1)
                    if len(positive_column) != 1 or set(model.classes_) != {0, 1}:
                        raise ValueError("Fitting fold requires encoded labels 0 and 1")
                    score = model.predict_proba(encoded_validation)[:, positive_column[0]]
                    prediction = (score >= .5).astype(int)
                else:
                    score = None
                    prediction = model.predict(encoded_validation)
                metrics = score_predictions(task, validation.targets, prediction, score)
                score_seconds = perf_counter() - stage_started
                folds.append(FoldResult(frozen.fold_index, frozen.validation_row_ids, "completed",
                                        metrics, fit_seconds, score_seconds))
            except (ValueError, RuntimeError, ArithmeticError):
                elapsed = perf_counter() - stage_started
                if scoring:
                    score_seconds = elapsed
                else:
                    fit_seconds = elapsed
                folds.append(FoldResult(frozen.fold_index, frozen.validation_row_ids, "failed", None,
                                        fit_seconds, score_seconds, "fold_execution_failed"))
            finally:
                # Do not retain previous fold matrices while allocating the next.
                encoded_fit = encoded_validation = model = preprocessor = None
    result = CandidateResult(task, config.family, config.model_seed, split.manifest_sha256,
                             primary, direction, tuple(f.fold_index for f in split.folds),
                             tuple(folds), (), perf_counter() - started)
    return replace(result, summary=summarize_candidate(result))
