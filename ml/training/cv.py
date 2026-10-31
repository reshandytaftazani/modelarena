"""Serial classical CV on persisted identities; never imports a holdout reader.

This is an ordinary ML function for the execution child, not an API request
handler. The supervisor handles native-fit cancellation and the job deadline;
durable candidate publication remains a later milestone. Resource guards apply
in every fold.
"""

from dataclasses import dataclass, replace
import os
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ml.contracts import (
    CandidateResult, DiagnosticPoint, FoldResult, MetricSummary, RunConfig, TaskType,
    TrainingContext, TrainingView,
)
from ml.data.splits import _config_digest, verify_manifest
from ml.evaluation.metrics import metric_direction, oof_diagnostics, score_predictions
from ml.models.classical import make_classical, model_families
from ml.preprocessing.factory import FoldPreprocessor
from ml.training.mlp_loop import MLPFoldResult, TrainingCancelled, derive_fold_seed, train_mlp_fold


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


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: CandidateResult
    mean: float
    std: float
    mean_fit_seconds: float
    baseline_delta: float | None
    tied: bool


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    ranked: tuple[RankedCandidate, ...]
    unranked: tuple[CandidateResult, ...]


def compare_candidates(context: TrainingContext, candidates: tuple[CandidateResult, ...]) -> CandidateComparison:
    """Rank comparable complete candidates by fold mean, never by pooled OOF scores.

    Delta is positive for improvement. Near ties (within 1e-8 of the group's
    best mean) use mean fit time, then the unique family identifier.
    """
    if not candidates:
        raise ValueError("At least one candidate is required")
    split = context.split
    if (not verify_manifest(split) or context.training.row_ids != split.train_row_ids
            or _config_digest(context.training.config) != split.config_sha256):
        raise ValueError("Invalid frozen training context")
    first = candidates[0]
    if first.run_config is None:
        raise ValueError("Candidate run configuration is required")
    common_config = replace(first.run_config, family="dummy")
    expected = tuple(f.fold_index for f in split.folds)
    frozen = {f.fold_index: f.validation_row_ids for f in split.folds}
    seen = set()
    ranked = []
    unranked = []
    for candidate in candidates:
        run = candidate.run_config
        if (run is None or replace(run, family="dummy") != common_config
                or run.family != candidate.family or run.model_seed != candidate.model_seed
                or run.split_seed != split.split_seed
                or candidate.split_sha256 != split.manifest_sha256
                or candidate.task != context.training.config.task
                or candidate.primary_metric != first.primary_metric
                or candidate.primary_metric != (run.primary_metric or
                    ("f1" if candidate.task == TaskType.BINARY_CLASSIFICATION else "rmse"))
                or candidate.direction != metric_direction(candidate.task, candidate.primary_metric)
                or candidate.expected_fold_indices != expected or candidate.family in seen):
            raise ValueError("Incompatible candidate split, configuration or metric")
        seen.add(candidate.family)
        for fold in candidate.folds:
            if fold.fold_index not in frozen or fold.validation_row_ids != frozen[fold.fold_index]:
                raise ValueError("Candidate validation rows do not match frozen folds")
            if fold.metrics is not None and (fold.metrics.task != candidate.task
                    or fold.metrics.sample_count != len(fold.validation_row_ids)):
                raise ValueError("Candidate fold metrics do not match frozen rows")
        if not candidate.eligible:
            unranked.append(candidate)
            continue
        summary = next(item for item in summarize_candidate(candidate) if item.name == candidate.primary_metric)
        if summary.mean is None or summary.std is None:
            unranked.append(candidate)
            continue
        fit_times = [fold.fit_seconds for fold in candidate.folds]
        if any(not np.isfinite(t) or t < 0 for t in fit_times):
            raise ValueError("Invalid fold fit timing")
        ranked.append(RankedCandidate(candidate, summary.mean, summary.std, float(np.mean(fit_times)), None, False))
    baseline = next((entry.mean for entry in ranked if entry.candidate.family == "dummy"), None)
    ranked = [replace(entry, baseline_delta=(None if baseline is None else
              entry.mean - baseline if first.direction == "maximize" else baseline - entry.mean)) for entry in ranked]
    ranked.sort(key=lambda entry: entry.mean, reverse=first.direction == "maximize")
    ordered = []
    while ranked:
        best = ranked[0].mean
        group = []
        while ranked and abs(ranked[0].mean - best) <= 1e-8:
            group.append(ranked.pop(0))
        group.sort(key=lambda entry: (entry.mean_fit_seconds, entry.candidate.family))
        ordered.extend(replace(entry, tied=len(group) > 1) for entry in group)
    return CandidateComparison(tuple(ordered), tuple(unranked))


def evaluate_candidate(context: TrainingContext, config: RunConfig, *,
                       cancelled: Callable[[], bool] = lambda: False,
                       on_fold: Callable[[int, bool], None] = lambda _index, _finished: None) -> CandidateResult:
    """Fit a fresh guarded preprocessor and estimator on each outer-fit set."""
    from ml.training.classical_suite import SuiteCancelled
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
    if config.family not in model_families(task):
        raise ValueError(f"Unsupported task or model family: {config.family}")
    if type(config.model_seed) is not int or not 0 <= config.model_seed < 2**32:
        raise ValueError("Model seed must be an unsigned 32-bit integer")
    if config.family != "mlp":
        make_classical(task, config.family, config.model_seed)
    positions = {row_id: index for index, row_id in enumerate(view.row_ids)}
    folds = []
    oof_points = []
    with threadpool_limits(limits=config.limits.compute_threads, user_api="blas"), \
            threadpool_limits(limits=1, user_api="openmp"):
        for frozen in split.folds:
            if cancelled():
                # Never translate cancellation into a failed fold or publish a partial candidate.
                raise SuiteCancelled()
            on_fold(frozen.fold_index, False)
            fit_seconds = score_seconds = 0.0
            stage_started = perf_counter()
            scoring = False
            mlp_fold_res = None
            try:
                if config.family == "mlp":
                    test_block = os.environ.get("_MODELARENA_TEST_MLP_BLOCK_SECONDS")
                    if test_block:
                        import time
                        time.sleep(float(test_block))
                    mlp_fold_res = train_mlp_fold(
                        view,
                        frozen,
                        config,
                        run_control=cancelled,
                        holdout_row_ids=split.holdout_row_ids,
                    )
                    fit_seconds = mlp_fold_res.fit_seconds
                    if fit_seconds >= config.limits.mlp_fold_seconds:
                        raise TimeoutError("MLP fold deadline exceeded")
                    scoring = True
                    stage_started = perf_counter()
                    validation = _subset(view, frozen.validation_row_ids, positions)
                    encoded_validation = mlp_fold_res.preprocessor.transform(validation, dtype=np.float32)
                    mlp_fold_res.model.eval()
                    with torch.no_grad():
                        raw_preds = mlp_fold_res.model(torch.from_numpy(encoded_validation)).squeeze(-1).numpy()
                    if raw_preds.ndim == 0:
                        raw_preds = np.expand_dims(raw_preds, 0)
                    if not np.isfinite(raw_preds).all():
                        raise ValueError("Model predictions must be finite")
                    if task == TaskType.BINARY_CLASSIFICATION:
                        score = torch.sigmoid(torch.from_numpy(raw_preds)).numpy().astype(np.float64)
                        prediction = (score >= .5).astype(int)
                    else:
                        score = None
                        prediction = raw_preds.astype(np.float64)
                else:
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
                fold_points = tuple(DiagnosticPoint(
                    row_id, frozen.fold_index, float(truth), float(predicted),
                    float(score[index]) if score is not None else None,
                    float(truth - predicted) if score is None else None,
                ) for index, (row_id, truth, predicted) in enumerate(
                    zip(validation.row_ids, validation.targets, prediction, strict=True)))
                score_seconds = perf_counter() - stage_started
                best_epoch = mlp_fold_res.training_result.best_epoch if mlp_fold_res else None
                epochs_trained = mlp_fold_res.training_result.epochs_trained if mlp_fold_res else None
                stop_reason = mlp_fold_res.training_result.stop_reason if mlp_fold_res else None
                inner_fit_count = len(mlp_fold_res.inner_fit_row_ids) if mlp_fold_res else None
                loss_curve = mlp_fold_res.training_result.history if mlp_fold_res else None
                folds.append(FoldResult(
                    frozen.fold_index, frozen.validation_row_ids, "completed",
                    metrics, fit_seconds, score_seconds,
                    best_epoch=best_epoch,
                    epochs_trained=epochs_trained,
                    stop_reason=stop_reason,
                    inner_fit_row_count=inner_fit_count,
                    loss_curve=loss_curve,
                ))
                oof_points.extend(fold_points)
            except (TrainingCancelled, SuiteCancelled):
                raise SuiteCancelled() from None
            except TimeoutError:
                folds.append(FoldResult(frozen.fold_index, frozen.validation_row_ids, "timeout", None,
                                        perf_counter() - stage_started, 0.0, "fold_timeout"))
            except (ValueError, RuntimeError, ArithmeticError):
                elapsed = perf_counter() - stage_started
                if scoring:
                    score_seconds = elapsed
                else:
                    fit_seconds = elapsed
                metadata = {}
                if mlp_fold_res is not None and scoring:
                    training = mlp_fold_res.training_result
                    metadata = {"best_epoch": training.best_epoch, "epochs_trained": training.epochs_trained,
                                "stop_reason": training.stop_reason,
                                "inner_fit_row_count": len(mlp_fold_res.inner_fit_row_ids),
                                "loss_curve": training.history}
                folds.append(FoldResult(frozen.fold_index, frozen.validation_row_ids, "failed", None,
                                        fit_seconds, score_seconds, "fold_execution_failed", **metadata))
            finally:
                # Do not retain previous fold matrices while allocating the next.
                encoded_fit = encoded_validation = model = preprocessor = mlp_fold_res = None
            on_fold(frozen.fold_index, True)
    result = CandidateResult(task, config.family, config.model_seed, split.manifest_sha256,
                             primary, direction, tuple(f.fold_index for f in split.folds),
                             tuple(folds), (), perf_counter() - started, config)
    diagnostics = None
    if result.complete:
        try:
            diagnostics = oof_diagnostics(task, tuple(oof_points), config.limits.max_diagnostic_points)
        except (ValueError, ArithmeticError):
            # A pooled diagnostic can overflow even when individual folds are
            # valid; never fabricate pooled values or affect fold-based ranking.
            diagnostics = None
    return replace(result, summary=summarize_candidate(result), diagnostics=diagnostics)
