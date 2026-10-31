"""Acceptance tests for repeated CV reproducibility and tolerance bounds (Day 20a).

Verifies D20a:
- Complete repeated CV under identical locked environment, threads, seeds, configs, and split identities.
- Metric tolerance: abs=1e-6 or rel=1e-5; timings (fit/score/total wall seconds) strictly excluded.
- Actual maximum deviations recorded and logged rather than promising cross-platform bitwise identity.
- Complete fold coverage: every fold completed, validation row IDs verified.
- Exact split and configuration identity equality.
- Zero holdout access guarded by PoisonHoldout on both classification and regression.
- Adversarial sensitivity tests: divergence detection on tampered metrics, model seed separation
  distinguished from split seeds, and split seed partition control.
"""

from dataclasses import replace
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.contracts import (
    CandidateResult,
    DatasetConfig,
    FeatureColumn,
    FeatureSchema,
    FeatureType,
    MetricValue,
    ResourceLimits,
    RunConfig,
    TaskType,
    TrainingContext,
)
from ml.data.csv_io import validate_csv
from ml.data.splits import _config_digest, create_split, training_from_manifest
from ml.data.validation import configure_dataset
from ml.models.classical import classical_families, model_families
from ml.training.cv import compare_candidates, evaluate_candidate

logger = logging.getLogger(__name__)


def assert_metric_within_tolerance(
    val1: float | None,
    val2: float | None,
    *,
    name: str = "metric",
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-5,
) -> tuple[float, float]:
    """Assert two metric values match within abs=1e-6 or rel=1e-5 tolerance.

    Returns (abs_deviation, rel_deviation).
    """
    if val1 is None and val2 is None:
        return 0.0, 0.0
    if val1 is None or val2 is None:
        raise AssertionError(f"Metric nullness mismatch for {name}: {val1} vs {val2}")
    if not (np.isfinite(val1) and np.isfinite(val2)):
        raise AssertionError(f"Non-finite metric values for {name}: {val1} vs {val2}")
    abs_dev = abs(val1 - val2)
    denom = max(abs(val1), abs(val2))
    rel_dev = (abs_dev / denom) if denom > 0.0 else 0.0
    if abs_dev > abs_tol and rel_dev > rel_tol:
        raise AssertionError(
            f"Metric divergence for {name} exceeds tolerance: {val1} vs {val2} "
            f"(abs_dev={abs_dev:.2e} > {abs_tol:.2e}, rel_dev={rel_dev:.2e} > {rel_tol:.2e})"
        )
    return abs_dev, rel_dev


def make_reproducibility_fixture(
    root: Path,
    task: TaskType,
    row_count: int = 100,
    split_seed: int = 42,
) -> TrainingContext:
    """Create a self-contained training-only fixture with PoisonHoldout protection."""
    fixture_dir = root / f"fixture_{task.value}_{split_seed}"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fixture_dir / "data.csv"

    # Deterministic generation with slight non-linearity to exercise model mechanics
    rng = np.random.RandomState(42)
    x_num = rng.uniform(-3.0, 3.0, size=row_count)
    categories = [f"cat_{i % 4}" for i in range(row_count)]

    if task == TaskType.BINARY_CLASSIFICATION:
        # Non-linear boundary with balanced classes
        logits = 0.8 * x_num - 0.3 * (x_num**2) + (rng.uniform(-0.5, 0.5, size=row_count))
        targets = ["yes" if val > 0.0 else "no" for val in logits]
        config = DatasetConfig(
            task=task,
            target="target",
            features=FeatureSchema((
                FeatureColumn("num_feat", FeatureType.NUMERIC),
                FeatureColumn("cat_feat", FeatureType.CATEGORICAL),
            )),
            positive_label="yes",
        )
    else:
        # Continuous non-linear target
        targets = [float(2.5 * x - 0.5 * (x**2) + rng.normal(0, 0.2)) for x in x_num]
        config = DatasetConfig(
            task=task,
            target="target",
            features=FeatureSchema((
                FeatureColumn("num_feat", FeatureType.NUMERIC),
                FeatureColumn("cat_feat", FeatureType.CATEGORICAL),
            )),
        )

    lines = ["num_feat,cat_feat,target"]
    for x, cat, y in zip(x_num, categories, targets, strict=True):
        lines.append(f"{x:.6f},{cat},{y}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data = validate_csv(csv_path, ResourceLimits(), expected_root=fixture_dir)
    split = create_split(data, config, RunConfig(split_seed=split_seed))
    configured = configure_dataset(data, config)

    # PoisonHoldout ensures any access to held-out rows immediately raises AssertionError
    class PoisonHoldout(tuple):
        def __getitem__(self, index):
            if index in split.holdout_row_ids:
                raise AssertionError(f"Reproducibility CV attempted forbidden holdout access on row {index}")
            return super().__getitem__(index)

    guarded = replace(
        configured,
        rows=PoisonHoldout(configured.rows),
        targets=PoisonHoldout(configured.targets),
    )
    training = training_from_manifest(guarded, split)
    return TrainingContext(training, split)


def _compare_candidate_results(
    first: CandidateResult,
    second: CandidateResult,
    context: TrainingContext,
) -> dict[str, float]:
    """Compare two complete repeated CV runs under identical setup.

    Asserts:
    - Exact split identity
    - Exact candidate and configuration identity
    - Complete fold coverage (all folds completed, validation row IDs match)
    - Metric tolerance (abs=1e-6 or rel=1e-5)
    - Timings strictly excluded from tolerance check (verified positive and finite only)
    - Diagnostics predictions exact match and probabilities within tolerance

    Returns a dictionary of actual maximum absolute and relative deviations observed.
    """
    assert first is not second, "Candidates must be separately computed instances"
    split = context.split

    # 1. Exact split identity
    assert first.split_sha256 == split.manifest_sha256
    assert second.split_sha256 == split.manifest_sha256
    assert first.split_sha256 == second.split_sha256

    # 2. Exact configuration identity
    assert first.task == second.task == context.training.config.task
    assert first.family == second.family
    assert first.model_seed == second.model_seed
    assert first.primary_metric == second.primary_metric
    assert first.direction == second.direction
    assert first.expected_fold_indices == second.expected_fold_indices == tuple(f.fold_index for f in split.folds)
    assert first.run_config is not None and first.run_config == second.run_config
    assert _config_digest(context.training.config) == split.config_sha256
    assert first.run_config.model_seed == first.model_seed
    assert first.run_config.family == first.family
    assert first.run_config.split_seed == split.split_seed

    # 3. Complete fold coverage
    assert len(first.folds) == split.fold_count
    assert len(second.folds) == split.fold_count
    assert first.complete and second.complete
    assert first.eligible and second.eligible

    # 4. Timings: excluded from tolerance assertion, but must be finite non-negative numbers
    assert first.total_seconds > 0.0 and np.isfinite(first.total_seconds)
    assert second.total_seconds > 0.0 and np.isfinite(second.total_seconds)
    # Wall clock times will naturally vary across repeated runs
    assert isinstance(first.total_seconds, float)
    assert isinstance(second.total_seconds, float)

    max_abs_dev = 0.0
    max_rel_dev = 0.0

    # 5. Per-fold metrics and row identities
    for f1, f2, frozen in zip(first.folds, second.folds, split.folds, strict=True):
        assert f1.fold_index == f2.fold_index == frozen.fold_index
        assert f1.status == f2.status == "completed"
        assert f1.validation_row_ids == f2.validation_row_ids == frozen.validation_row_ids

        # Timing exclusion per fold
        assert f1.fit_seconds >= 0.0 and np.isfinite(f1.fit_seconds)
        assert f2.fit_seconds >= 0.0 and np.isfinite(f2.fit_seconds)
        assert f1.score_seconds >= 0.0 and np.isfinite(f1.score_seconds)
        assert f2.score_seconds >= 0.0 and np.isfinite(f2.score_seconds)

        # For MLP, fold training metadata must match
        if first.family == "mlp":
            assert f1.epochs_trained == f2.epochs_trained
            assert f1.best_epoch == f2.best_epoch
            assert f1.stop_reason == f2.stop_reason
            assert f1.inner_fit_row_count == f2.inner_fit_row_count
            assert f1.loss_curve is not None and f2.loss_curve is not None
            if f1.loss_curve is not None and f2.loss_curve is not None:
                assert len(f1.loss_curve) == len(f2.loss_curve)
                for ep1, ep2 in zip(f1.loss_curve, f2.loss_curve, strict=True):
                    assert ep1.epoch == ep2.epoch
                    ad, rd = assert_metric_within_tolerance(ep1.train_loss, ep2.train_loss, name=f"fold_{f1.fold_index}_train_loss")
                    max_abs_dev = max(max_abs_dev, ad)
                    max_rel_dev = max(max_rel_dev, rd)
                    if ep1.val_loss is not None and ep2.val_loss is not None:
                        ad, rd = assert_metric_within_tolerance(ep1.val_loss, ep2.val_loss, name=f"fold_{f1.fold_index}_val_loss")
                        max_abs_dev = max(max_abs_dev, ad)
                        max_rel_dev = max(max_rel_dev, rd)

        # Fold metrics
        assert f1.metrics is not None and f2.metrics is not None
        assert f1.metrics.task == f2.metrics.task
        assert f1.metrics.sample_count == f2.metrics.sample_count == len(frozen.validation_row_ids)
        metric_names = tuple(m.name for m in f1.metrics.metrics)
        assert metric_names == tuple(m.name for m in f2.metrics.metrics)

        for mv1, mv2 in zip(f1.metrics.metrics, f2.metrics.metrics, strict=True):
            assert mv1.reason == mv2.reason
            ad, rd = assert_metric_within_tolerance(mv1.value, mv2.value, name=f"fold_{f1.fold_index}_{mv1.name}")
            max_abs_dev = max(max_abs_dev, ad)
            max_rel_dev = max(max_rel_dev, rd)

    # 6. Summary metrics
    assert len(first.summary) == len(second.summary)
    for s1, s2 in zip(first.summary, second.summary, strict=True):
        assert s1.name == s2.name
        assert s1.valid_folds == s2.valid_folds
        assert s1.reason == s2.reason
        if s1.name == first.primary_metric:
            assert s1.valid_folds == split.fold_count
            assert s1.reason is None
        ad_m, rd_m = assert_metric_within_tolerance(s1.mean, s2.mean, name=f"summary_mean_{s1.name}")
        ad_s, rd_s = assert_metric_within_tolerance(s1.std, s2.std, name=f"summary_std_{s1.name}")
        max_abs_dev = max(max_abs_dev, ad_m, ad_s)
        max_rel_dev = max(max_rel_dev, rd_m, rd_s)

    # 7. Diagnostic points
    assert first.diagnostics is not None and second.diagnostics is not None
    if first.diagnostics is not None and second.diagnostics is not None:
        assert len(first.diagnostics.points) == len(second.diagnostics.points)
        for p1, p2 in zip(first.diagnostics.points, second.diagnostics.points, strict=True):
            assert p1.row_id == p2.row_id
            assert p1.fold_index == p2.fold_index
            assert p1.truth == p2.truth
            ad_pred, rd_pred = assert_metric_within_tolerance(p1.prediction, p2.prediction, name=f"diag_prediction_row_{p1.row_id}")
            max_abs_dev = max(max_abs_dev, ad_pred)
            max_rel_dev = max(max_rel_dev, rd_pred)
            ad_p, rd_p = assert_metric_within_tolerance(p1.probability, p2.probability, name=f"diag_prob_row_{p1.row_id}")
            ad_r, rd_r = assert_metric_within_tolerance(p1.residual, p2.residual, name=f"diag_resid_row_{p1.row_id}")
            max_abs_dev = max(max_abs_dev, ad_p, ad_r)
            max_rel_dev = max(max_rel_dev, rd_p, rd_r)

    logger.info(
        "Candidate %s (%s) repeated CV deviation: max_abs=%.2e, max_rel=%.2e",
        first.family, first.task.value, max_abs_dev, max_rel_dev,
    )
    return {"max_abs_deviation": max_abs_dev, "max_rel_deviation": max_rel_dev}


# -----------------------------------------------------------------------------
# D20a Test Cases
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["dummy", "logistic", "random_forest", "hist_gradient_boosting", "mlp"])
def test_repeated_cv_binary_classification(tmp_path, family):
    """Repeat complete 5-fold CV for every binary classification family under locked seeds and threads."""
    context = make_reproducibility_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION, row_count=100, split_seed=42)
    run_config = RunConfig(
        family=family,
        split_seed=42,
        model_seed=42,
        limits=ResourceLimits(compute_threads=2, max_epochs=6),
    )

    # Run 1
    first = evaluate_candidate(context, run_config)
    # Run 2 (fresh execution, re-instantiated estimators and preprocessors)
    second = evaluate_candidate(context, run_config)

    deviations = _compare_candidate_results(first, second, context)
    assert deviations["max_abs_deviation"] >= 0 and deviations["max_rel_deviation"] >= 0


@pytest.mark.parametrize("family", ["dummy", "ridge", "random_forest", "hist_gradient_boosting", "mlp"])
def test_repeated_cv_regression(tmp_path, family):
    """Repeat complete 5-fold CV for every regression family under locked seeds and threads."""
    context = make_reproducibility_fixture(tmp_path, TaskType.REGRESSION, row_count=100, split_seed=42)
    run_config = RunConfig(
        family=family,
        split_seed=42,
        model_seed=42,
        limits=ResourceLimits(compute_threads=2, max_epochs=6),
    )

    first = evaluate_candidate(context, run_config)
    second = evaluate_candidate(context, run_config)

    deviations = _compare_candidate_results(first, second, context)
    assert deviations["max_abs_deviation"] >= 0 and deviations["max_rel_deviation"] >= 0


def test_candidate_comparison_is_deterministic(tmp_path):
    """Candidate comparison and ranking order is identical across repeated suite runs."""
    context = make_reproducibility_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION, row_count=100, split_seed=42)
    families = ("dummy", "logistic", "random_forest", "mlp")

    run_1_results = []
    run_2_results = []

    for fam in families:
        run = RunConfig(family=fam, split_seed=42, model_seed=42, limits=ResourceLimits(compute_threads=2, max_epochs=4))
        res1 = evaluate_candidate(context, run)
        res2 = evaluate_candidate(context, run)
        run_1_results.append(res1)
        run_2_results.append(res2)

    comp1 = compare_candidates(context, tuple(run_1_results))
    comp2 = compare_candidates(context, tuple(run_2_results))

    assert len(comp1.ranked) == len(comp2.ranked) == len(families)
    assert not comp1.unranked and not comp2.unranked

    for r1, r2 in zip(comp1.ranked, comp2.ranked, strict=True):
        assert r1.candidate.family == r2.candidate.family
        ad_m, rd_m = assert_metric_within_tolerance(r1.mean, r2.mean, name=f"ranked_mean_{r1.candidate.family}")
        ad_s, rd_s = assert_metric_within_tolerance(r1.std, r2.std, name=f"ranked_std_{r1.candidate.family}")
        assert ad_m <= 1e-6 or rd_m <= 1e-5
        assert ad_s <= 1e-6 or rd_s <= 1e-5
        if r1.baseline_delta is not None:
            assert r2.baseline_delta is not None
            ad_b, rd_b = assert_metric_within_tolerance(r1.baseline_delta, r2.baseline_delta, name=f"ranked_delta_{r1.candidate.family}")
            assert ad_b <= 1e-6 or rd_b <= 1e-5
        assert r1.tied == r2.tied


# -----------------------------------------------------------------------------
# Adversarial / Mismatch Detection Tests
# -----------------------------------------------------------------------------

def test_tolerance_assertion_rejects_divergence():
    """Verify that assert_metric_within_tolerance rejects divergence beyond abs=1e-6 and rel=1e-5."""
    # Exact match passes
    assert assert_metric_within_tolerance(0.5, 0.5) == (0.0, 0.0)

    # Within absolute tolerance (1e-6)
    ad, rd = assert_metric_within_tolerance(1.0, 1.0 + 5e-7)
    assert ad <= 1e-6

    # Within relative tolerance (1e-5)
    ad, rd = assert_metric_within_tolerance(100.0, 100.0005)
    assert rd <= 1e-5

    # Divergence exceeding both tolerances fails with AssertionError
    with pytest.raises(AssertionError, match="exceeds tolerance"):
        assert_metric_within_tolerance(0.5, 0.501, abs_tol=1e-6, rel_tol=1e-5)

    with pytest.raises(AssertionError, match="exceeds tolerance"):
        assert_metric_within_tolerance(10.0, 10.1, abs_tol=1e-6, rel_tol=1e-5)

    # Nullness mismatch fails
    with pytest.raises(AssertionError, match="nullness mismatch"):
        assert_metric_within_tolerance(0.5, None)

    # Non-finite values fail
    with pytest.raises(AssertionError, match="Non-finite"):
        assert_metric_within_tolerance(float("nan"), 0.5)


def test_model_seeds_distinguished_from_split_seeds(tmp_path):
    """Model seed changes stochastic predictions and metrics, while split seed defines identical partitions."""
    context = make_reproducibility_fixture(tmp_path, TaskType.REGRESSION, row_count=100, split_seed=42)

    # Both runs use identical split (split_seed=42)
    run_seed_42 = RunConfig(family="random_forest", split_seed=42, model_seed=42, limits=ResourceLimits(compute_threads=2))
    run_seed_43 = RunConfig(family="random_forest", split_seed=42, model_seed=43, limits=ResourceLimits(compute_threads=2))

    res_42 = evaluate_candidate(context, run_seed_42)
    res_43 = evaluate_candidate(context, run_seed_43)

    # 1. Split identity is completely identical
    assert res_42.split_sha256 == res_43.split_sha256 == context.split.manifest_sha256
    for f42, f43 in zip(res_42.folds, res_43.folds, strict=True):
        assert f42.validation_row_ids == f43.validation_row_ids

    # 2. But model seed difference produces different metrics on stochastic models
    assert res_42.model_seed != res_43.model_seed
    mean_42 = next(s.mean for s in res_42.summary if s.name == res_42.primary_metric)
    mean_43 = next(s.mean for s in res_43.summary if s.name == res_43.primary_metric)

    # Difference must be observable and non-zero
    diff = abs(mean_42 - mean_43)
    assert diff > 1e-5, f"Expected model seed variation, but difference was {diff:.2e}"


def test_mlp_model_seeds_produce_different_initializations(tmp_path):
    """MLP derives different fold seeds and weights across model seeds while preserving fold boundaries."""
    context = make_reproducibility_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION, row_count=100, split_seed=42)

    run_42 = RunConfig(family="mlp", split_seed=42, model_seed=42, limits=ResourceLimits(compute_threads=2, max_epochs=10))
    run_43 = RunConfig(family="mlp", split_seed=42, model_seed=43, limits=ResourceLimits(compute_threads=2, max_epochs=10))

    res_42 = evaluate_candidate(context, run_42)
    res_43 = evaluate_candidate(context, run_43)

    assert res_42.split_sha256 == res_43.split_sha256
    # Diagnostic predictions/probabilities differ due to different weight initialization
    p42 = np.array([p.probability for p in res_42.diagnostics.points if p.probability is not None])
    p43 = np.array([p.probability for p in res_43.diagnostics.points if p.probability is not None])
    assert not np.allclose(p42, p43, atol=1e-4), "Expected MLP model seed variation across seeds 42 and 43"


def test_split_seed_controls_partitioning(tmp_path):
    """Different split seeds produce different partitions and hashes."""
    ctx_42 = make_reproducibility_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION, row_count=100, split_seed=42)
    ctx_43 = make_reproducibility_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION, row_count=100, split_seed=43)

    assert ctx_42.split.split_seed != ctx_43.split.split_seed
    assert ctx_42.split.manifest_sha256 != ctx_43.split.manifest_sha256
    assert ctx_42.split.train_row_ids != ctx_43.split.train_row_ids
    assert ctx_42.split.holdout_row_ids != ctx_43.split.holdout_row_ids


def test_replay_comparator_detects_real_result_tampering_and_ignores_timing(tmp_path):
    context = make_reproducibility_fixture(tmp_path, TaskType.REGRESSION)
    run = RunConfig(family="random_forest", limits=ResourceLimits(compute_threads=2))
    first = evaluate_candidate(context, run)
    second = evaluate_candidate(context, run)
    changed_time = replace(second.folds[0], fit_seconds=second.folds[0].fit_seconds + 1000)
    _compare_candidate_results(first, replace(second, total_seconds=second.total_seconds + 1000,
        folds=(changed_time, *second.folds[1:])), context)
    metric = second.folds[0].metrics
    altered = replace(metric.metrics[0], value=metric.metrics[0].value + 0.1)
    bad_fold = replace(second.folds[0], metrics=replace(metric, metrics=(altered, *metric.metrics[1:])))
    with pytest.raises(AssertionError, match="divergence"):
        _compare_candidate_results(first, replace(second, folds=(bad_fold, *second.folds[1:])), context)
    with pytest.raises(AssertionError):
        _compare_candidate_results(first, replace(second, run_config=replace(run, model_seed=43)), context)
    with pytest.raises(AssertionError):
        _compare_candidate_results(first, replace(second, folds=second.folds[:-1]), context)
    with pytest.raises(AssertionError):
        _compare_candidate_results(first, replace(second, diagnostics=None), context)
