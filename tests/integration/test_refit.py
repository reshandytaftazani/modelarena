"""Day 18: training-only refit, frozen evidence and internal artifacts."""
from dataclasses import asdict, replace
import math
from statistics import median

import numpy as np
import pytest
import torch

from apps.api.services.finalization import compute_candidate_digest
from ml.contracts import EpochLoss, FrozenSelection, ResourceLimits, RunConfig, TaskType
from ml.tracking.manifest import canonical_bytes, digest_bytes
from ml.training.cv import evaluate_candidate
from tests.integration.test_baseline_cv import context_for


def evidence(tmp_path, task, family):
    context = context_for(tmp_path, task, positives=30)
    run = RunConfig(family=family, limits=ResourceLimits(compute_threads=2, max_epochs=4))
    candidate = evaluate_candidate(context, run)
    assert candidate.eligible
    if family == "mlp":
        folds = tuple(replace(f, best_epoch=n, loss_curve=tuple(
            EpochLoss(p.epoch, p.train_loss, 10.0 if p.epoch >= n else 20.0)
            for p in f.loss_curve)) for f, n in zip(candidate.folds, (1, 2, 3, 4, 4), strict=True))
        candidate = replace(candidate, folds=folds)
    selection = make_selection(context, candidate)
    return context, candidate, selection


def make_selection(context, candidate):
    run = candidate.run_config
    family, task = candidate.family, candidate.task
    score = next(x.mean for x in candidate.summary if x.name == candidate.primary_metric)
    cfg_hash = digest_bytes(canonical_bytes(asdict(run)))
    digest = compute_candidate_digest("candidate", family, context.split.manifest_sha256,
                                      cfg_hash, candidate.primary_metric, score, run.model_seed)
    selection = FrozenSelection("final", "scope", "experiment", "candidate", digest, family, task,
                                candidate.primary_metric, score, candidate.direction,
                                context.split.manifest_sha256, cfg_hash, run.model_seed, run.limits, "now")
    return selection


@pytest.mark.parametrize("task", list(TaskType))
@pytest.mark.parametrize("family", ["dummy", "mlp"])
def test_full_training_fresh_objects_and_exact_epochs(tmp_path, monkeypatch, task, family):
    from ml.training import refit

    context, candidate, selection = evidence(tmp_path, task, family)
    plan = refit.freeze_refit(context, selection, candidate)
    assert plan.epoch_budget == (3 if family == "mlp" else None)
    fitted, loaders, calls, accessed = [], [], [], []
    old_fit = refit.FoldPreprocessor.fit
    old_loader = refit.create_dataloader
    old_train = refit.train_mlp
    from ml.models.mlp import ArrayDataset
    old_getitem = ArrayDataset.__getitem__

    def spy_getitem(self, index):
        accessed.append(index)
        return old_getitem(self, index)

    def spy_fit(self, view):
        fitted.append((self, view.row_ids))
        return old_fit(self, view)

    def spy_loader(features, targets, **kwargs):
        loaders.append((features.copy(), targets.copy(), kwargs))
        return old_loader(features, targets, **kwargs)

    def spy_train(model, train_loader, val_loader=None, **kwargs):
        calls.append((model, val_loader, kwargs))
        return old_train(model, train_loader, val_loader, **kwargs)

    monkeypatch.setattr(refit.FoldPreprocessor, "fit", spy_fit)
    monkeypatch.setattr(refit, "create_dataloader", spy_loader)
    monkeypatch.setattr(refit, "train_mlp", spy_train)
    monkeypatch.setattr(ArrayDataset, "__getitem__", spy_getitem)
    first = refit.refit_selected(context, plan, candidate)
    second = refit.refit_selected(context, plan, candidate)
    assert first.model is not second.model and first.preprocessor is not second.preprocessor
    assert [x[1] for x in fitted] == [context.split.train_row_ids] * 2
    assert set(context.split.train_row_ids).isdisjoint(context.split.holdout_row_ids)
    if family == "mlp":
        assert len(loaders) == len(calls) == 2
        for features, targets, options in loaders:
            np.testing.assert_array_equal(features, first.preprocessor.transform(context.training, dtype=np.float32))
            np.testing.assert_array_equal(targets, np.asarray(context.training.targets, dtype=np.float32))
            assert options["shuffle"] is True
        assert all(val is None and kw["max_epochs"] == 3 and not kw["restore_best_weights"] for _, val, kw in calls)
        assert [x.epoch for x in first.training_loss] == [1, 2, 3]
        assert all(x.val_loss is None for x in first.training_loss)
        rows = len(context.training.row_ids)
        assert len(accessed) == 2 * 3 * rows
        assert all(sorted(accessed[start:start + rows]) == list(range(rows))
                   for start in range(0, len(accessed), rows))
        assert all(torch.equal(a, b) for a, b in zip(first.model.parameters(), second.model.parameters(), strict=True))
    else:
        assert not loaders and not calls and first.training_loss == ()


def test_rejects_incomplete_mismatched_and_changed_evidence(tmp_path):
    from ml.training.refit import freeze_refit, refit_selected
    context, candidate, selection = evidence(tmp_path, TaskType.REGRESSION, "mlp")
    plan = freeze_refit(context, selection, candidate)
    changes = [replace(candidate, folds=candidate.folds[:-1]),
               replace(candidate, expected_fold_indices=(0, 1, 2, 3), folds=candidate.folds[:-1]),
               replace(candidate, folds=(replace(candidate.folds[0], best_epoch=None, loss_curve=None), *candidate.folds[1:])),
               replace(candidate, folds=(replace(candidate.folds[0], status="failed", metrics=None, error="failed"), *candidate.folds[1:])),
               replace(candidate, folds=(replace(candidate.folds[0], validation_row_ids=candidate.folds[1].validation_row_ids), *candidate.folds[1:])),
               replace(candidate, folds=(replace(candidate.folds[0], best_epoch=2), *candidate.folds[1:]))]
    for bad in changes[:-1]:
        with pytest.raises(ValueError):
            freeze_refit(context, selection, bad)
    with pytest.raises(ValueError):
        refit_selected(context, plan, changes[-1])
    with pytest.raises(ValueError):
        freeze_refit(context, replace(selection, config_sha256="0" * 64), candidate)
    with pytest.raises(ValueError):
        freeze_refit(context, replace(selection, candidate_digest="0" * 64), candidate)
    with pytest.raises(ValueError):
        freeze_refit(context, selection, changes[-1])
    # A different but internally coherent curve must still fail against the
    # already frozen evidence digest, even if the best epoch stays the same.
    first = candidate.folds[0]
    curve = tuple(replace(point, val_loss=point.val_loss + 1) for point in first.loss_curve)
    coherent = replace(candidate, folds=(replace(first, loss_curve=curve), *candidate.folds[1:]))
    assert freeze_refit(context, selection, coherent).epoch_budget == plan.epoch_budget
    with pytest.raises(ValueError):
        refit_selected(context, plan, coherent)
    altered = replace(context.training, targets=(context.training.targets[0] + 1, *context.training.targets[1:]))
    with pytest.raises(ValueError):
        refit_selected(replace(context, training=altered), plan, candidate)


def test_even_fold_median_rounds_up(tmp_path):
    from ml.contracts import TrainingContext
    from ml.data.splits import training_from_manifest
    from ml.data.validation import configure_dataset
    from ml.training.refit import freeze_refit
    from tests.unit.test_splits import balanced_labels, split_binary
    data, config, split = split_binary(tmp_path, balanced_labels(5, 45))
    assert split.fold_count == 4
    context = TrainingContext(training_from_manifest(configure_dataset(data, config), split), split)
    run = RunConfig(family="mlp", limits=ResourceLimits(compute_threads=2, max_epochs=4))
    candidate = evaluate_candidate(context, run)
    assert candidate.eligible
    folds = tuple(replace(f, best_epoch=n, loss_curve=tuple(
        EpochLoss(p.epoch, p.train_loss, 10.0 if p.epoch >= n else 20.0)
        for p in f.loss_curve)) for f, n in zip(candidate.folds, (1, 2, 3, 4), strict=True))
    candidate = replace(candidate, folds=folds)
    plan = freeze_refit(context, make_selection(context, candidate), candidate)
    assert plan.best_epochs == (1, 2, 3, 4)
    assert plan.epoch_budget == 3


@pytest.mark.parametrize("task", list(TaskType))
def test_real_cv_best_epochs_are_used_without_manipulation(tmp_path, task):
    from ml.training.refit import freeze_refit, refit_selected
    context = context_for(tmp_path, task, positives=30)
    run = RunConfig(family="mlp", limits=ResourceLimits(compute_threads=2, max_epochs=3))
    candidate = evaluate_candidate(context, run)
    assert candidate.eligible
    plan = freeze_refit(context, make_selection(context, candidate), candidate)
    assert plan.best_epochs == tuple(f.best_epoch for f in candidate.folds)
    assert plan.epoch_budget == math.ceil(median(plan.best_epochs))
    model = refit_selected(context, plan, candidate)
    assert len(model.training_loss) == plan.epoch_budget


@pytest.mark.parametrize("task,family", [
    (TaskType.BINARY_CLASSIFICATION, "logistic"),
    (TaskType.REGRESSION, "ridge"),
])
def test_classical_estimator_receives_all_training_data(tmp_path, monkeypatch, task, family):
    from ml.training import refit
    context, candidate, selection = evidence(tmp_path, task, family)
    plan = refit.freeze_refit(context, selection, candidate)
    original = refit.make_classical
    fitted = []

    def inspect(*args):
        estimator = original(*args)
        fit = estimator.fit

        def spy(x, y):
            fitted.append((x.copy(), y.copy()))
            return fit(x, y)

        estimator.fit = spy
        return estimator

    monkeypatch.setattr(refit, "make_classical", inspect)
    result = refit.refit_selected(context, plan, candidate)
    assert len(fitted) == 1
    np.testing.assert_array_equal(fitted[0][0], result.preprocessor.transform(context.training))
    np.testing.assert_array_equal(fitted[0][1], np.asarray(context.training.targets))


@pytest.mark.parametrize("family", ["dummy", "mlp"])
def test_internal_artifact_provenance_and_integrity(tmp_path, family):
    from ml.training.refit import freeze_refit, refit_selected, save_internal_artifact, verify_internal_artifact
    context, candidate, selection = evidence(tmp_path, TaskType.REGRESSION, family)
    plan = freeze_refit(context, selection, candidate)
    model = refit_selected(context, plan, candidate)
    handle = save_internal_artifact(tmp_path / "private", model)
    manifest = verify_internal_artifact(handle)
    assert manifest["origin"] == "modelarena-local-refit-v1"
    assert manifest["selection_digest"] == selection.candidate_digest
    assert manifest["evidence_sha256"] == plan.evidence_sha256
    assert manifest["train_row_ids_sha256"] == digest_bytes(canonical_bytes(context.training.row_ids))
    assert manifest["epoch_budget"] == plan.epoch_budget
    assert manifest["provenance"]["packages"]["scikit-learn"]
    assert manifest["dataset_sha256"] == context.split.dataset_sha256
    assert manifest["training_view_sha256"] == plan.training_view_sha256
    assert manifest["dataset_config"]["target"] == context.training.config.target
    assert manifest["model_parameters"]
    assert manifest["fit_seconds"] >= 0
    # Hashes must reject altered metadata even when the model bytes are intact.
    second = save_internal_artifact(tmp_path / "private", model)
    metadata_file = second.directory / "manifest.json"
    metadata_file.write_bytes(metadata_file.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_internal_artifact(second)
    with pytest.raises((TypeError, ValueError, PermissionError)):
        verify_internal_artifact(tmp_path / "private")
    with pytest.raises(PermissionError):
        verify_internal_artifact(replace(handle, directory=tmp_path))
    with pytest.raises(PermissionError):
        save_internal_artifact(tmp_path / "private", replace(model, model=object()))
    artifact = handle.directory / next(iter(manifest["artifact_sha256"]))
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        verify_internal_artifact(handle)
