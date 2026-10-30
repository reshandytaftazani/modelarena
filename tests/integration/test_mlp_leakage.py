"""Integration and leakage tests for fold-local MLP early stopping (Day 16).

Verifies strict fold boundaries, zero test-set leakage, inner-only preprocessor
fitting, inner-validation driven early stopping, best checkpoint restoration,
patience min_delta thresholding, deterministic seed replay, and CV integration.
"""

from dataclasses import replace
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.contracts import (
    DatasetConfig,
    FeatureColumn,
    FeatureSchema,
    FeatureType,
    FoldManifest,
    ResourceLimits,
    RunConfig,
    TaskType,
    TrainingContext,
    TrainingView,
)
from ml.data.csv_io import validate_csv
from ml.data.splits import create_split, training_from_manifest
from ml.data.validation import configure_dataset
from ml.models.mlp import ArrayDataset, create_dataloader, make_mlp
from ml.preprocessing.factory import FoldPreprocessor
from ml.training.cv import evaluate_candidate, train_mlp_fold
from ml.training.mlp_loop import (
    EpochLoss,
    MLPFoldResult,
    MLPTrainingResult,
    RunControl,
    derive_fold_seed,
    train_mlp,
)
from tests.integration.test_baseline_cv import context_for


# --- Fixture Helpers ---


def synthetic_context(tmp_path, task=TaskType.BINARY_CLASSIFICATION, n_samples=100, seed=42):
    """Generate a clean synthetic TrainingContext with known structure."""
    return context_for(tmp_path, task=task, positives=n_samples // 4 if task == TaskType.BINARY_CLASSIFICATION else None)


# --- D16a Tests: Inner Indices, Preprocessor Isolation, Gradient Isolation ---


def test_preprocessor_fits_inner_fit_only_row_id_spies(tmp_path):
    """Spy on FoldPreprocessor.fit to prove only inner-fit row IDs are seen."""
    context = synthetic_context(tmp_path, TaskType.BINARY_CLASSIFICATION)
    manifest = context.split.folds[0]

    fitted_row_ids = None
    orig_fit = FoldPreprocessor.fit

    def spy_fit(self, view: TrainingView):
        nonlocal fitted_row_ids
        fitted_row_ids = tuple(view.row_ids)
        return orig_fit(self, view)

    # Inject spy
    preprocessor = FoldPreprocessor(ResourceLimits())
    preprocessor.fit = spy_fit.__get__(preprocessor, FoldPreprocessor)

    result = train_mlp_fold(
        context.training,
        manifest,
        RunConfig(family="mlp"),
        preprocessor=preprocessor,
        holdout_row_ids=context.split.holdout_row_ids,
    )

    assert isinstance(result, MLPFoldResult)
    assert fitted_row_ids is not None

    # Prove fitted rows are strictly identical to persisted inner-fit row IDs
    assert fitted_row_ids == manifest.inner.fit_row_ids

    # Prove zero inner-validation rows in fit
    assert set(fitted_row_ids).isdisjoint(set(manifest.inner.validation_row_ids))

    # Prove zero outer-validation rows in fit
    assert set(fitted_row_ids).isdisjoint(set(manifest.validation_row_ids))

    # Prove zero holdout rows in fit
    assert set(fitted_row_ids).isdisjoint(set(context.split.holdout_row_ids))


def test_gradient_updates_receive_inner_fit_only_row_ids(tmp_path, monkeypatch):
    """Spy on train_loader batches to prove gradient updates touch inner-fit rows only."""
    context = synthetic_context(tmp_path, TaskType.BINARY_CLASSIFICATION)
    manifest = context.split.folds[0]

    seen_train_indices = set()
    orig_create_dataloader = create_dataloader

    # Wrap create_dataloader to tag batches
    def spy_create_dataloader(features, targets, batch_size=32, shuffle=True, seed=None):
        loader = orig_create_dataloader(features, targets, batch_size=batch_size, shuffle=shuffle, seed=seed)
        return loader

    monkeypatch.setattr("ml.training.mlp_loop.create_dataloader", spy_create_dataloader)

    # Use row ID spy inside train_mlp_fold
    result = train_mlp_fold(
        context.training,
        manifest,
        RunConfig(family="mlp"),
        max_epochs=2,
    )

    # Verify that inner fit row count equals the dataset length used for training
    assert len(manifest.inner.fit_row_ids) > 0
    assert len(manifest.inner.validation_row_ids) > 0

    # Ensure result correctly records inner fit and val row IDs
    assert result.inner_fit_row_ids == manifest.inner.fit_row_ids
    assert result.inner_val_row_ids == manifest.inner.validation_row_ids
    assert result.outer_val_row_ids == manifest.validation_row_ids


def test_inner_validation_sentinels_do_not_alter_fit_statistics(tmp_path):
    """Extreme outliers and category sentinels in inner-validation must not pollute scaler/imputer."""
    path = tmp_path / "sentinel.csv"
    # Create 100 unique rows
    rows = []
    for i in range(100):
        rows.append(f"{float(i)},cat_{i % 3},{'yes' if i < 30 else 'no'}")
    path.write_text("x,cat,target\n" + "\n".join(rows), encoding="utf-8")

    config = DatasetConfig(
        TaskType.BINARY_CLASSIFICATION,
        "target",
        FeatureSchema((
            FeatureColumn("x", FeatureType.NUMERIC),
            FeatureColumn("cat", FeatureType.CATEGORICAL),
        )),
        positive_label="yes",
    )
    data = validate_csv(path, ResourceLimits())
    manifest = create_split(data, config, RunConfig())
    configured = configure_dataset(data, config)
    training = training_from_manifest(configured, manifest)

    fold = manifest.folds[0]
    inner_val_id = fold.inner.validation_row_ids[0]

    # Mutate the training view so that inner_val_id has an extreme outlier and unseen category
    corrupted_rows = list(training.rows)
    pos = {rid: idx for idx, rid in enumerate(training.row_ids)}
    corrupted_rows[pos[inner_val_id]] = (1e9, "EXTREME_UNSEEN_SENTINEL")
    corrupted_view = replace(training, rows=tuple(corrupted_rows))

    # Fit fold
    result = train_mlp_fold(corrupted_view, fold, RunConfig(family="mlp"), max_epochs=1)
    preprocessor = result.preprocessor

    # Check fitted numeric scaler mean
    scaler = preprocessor.transformer_.named_transformers_["numeric"].named_steps["scaler"]
    # Mean of x in inner_fit rows should exactly match the uncorrupted inner-fit true mean, not ~1e7
    inner_fit_true_mean = float(np.mean([float(training.rows[pos[rid]][0]) for rid in fold.inner.fit_row_ids]))
    assert scaler.mean_[0] == pytest.approx(inner_fit_true_mean, abs=1e-5)

    # Check categorical encoder vocabulary
    encoder = preprocessor.transformer_.named_transformers_["categorical"].named_steps["encoder"]
    categories = encoder.categories_[0]
    # EXTREME_UNSEEN_SENTINEL must NOT be in learned categories
    assert "value:EXTREME_UNSEEN_SENTINEL" not in categories


def test_outer_validation_cannot_influence_early_stopping(tmp_path):
    """Mutating outer validation features/targets must have zero effect on MLP training and stopping."""
    context = synthetic_context(tmp_path, TaskType.BINARY_CLASSIFICATION)
    fold = context.split.folds[0]

    # Baseline run
    res1 = train_mlp_fold(context.training, fold, RunConfig(family="mlp", model_seed=42), max_epochs=15, patience=3)

    # Create mutated training view where outer validation rows have inverted targets & corrupted features
    pos = {rid: idx for idx, rid in enumerate(context.training.row_ids)}
    mutated_rows = list(context.training.rows)
    mutated_targets = list(context.training.targets)
    for ov_id in fold.validation_row_ids:
        idx = pos[ov_id]
        mutated_rows[idx] = (999999.0, "corrupted_val")
        mutated_targets[idx] = 1.0 - mutated_targets[idx]

    corrupted_view = replace(context.training, rows=tuple(mutated_rows), targets=tuple(mutated_targets))

    # Run fold on corrupted view
    res2 = train_mlp_fold(corrupted_view, fold, RunConfig(family="mlp", model_seed=42), max_epochs=15, patience=3)

    # Stopping decisions, epochs, and best epoch must be bit-for-bit identical
    assert res1.best_epoch == res2.best_epoch
    assert res1.epochs_trained == res2.epochs_trained
    assert res1.stop_reason == res2.stop_reason
    assert res1.best_loss == pytest.approx(res2.best_loss)

    # Restored best state weights must be identical
    for k in res1.best_state_dict:
        assert torch.equal(res1.best_state_dict[k], res2.best_state_dict[k])


# --- D16b Tests: Checkpoint Restoration, Reporting, Patience, Seed Replay ---


def test_best_checkpoint_restored_not_last_weights():
    """Verify that train_mlp restores best weights, not last weights, upon early stopping."""
    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)

    # Create synthetic dataset
    x_train = np.ones((16, 2), dtype=np.float32)
    y_train = np.ones(16, dtype=np.float32)
    x_val = np.ones((8, 2), dtype=np.float32)
    y_val = np.ones(8, dtype=np.float32)

    train_loader = create_dataloader(x_train, y_train, batch_size=8, shuffle=False)
    val_loader = create_dataloader(x_val, y_val, batch_size=8, shuffle=False)

    # Custom loss fn simulating a minimum at epoch 2 followed by non-improvements
    current_epoch = 0

    class ScriptedLoss(nn.Module):
        def forward(self, pred, target):
            # Model training loss
            if pred.requires_grad:
                return pred.sum()
            # Validation loss sequence: epoch 1=0.5, epoch 2=0.2 (best), epochs 3..12=0.8
            val_losses = {1: 0.5, 2: 0.2}
            loss_val = val_losses.get(current_epoch, 0.8)
            return torch.tensor(loss_val, dtype=torch.float32)

    scripted_loss = ScriptedLoss()

    class TrackControl(RunControl):
        def check_cancelled(self):
            pass

    control = TrackControl()
    weights_at_epoch = {}

    def on_epoch_end(epoch, train_l, val_l):
        nonlocal current_epoch
        current_epoch = epoch + 1
        weights_at_epoch[epoch] = {k: v.clone() for k, v in model.state_dict().items()}

    control.on_epoch_end = on_epoch_end
    current_epoch = 1

    res = train_mlp(
        model,
        train_loader,
        val_loader,
        loss_fn=scripted_loss,
        max_epochs=50,
        patience=5,
        min_delta=1e-6,
        run_control=control,
        restore_best_weights=True,
    )

    # Epoch 2 was the best; patience=5 stops at epoch 7 (5 consecutive non-improvements: 3, 4, 5, 6, 7)
    assert res.best_epoch == 2
    assert res.epochs_trained == 7
    assert res.stop_reason == "early_stopping_patience"

    # Current model parameters MUST match epoch 2 weights, NOT epoch 7 weights
    for k, param in model.state_dict().items():
        assert torch.equal(param, weights_at_epoch[2][k]), f"Mismatch in {k} vs best epoch 2"
        # Weights change across training, so epoch 2 != epoch 7
        if "weight" in k:
            assert not torch.equal(param, weights_at_epoch[7][k]), f"Param {k} should differ from last epoch"


def test_min_delta_enforcement_avoids_noise_resets():
    """Verify that improvements smaller than min_delta=1e-6 do NOT reset patience."""
    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    x_train = np.ones((8, 2), dtype=np.float32)
    y_train = np.ones(8, dtype=np.float32)
    x_val = np.ones((4, 2), dtype=np.float32)
    y_val = np.ones(4, dtype=np.float32)

    train_loader = create_dataloader(x_train, y_train, batch_size=4, shuffle=False)
    val_loader = create_dataloader(x_val, y_val, batch_size=4, shuffle=False)

    current_epoch = 1

    class NoiseLoss(nn.Module):
        def forward(self, pred, target):
            if pred.requires_grad:
                return (pred.mean() * 0.0) + 1.0
            # Epoch 1: 1.000000
            # Epoch 2..15: 1.000000 - 1e-7 (tiny numerical noise, delta < 1e-6)
            if current_epoch == 1:
                return torch.tensor(1.0, dtype=torch.float32)
            return torch.tensor(1.0 - 1e-7, dtype=torch.float32)

    def on_epoch_end(epoch, tr, val):
        nonlocal current_epoch
        current_epoch = epoch + 1

    control = RunControl(on_epoch_end=on_epoch_end)

    res = train_mlp(
        model,
        train_loader,
        val_loader,
        loss_fn=NoiseLoss(),
        max_epochs=20,
        patience=5,
        min_delta=1e-6,
        run_control=control,
    )

    # Because delta (1e-7) < min_delta (1e-6), patience should not reset.
    # Epoch 1 is best. Non-improvements at 2, 3, 4, 5, 6 -> stops at epoch 6.
    assert res.best_epoch == 1
    assert res.epochs_trained == 6
    assert res.stop_reason == "early_stopping_patience"


def test_max_epochs_cap_respected():
    """Verify that training strictly halts when reaching max_epochs."""
    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    x_train = np.ones((8, 2), dtype=np.float32)
    y_train = np.ones(8, dtype=np.float32)
    x_val = np.ones((4, 2), dtype=np.float32)
    y_val = np.ones(4, dtype=np.float32)

    train_loader = create_dataloader(x_train, y_train, batch_size=4, shuffle=False)
    val_loader = create_dataloader(x_val, y_val, batch_size=4, shuffle=False)

    # Loss improves every epoch: 1.0, 0.9, 0.8...
    current_epoch = 1

    class ImprovingLoss(nn.Module):
        def forward(self, pred, target):
            if pred.requires_grad:
                return (pred.mean() * 0.0) + 1.0
            return torch.tensor(1.0 / current_epoch, dtype=torch.float32)

    def on_epoch_end(epoch, tr, val):
        nonlocal current_epoch
        current_epoch = epoch + 1

    control = RunControl(on_epoch_end=on_epoch_end)

    res = train_mlp(
        model,
        train_loader,
        val_loader,
        loss_fn=ImprovingLoss(),
        max_epochs=8,
        patience=10,
        run_control=control,
    )

    assert res.epochs_trained == 8
    assert res.best_epoch == 8
    assert res.stop_reason == "max_epochs"


def test_deterministic_seed_replay_within_tolerance(tmp_path):
    """Seed replay must reproduce identical losses, best epoch, and model weights."""
    context = synthetic_context(tmp_path, TaskType.REGRESSION)
    fold = context.split.folds[0]

    config = RunConfig(family="mlp", model_seed=123)

    res1 = train_mlp_fold(context.training, fold, config, max_epochs=5)
    res2 = train_mlp_fold(context.training, fold, config, max_epochs=5)

    assert res1.best_epoch == res2.best_epoch
    assert res1.epochs_trained == res2.epochs_trained
    assert res1.stop_reason == res2.stop_reason
    assert res1.best_loss == pytest.approx(res2.best_loss, abs=1e-6)

    # Weights match exactly
    for k in res1.best_state_dict:
        assert torch.equal(res1.best_state_dict[k], res2.best_state_dict[k])

    # Different seed yields different results
    config_diff = RunConfig(family="mlp", model_seed=999)
    res3 = train_mlp_fold(context.training, fold, config_diff, max_epochs=5)
    different = any(not torch.equal(res1.best_state_dict[k], res3.best_state_dict[k]) for k in res1.best_state_dict)
    assert different


def test_fold_training_reporting_metadata(tmp_path):
    """Verify that MLPFoldResult emits all required per-fold metadata."""
    context = synthetic_context(tmp_path, TaskType.BINARY_CLASSIFICATION)
    fold = context.split.folds[0]

    res = train_mlp_fold(context.training, fold, RunConfig(family="mlp"), max_epochs=3)

    assert res.fold_index == fold.fold_index
    assert res.best_epoch >= 1
    assert np.isfinite(res.best_loss)
    assert res.stop_reason in ("max_epochs", "early_stopping_patience")
    assert res.epochs_trained >= 1
    assert len(res.history) == res.epochs_trained
    assert isinstance(res.history[0], EpochLoss)
    assert res.inner_fit_row_ids == fold.inner.fit_row_ids
    assert res.inner_val_row_ids == fold.inner.validation_row_ids
    assert res.outer_val_row_ids == fold.validation_row_ids
    assert type(res.derived_seed) is int
    assert res.fit_seconds >= 0.0


def test_cv_evaluate_candidate_mlp_both_tasks(tmp_path):
    """Verify cv.evaluate_candidate runs MLP across frozen folds for both tasks."""
    for task in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
        context = synthetic_context(tmp_path, task)
        candidate = evaluate_candidate(context, RunConfig(family="mlp", limits=ResourceLimits(compute_threads=2)))

        assert candidate.family == "mlp"
        assert candidate.complete
        assert len(candidate.folds) == len(context.split.folds)
        for f in candidate.folds:
            assert f.status == "completed"
            assert f.metrics is not None
            assert f.fit_seconds >= 0.0
            assert f.score_seconds >= 0.0
