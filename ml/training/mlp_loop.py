"""PyTorch CPU MLP training loop with explicit modes, weighted loss, and RunControl."""

from dataclasses import dataclass, field
import hashlib
from time import perf_counter
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.contracts import (
    FoldManifest,
    ResourceLimits,
    RunConfig,
    TaskType,
    TrainingContext,
    TrainingView,
)
from ml.models.mlp import (
    MLP,
    create_dataloader,
    create_optimizer,
    get_task_loss,
    make_mlp,
)
from ml.preprocessing.factory import FoldPreprocessor


class NonFiniteLossError(ValueError):
    """Raised when NaN or Inf loss occurs during training or validation."""


class TrainingCancelled(Exception):
    """Raised when training is cancelled via RunControl."""


@dataclass
class RunControl:
    """Execution control supporting batch/epoch cancellation checks and callbacks."""

    cancelled: Callable[[], bool] = field(default=lambda: False)
    on_epoch_end: Callable[[int, float, float | None], None] | None = None
    on_batch_end: Callable[[int, int], None] | None = None

    def is_cancelled(self) -> bool:
        return bool(self.cancelled())

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise TrainingCancelled("MLP training cancelled by RunControl")


@dataclass(frozen=True, slots=True)
class EpochLoss:
    """Loss record for one training epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None


@dataclass(frozen=True, slots=True)
class MLPTrainingResult:
    """Outcome of an MLP training run with detached best state."""

    best_epoch: int
    best_loss: float
    best_state_dict: dict[str, torch.Tensor]
    history: tuple[EpochLoss, ...]
    stop_reason: str
    epochs_trained: int


def _detach_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU copy of the model state dict."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    loss_fn: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-6,
    compute_threads: int | None = None,
    run_control: RunControl | Callable[[], bool] | None = None,
    restore_best_weights: bool = True,
) -> MLPTrainingResult:
    """Train an MLP on CPU with bounded epochs, early stopping, and cancellation checks."""
    if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("train_loader must not be empty")
    if val_loader is not None and len(val_loader.dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("val_loader must not be empty when provided")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    if compute_threads is not None and (type(compute_threads) is not int or compute_threads <= 0):
        raise ValueError("compute_threads must be a positive integer")

    if run_control is None:
        control = RunControl()
    elif callable(run_control):
        control = RunControl(cancelled=run_control)
    elif isinstance(run_control, RunControl):
        control = run_control
    else:
        raise TypeError("run_control must be a RunControl instance or a callable returning bool")

    if loss_fn is None:
        task = getattr(model, "task", TaskType.BINARY_CLASSIFICATION)
        loss_fn = get_task_loss(task)

    if optimizer is None:
        optimizer = create_optimizer(model)

    orig_threads = torch.get_num_threads()
    if compute_threads is not None:
        torch.set_num_threads(compute_threads)

    best_epoch = 1
    best_loss = float("inf")
    best_state_dict = _detach_state_dict(model)
    patience_counter = 0
    history: list[EpochLoss] = []
    stop_reason = "max_epochs"
    epochs_trained = 0

    try:
        for epoch in range(1, max_epochs + 1):
            control.check_cancelled()

            # --- Training phase ---
            model.train()
            total_train_loss = 0.0
            total_train_samples = 0

            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                control.check_cancelled()
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = loss_fn(preds, batch_y)

                loss_val = float(loss.item())
                if not np.isfinite(loss_val) or torch.isnan(loss) or torch.isinf(loss):
                    raise NonFiniteLossError(f"Non-finite train loss encountered at epoch {epoch}, batch {batch_idx}: {loss_val}")

                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()

                batch_size = len(batch_x)
                total_train_loss += loss_val * batch_size
                total_train_samples += batch_size

                if control.on_batch_end is not None:
                    control.on_batch_end(epoch, batch_idx)

            weighted_train_loss = total_train_loss / max(1, total_train_samples)

            # --- Validation phase ---
            weighted_val_loss = None
            if val_loader is not None:
                model.eval()
                total_val_loss = 0.0
                total_val_samples = 0

                with torch.no_grad():
                    for batch_idx, (val_x, val_y) in enumerate(val_loader):
                        control.check_cancelled()
                        preds = model(val_x)
                        loss = loss_fn(preds, val_y)

                        loss_val = float(loss.item())
                        if not np.isfinite(loss_val) or torch.isnan(loss) or torch.isinf(loss):
                            raise NonFiniteLossError(f"Non-finite val loss encountered at epoch {epoch}, batch {batch_idx}: {loss_val}")

                        batch_size = len(val_x)
                        total_val_loss += loss_val * batch_size
                        total_val_samples += batch_size

                weighted_val_loss = total_val_loss / max(1, total_val_samples)
                comparison_loss = weighted_val_loss
            else:
                comparison_loss = weighted_train_loss

            epochs_trained = epoch
            record = EpochLoss(epoch, weighted_train_loss, weighted_val_loss)
            history.append(record)

            if control.on_epoch_end is not None:
                control.on_epoch_end(epoch, weighted_train_loss, weighted_val_loss)

            # Early stopping improvement check: strictly lower by > min_delta (1e-6)
            if epoch == 1 or comparison_loss < best_loss - min_delta:
                best_loss = comparison_loss
                best_epoch = epoch
                best_state_dict = _detach_state_dict(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if val_loader is not None and patience_counter >= patience:
                    stop_reason = "early_stopping_patience"
                    break

            control.check_cancelled()
    finally:
        if compute_threads is not None:
            torch.set_num_threads(orig_threads)

    if restore_best_weights and best_state_dict:
        model.load_state_dict(best_state_dict)

    return MLPTrainingResult(
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_state_dict=best_state_dict,
        history=tuple(history),
        stop_reason=stop_reason,
        epochs_trained=epochs_trained,
    )


def derive_fold_seed(model_seed: int, family: str, fold_index: int) -> int:
    """Derive deterministic seed from model seed, family, and fold index."""
    payload = f"{model_seed}:{family}:{fold_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % (2**32)


def subset_view(view: TrainingView, ids: tuple[int, ...], positions: dict[int, int] | None = None) -> TrainingView:
    """Subset a TrainingView by given row IDs without leaking non-selected rows."""
    if positions is None:
        positions = {row_id: index for index, row_id in enumerate(view.row_ids)}
    return TrainingView(
        view.config,
        ids,
        tuple(view.rows[positions[i]] for i in ids),
        tuple(view.targets[positions[i]] for i in ids),
        view.label_mapping,
    )


@dataclass(frozen=True, slots=True)
class MLPFoldResult:
    """Outcome of an MLP fold training run with restored best checkpoint and full metadata."""

    fold_index: int
    model: MLP
    preprocessor: FoldPreprocessor
    training_result: MLPTrainingResult
    inner_fit_row_ids: tuple[int, ...]
    inner_val_row_ids: tuple[int, ...]
    outer_val_row_ids: tuple[int, ...]
    derived_seed: int
    fit_seconds: float

    @property
    def best_epoch(self) -> int:
        return self.training_result.best_epoch

    @property
    def best_loss(self) -> float:
        return self.training_result.best_loss

    @property
    def stop_reason(self) -> str:
        return self.training_result.stop_reason

    @property
    def epochs_trained(self) -> int:
        return self.training_result.epochs_trained

    @property
    def history(self) -> tuple[EpochLoss, ...]:
        return self.training_result.history

    @property
    def best_state_dict(self) -> dict[str, torch.Tensor]:
        return self.training_result.best_state_dict


def train_mlp_fold(
    training_or_context: TrainingView | TrainingContext,
    manifest_or_fold_index: FoldManifest | int,
    config: RunConfig = RunConfig(family="mlp"),
    *,
    loss_fn: nn.Module | None = None,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-6,
    compute_threads: int | None = None,
    run_control: RunControl | Callable[[], bool] | None = None,
    preprocessor: FoldPreprocessor | None = None,
    holdout_row_ids: tuple[int, ...] | set[int] | None = None,
) -> MLPFoldResult:
    """Train MLP on a single fold with fold-local preprocessing and early stopping.

    Guarantees:
    - Preprocessor fits strictly on inner-fit rows only.
    - Inner validation drives early stopping (patience 10, min_delta 1e-6, max 100 epochs).
    - Neither outer validation nor holdout rows touch preprocessing fit or gradient updates.
    - Outer validation cannot influence early stopping decisions.
    - Best checkpoint (not last epoch) is restored upon completion.
    - Deterministic seed replay is guaranteed.
    """
    if isinstance(training_or_context, TrainingContext):
        view = training_or_context.training
        split = training_or_context.split
        holdout_ids = split.holdout_row_ids if holdout_row_ids is None else holdout_row_ids
        if isinstance(manifest_or_fold_index, int):
            manifest = split.folds[manifest_or_fold_index]
        else:
            manifest = manifest_or_fold_index
    elif isinstance(training_or_context, TrainingView):
        view = training_or_context
        holdout_ids = holdout_row_ids
        if isinstance(manifest_or_fold_index, int):
            raise TypeError("FoldManifest required when TrainingView is passed")
        manifest = manifest_or_fold_index
    else:
        raise TypeError("training_or_context must be TrainingView or TrainingContext")

    # --- Strict fold boundaries and leakage guards ---
    inner = manifest.inner
    inner_fit_set = set(inner.fit_row_ids)
    inner_val_set = set(inner.validation_row_ids)
    outer_val_set = set(manifest.validation_row_ids)

    if not inner_fit_set.isdisjoint(inner_val_set):
        raise ValueError("Inner fit and inner validation row IDs must be disjoint")
    if (inner_fit_set | inner_val_set) != set(manifest.fit_row_ids):
        raise ValueError("Inner split partitions must exactly cover outer fit rows")
    if not inner_fit_set.isdisjoint(outer_val_set):
        raise ValueError("Inner fit row IDs overlap with outer validation row IDs")
    if not inner_val_set.isdisjoint(outer_val_set):
        raise ValueError("Inner validation row IDs overlap with outer validation row IDs")

    if holdout_ids:
        holdout_set = set(holdout_ids)
        if not inner_fit_set.isdisjoint(holdout_set):
            raise ValueError("Inner fit contains holdout row IDs")
        if not inner_val_set.isdisjoint(holdout_set):
            raise ValueError("Inner validation contains holdout row IDs")
        if not outer_val_set.isdisjoint(holdout_set):
            raise ValueError("Outer validation contains holdout row IDs")

    positions = {row_id: index for index, row_id in enumerate(view.row_ids)}
    inner_fit_view = subset_view(view, inner.fit_row_ids, positions)
    inner_val_view = subset_view(view, inner.validation_row_ids, positions)

    # --- Preprocessor fits on inner-fit data ONLY ---
    if preprocessor is None:
        preprocessor = FoldPreprocessor(config.limits)

    x_inner_fit = preprocessor.fit_transform(inner_fit_view, dtype=np.float32)
    x_inner_val = preprocessor.transform(inner_val_view, dtype=np.float32)

    y_inner_fit = np.asarray(inner_fit_view.targets, dtype=np.float32)
    y_inner_val = np.asarray(inner_val_view.targets, dtype=np.float32)

    derived_seed = derive_fold_seed(config.model_seed, config.family or "mlp", manifest.fold_index)
    torch.manual_seed(derived_seed)
    np.random.seed(derived_seed)

    train_loader = create_dataloader(
        x_inner_fit,
        y_inner_fit,
        batch_size=32,
        shuffle=True,
        seed=derived_seed,
    )
    val_loader = create_dataloader(
        x_inner_val,
        y_inner_val,
        batch_size=32,
        shuffle=False,
    )

    in_features = x_inner_fit.shape[1]
    task = view.config.task
    model = make_mlp(in_features=in_features, seed=derived_seed, task=task)
    optimizer = create_optimizer(model, lr=1e-3)

    if loss_fn is None:
        loss_fn = get_task_loss(task)

    threads = compute_threads if compute_threads is not None else config.limits.compute_threads

    started = perf_counter()
    training_res = train_mlp(
        model,
        train_loader,
        val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        compute_threads=threads,
        run_control=run_control,
        restore_best_weights=True,
    )
    fit_seconds = perf_counter() - started

    return MLPFoldResult(
        fold_index=manifest.fold_index,
        model=model,
        preprocessor=preprocessor,
        training_result=training_res,
        inner_fit_row_ids=inner.fit_row_ids,
        inner_val_row_ids=inner.validation_row_ids,
        outer_val_row_ids=manifest.validation_row_ids,
        derived_seed=derived_seed,
        fit_seconds=fit_seconds,
    )
