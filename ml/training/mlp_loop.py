"""PyTorch CPU MLP training loop with explicit modes, weighted loss, and RunControl."""

from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.contracts import TaskType
from ml.models.mlp import create_optimizer, get_task_loss
from ml.training.classical_suite import SuiteCancelled


class NonFiniteLossError(ValueError):
    """Raised when NaN or Inf loss occurs during training or validation."""


class TrainingCancelled(SuiteCancelled):
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
