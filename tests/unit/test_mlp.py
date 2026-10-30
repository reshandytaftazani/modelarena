"""Unit tests for PyTorch CPU MLP architecture, dataset/dataloader, and training loop."""

from dataclasses import replace
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.contracts import TaskType
from ml.models.mlp import (
    ArrayDataset,
    MLP,
    create_dataloader,
    create_optimizer,
    get_task_loss,
    make_mlp,
)
from ml.training.classical_suite import SuiteCancelled
from ml.training.mlp_loop import (
    EpochLoss,
    MLPTrainingResult,
    NonFiniteLossError,
    RunControl,
    TrainingCancelled,
    train_mlp,
)


# --- D15a: Architecture, Loss, Dataset, DataLoader, Optimizer ---


def test_make_mlp_default_architecture():
    model = make_mlp(in_features=10)
    assert isinstance(model, MLP)
    assert isinstance(model, nn.Module)
    # Default widths: 10 -> 64 -> ReLU -> 32 -> ReLU -> 1
    layers = list(model.net.children())
    assert len(layers) == 5
    assert isinstance(layers[0], nn.Linear) and layers[0].in_features == 10 and layers[0].out_features == 64
    assert isinstance(layers[1], nn.ReLU)
    assert isinstance(layers[2], nn.Linear) and layers[2].in_features == 64 and layers[2].out_features == 32
    assert isinstance(layers[3], nn.ReLU)
    assert isinstance(layers[4], nn.Linear) and layers[4].in_features == 32 and layers[4].out_features == 1


def test_make_mlp_configurable_hidden_dims():
    model = make_mlp(in_features=8, hidden_dims=(128, 64, 16))
    layers = list(model.net.children())
    assert len(layers) == 7
    assert layers[0].out_features == 128
    assert layers[2].out_features == 64
    assert layers[4].out_features == 16
    assert layers[6].out_features == 1


def test_make_mlp_bounded_dimensions_validation():
    with pytest.raises(ValueError, match="in_features"):
        make_mlp(in_features=0)
    with pytest.raises(ValueError, match="in_features"):
        make_mlp(in_features=-5)
    with pytest.raises(ValueError, match="hidden_dims"):
        make_mlp(in_features=10, hidden_dims=())
    with pytest.raises(ValueError, match="hidden_dims"):
        make_mlp(in_features=10, hidden_dims=(0,))
    with pytest.raises(ValueError, match="hidden_dims"):
        make_mlp(in_features=10, hidden_dims=(64, -1))
    with pytest.raises(ValueError, match="hidden_dims"):
        make_mlp(in_features=10, hidden_dims=(4096,))  # Exceeds max 2048 bound


def test_make_mlp_forward_scalar_output_shape():
    model = make_mlp(in_features=5, hidden_dims=(16, 8))
    batch_x = torch.randn(7, 5)
    output = model(batch_x)
    assert output.shape == (7, 1)

    single_x = torch.randn(1, 5)
    output_single = model(single_x)
    assert output_single.shape == (1, 1)


def test_make_mlp_deterministic_seeding_and_cpu_device():
    model1 = make_mlp(in_features=4, hidden_dims=(8,), seed=42)
    model2 = make_mlp(in_features=4, hidden_dims=(8,), seed=42)
    model3 = make_mlp(in_features=4, hidden_dims=(8,), seed=99)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
        assert p1.device.type == "cpu"

    different = any(not torch.equal(p1, p3) for p1, p3 in zip(model1.parameters(), model3.parameters()))
    assert different


def test_make_mlp_seed_validation():
    with pytest.raises(ValueError, match="seed"):
        make_mlp(in_features=4, seed=-1)
    with pytest.raises(ValueError, match="seed"):
        make_mlp(in_features=4, seed=2**32)


def test_task_loss_selection():
    bce = get_task_loss(TaskType.BINARY_CLASSIFICATION)
    assert isinstance(bce, nn.BCEWithLogitsLoss)

    mse = get_task_loss(TaskType.REGRESSION)
    assert isinstance(mse, nn.MSELoss)

    with pytest.raises(ValueError, match="task"):
        get_task_loss("unsupported_task")  # type: ignore


def test_loss_computation_shapes():
    bce = get_task_loss(TaskType.BINARY_CLASSIFICATION)
    logits = torch.tensor([[0.5], [-1.2], [2.3]], dtype=torch.float32)
    binary_targets = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)
    loss = bce(logits, binary_targets)
    assert loss.ndim == 0 and torch.isfinite(loss)

    mse = get_task_loss(TaskType.REGRESSION)
    preds = torch.tensor([[1.5], [2.2], [3.1]], dtype=torch.float32)
    reg_targets = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    loss = mse(preds, reg_targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_array_dataset_validation_and_conversion():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    y = np.array([0, 1, 0], dtype=np.int64)
    ds = ArrayDataset(x, y)
    assert len(ds) == 3
    bx, by = ds[0]
    assert isinstance(bx, torch.Tensor) and bx.dtype == torch.float32 and bx.device.type == "cpu"
    assert isinstance(by, torch.Tensor) and by.dtype == torch.float32 and by.device.type == "cpu"
    assert by.shape == (1,)

    # Mismatched lengths
    with pytest.raises(ValueError, match="Length mismatch"):
        ArrayDataset(np.zeros((3, 2)), np.zeros(4))


def test_create_dataloader_cpu_only_and_workers_zero():
    x = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=(64,)).astype(np.float32)
    loader = create_dataloader(x, y, batch_size=32, shuffle=True, seed=42)
    assert loader.num_workers == 0
    assert not loader.pin_memory

    batches = list(loader)
    assert len(batches) == 2
    b1_x, b1_y = batches[0]
    assert b1_x.shape == (32, 4)
    assert b1_y.shape == (32, 1)
    assert b1_x.device.type == "cpu"
    assert b1_y.device.type == "cpu"


def test_create_dataloader_deterministic_shuffle():
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.arange(20, dtype=np.float32)
    loader1 = create_dataloader(x, y, batch_size=5, shuffle=True, seed=123)
    loader2 = create_dataloader(x, y, batch_size=5, shuffle=True, seed=123)
    loader3 = create_dataloader(x, y, batch_size=5, shuffle=True, seed=999)

    order1 = [batch_y.numpy() for _, batch_y in loader1]
    order2 = [batch_y.numpy() for _, batch_y in loader2]
    order3 = [batch_y.numpy() for _, batch_y in loader3]

    for o1, o2 in zip(order1, order2):
        np.testing.assert_array_equal(o1, o2)

    different = any(not np.array_equal(o1, o3) for o1, o3 in zip(order1, order3))
    assert different


def test_create_optimizer_defaults():
    model = make_mlp(in_features=4)
    opt = create_optimizer(model)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == 1e-3

    with pytest.raises(ValueError, match="lr"):
        create_optimizer(model, lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        create_optimizer(model, lr=-0.01)


# --- D15b: Training Loop, Explicit Modes, Weighted Loss, Guards, RunControl ---


def test_train_mlp_switches_train_and_eval_modes():
    modes_seen = []

    class SpyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            modes_seen.append("train" if self.training else "eval")
            return self.linear(x)

    model = SpyModel()
    x_train = np.ones((8, 2), dtype=np.float32)
    y_train = np.ones(8, dtype=np.float32)
    x_val = np.ones((4, 2), dtype=np.float32)
    y_val = np.ones(4, dtype=np.float32)

    train_loader = create_dataloader(x_train, y_train, batch_size=4, shuffle=False)
    val_loader = create_dataloader(x_val, y_val, batch_size=4, shuffle=False)

    train_mlp(model, train_loader, val_loader, max_epochs=1, patience=5)

    assert "train" in modes_seen
    assert "eval" in modes_seen


def test_train_mlp_no_gradient_leakage_into_validation():
    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    optimizer = create_optimizer(model)

    x_train = np.random.randn(10, 2).astype(np.float32)
    y_train = np.random.randint(0, 2, size=10).astype(np.float32)
    x_val = np.random.randn(5, 2).astype(np.float32)
    y_val = np.random.randint(0, 2, size=5).astype(np.float32)

    train_loader = create_dataloader(x_train, y_train, batch_size=5, shuffle=False)
    val_loader = create_dataloader(x_val, y_val, batch_size=5, shuffle=False)

    # Spy on backward
    backward_called = 0
    orig_backward = torch.Tensor.backward

    def spy_backward(self, *args, **kwargs):
        nonlocal backward_called
        backward_called += 1
        return orig_backward(self, *args, **kwargs)

    torch.Tensor.backward = spy_backward
    try:
        result = train_mlp(model, train_loader, val_loader, optimizer=optimizer, max_epochs=2, patience=5)
        # 2 epochs * 2 train batches = exactly 4 backward calls. 0 from validation.
        assert backward_called == 4
    finally:
        torch.Tensor.backward = orig_backward


def test_train_mlp_weighted_epoch_loss_tracking():
    # Construct batches with known distinct sizes and target losses
    # 32 items in batch 1 with loss 1.0; 16 items in batch 2 with loss 4.0
    # True weighted loss: (32*1.0 + 16*4.0) / 48 = (32 + 64) / 48 = 96 / 48 = 2.0
    # Simple batch average would be (1.0 + 4.0) / 2 = 2.5 (incorrect)
    class FixedLoss(nn.Module):
        def forward(self, pred, target):
            # Return loss based on target marker while maintaining computational graph
            return (pred * 0.0 + target).mean()

    model = make_mlp(in_features=1, hidden_dims=(2,), seed=42)
    x = np.zeros((48, 1), dtype=np.float32)
    y = np.concatenate([np.full((32,), 1.0, dtype=np.float32), np.full((16,), 4.0, dtype=np.float32)])
    loader = create_dataloader(x, y, batch_size=32, shuffle=False)

    result = train_mlp(model, loader, val_loader=loader, loss_fn=FixedLoss(), max_epochs=1)
    assert np.isclose(result.history[0].train_loss, 2.0, atol=1e-5)
    assert np.isclose(result.history[0].val_loss, 2.0, atol=1e-5)


def test_train_mlp_detached_best_state_copies():
    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    x = np.random.randn(20, 2).astype(np.float32)
    y = np.ones(20, dtype=np.float32)
    loader = create_dataloader(x, y, batch_size=10, shuffle=False)

    result = train_mlp(model, loader, val_loader=loader, max_epochs=2, patience=5)
    best_state = result.best_state_dict

    # Verify best_state holds detached tensors, not references to model params
    param_tensor = next(iter(model.parameters()))
    best_tensor = next(iter(best_state.values()))
    assert best_tensor.data_ptr() != param_tensor.data_ptr()

    # Manually mutating model does not mutate saved best_state
    with torch.no_grad():
        param_tensor.fill_(999.0)
    assert not torch.equal(param_tensor, best_tensor)


def test_train_mlp_reduces_loss_on_separable_fixture():
    # Simple linearly separable classification
    torch.manual_seed(42)
    np.random.seed(42)
    x_pos = np.random.randn(30, 2).astype(np.float32) + 3.0
    x_neg = np.random.randn(30, 2).astype(np.float32) - 3.0
    x = np.vstack([x_pos, x_neg])
    y = np.array([1.0] * 30 + [0.0] * 30, dtype=np.float32)

    model = make_mlp(in_features=2, hidden_dims=(16, 8), seed=42)
    loader = create_dataloader(x, y, batch_size=16, shuffle=True, seed=42)
    val_loader = create_dataloader(x, y, batch_size=16, shuffle=False)

    result = train_mlp(model, loader, val_loader, max_epochs=15, patience=10)

    first_loss = result.history[0].train_loss
    final_loss = result.history[-1].train_loss
    assert final_loss < first_loss, f"Expected loss reduction: {first_loss} -> {final_loss}"
    assert final_loss < first_loss * 0.8, f"Expected at least 20% loss reduction: {first_loss} -> {final_loss}"


def test_train_mlp_finite_loss_guard_nan_or_inf():
    class NanLoss(nn.Module):
        def forward(self, pred, target):
            return torch.tensor(float("nan"), dtype=torch.float32)

    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    loader = create_dataloader(np.ones((10, 2), dtype=np.float32), np.ones(10, dtype=np.float32), batch_size=5)

    with pytest.raises(NonFiniteLossError):
        train_mlp(model, loader, loss_fn=NanLoss(), max_epochs=1)

    class InfLoss(nn.Module):
        def forward(self, pred, target):
            return torch.tensor(float("inf"), dtype=torch.float32)

    with pytest.raises(NonFiniteLossError):
        train_mlp(model, loader, loss_fn=InfLoss(), max_epochs=1)


def test_train_mlp_run_control_cancellation_batch_level():
    batch_count = 0

    def cancel_after_2_batches():
        nonlocal batch_count
        batch_count += 1
        return batch_count >= 2

    control = RunControl(cancelled=cancel_after_2_batches)
    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    loader = create_dataloader(np.ones((20, 2), dtype=np.float32), np.ones(20, dtype=np.float32), batch_size=2)

    with pytest.raises((TrainingCancelled, SuiteCancelled)):
        train_mlp(model, loader, run_control=control, max_epochs=5)

    # Cancelled before completing all 10 batches of epoch 1
    assert batch_count == 2


def test_train_mlp_run_control_cancellation_epoch_level():
    epoch_count = 0

    def cancel_on_second_epoch():
        return epoch_count >= 1

    def count_epoch(ep, tr_loss, val_loss):
        nonlocal epoch_count
        epoch_count += 1

    control = RunControl(cancelled=cancel_on_second_epoch, on_epoch_end=count_epoch)
    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    loader = create_dataloader(np.ones((10, 2), dtype=np.float32), np.ones(10, dtype=np.float32), batch_size=5)

    with pytest.raises((TrainingCancelled, SuiteCancelled)):
        train_mlp(model, loader, run_control=control, max_epochs=5)

    assert epoch_count == 1


def test_train_mlp_early_stopping_patience():
    # Validation loss that strictly worsens after epoch 2
    class WorseningLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def forward(self, pred, target):
            self.call_count += 1
            # Return increasing loss connected to pred
            return (pred * 0.0).sum() + torch.tensor(float(self.call_count), dtype=torch.float32)

    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    loader = create_dataloader(np.ones((10, 2), dtype=np.float32), np.ones(10, dtype=np.float32), batch_size=10)

    result = train_mlp(model, loader, val_loader=loader, loss_fn=WorseningLoss(), max_epochs=20, patience=3)
    assert result.stop_reason == "early_stopping_patience"
    assert result.best_epoch == 1
    assert result.epochs_trained == 4  # Best at 1, + 3 patience non-improvements = stops after 4


def test_train_mlp_scoped_compute_threads():
    model = make_mlp(in_features=2, hidden_dims=(4,), seed=42)
    loader = create_dataloader(np.ones((10, 2), dtype=np.float32), np.ones(10, dtype=np.float32), batch_size=5)

    orig = torch.get_num_threads()
    # Scoped execution with specific thread count
    train_mlp(model, loader, compute_threads=2, max_epochs=1)
    assert torch.get_num_threads() == orig

    with pytest.raises(ValueError, match="compute_threads"):
        train_mlp(model, loader, compute_threads=0, max_epochs=1)
    with pytest.raises(ValueError, match="compute_threads"):
        train_mlp(model, loader, compute_threads=-2, max_epochs=1)
