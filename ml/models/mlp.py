"""Configurable PyTorch CPU Multi-Layer Perceptron (MLP) and dataset utilities.

Strictly CPU-only execution with num_workers=0 DataLoader and deterministic seeding.
No CUDA paths.
"""

from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.contracts import TaskType

#: Maximum bounded hidden dimension according to resource constraints.
MAX_HIDDEN_DIM = 2048
MAX_HIDDEN_LAYERS = 8


class MLP(nn.Module):
    """Feedforward network with ReLU activations and a scalar output."""

    def __init__(self, in_features: int, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        # Scalar linear output (logits for classification, raw prediction for regression)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.to(torch.device("cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            raise ValueError("MLP only supports CPU tensors")
        return self.net(x.to(dtype=torch.float32))


def make_mlp(
    first: int | TaskType | None = None,
    second: int | Sequence[int] | None = None,
    hidden_dims: Sequence[int] = (64, 32),
    seed: int | None = None,
    task: TaskType | None = None,
    *,
    in_features: int | None = None,
) -> MLP:
    """Construct an MLP with bounded hidden dimensions and deterministic seed."""
    actual_task = task
    actual_in_features = in_features
    actual_hidden_dims = hidden_dims

    if isinstance(first, TaskType):
        actual_task = first
        if isinstance(second, int):
            actual_in_features = second
    elif isinstance(first, int):
        actual_in_features = first
        if isinstance(second, (tuple, list)):
            actual_hidden_dims = second

    if actual_in_features is None:
        raise ValueError("in_features must be specified as a positive integer")
    if type(actual_in_features) is not int or actual_in_features <= 0:
        raise ValueError("in_features must be a positive integer")

    dims = tuple(actual_hidden_dims)
    if not dims or len(dims) > MAX_HIDDEN_LAYERS:
        raise ValueError(f"hidden_dims length must be between 1 and {MAX_HIDDEN_LAYERS}")
    for dim in dims:
        if type(dim) is not int or dim <= 0 or dim > MAX_HIDDEN_DIM:
            raise ValueError(f"hidden_dims: each hidden dimension must be an integer between 1 and {MAX_HIDDEN_DIM}")

    if seed is not None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        torch.manual_seed(seed)

    model = MLP(actual_in_features, dims)
    if actual_task is not None:
        model.task = actual_task  # type: ignore[attr-defined]
    return model


def get_task_loss(task: TaskType) -> nn.Module:
    """Return task loss module: BCEWithLogitsLoss for classification, MSELoss for regression."""
    if task == TaskType.BINARY_CLASSIFICATION:
        return nn.BCEWithLogitsLoss()
    if task == TaskType.REGRESSION:
        return nn.MSELoss()
    raise ValueError("Unsupported task: expected TaskType.BINARY_CLASSIFICATION or TaskType.REGRESSION")


class ArrayDataset(Dataset):
    """In-memory dataset converting arrays/tensors to float32 CPU tensors."""

    def __init__(self, features: np.ndarray | torch.Tensor | Sequence, targets: np.ndarray | torch.Tensor | Sequence):
        feat_tensor = torch.as_tensor(features, dtype=torch.float32, device="cpu")
        targ_tensor = torch.as_tensor(targets, dtype=torch.float32, device="cpu")
        if len(feat_tensor) != len(targ_tensor):
            raise ValueError(f"Length mismatch: {len(feat_tensor)} features vs {len(targ_tensor)} targets")
        if targ_tensor.ndim == 1:
            targ_tensor = targ_tensor.unsqueeze(-1)
        self.features = feat_tensor
        self.targets = targ_tensor

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def create_dataloader(
    features: np.ndarray | torch.Tensor | Sequence,
    targets: np.ndarray | torch.Tensor | Sequence,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """Deterministic seeded CPU DataLoader strictly bounded to num_workers=0."""
    dataset = ArrayDataset(features, targets)
    generator = None
    if shuffle:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=False,
    )


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Adam:
    """Adam optimizer with standard defaults (lr=1e-3)."""
    if lr <= 0:
        raise ValueError("lr must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
