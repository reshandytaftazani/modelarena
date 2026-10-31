"""Bounded task-specific model factories and model registry."""

from ml.models.classical import classical_families, make_classical, model_families
from ml.models.mlp import (
    ArrayDataset,
    MLP,
    create_dataloader,
    create_optimizer,
    get_task_loss,
    make_mlp,
)

__all__ = [
    "classical_families",
    "model_families",
    "make_classical",
    "ArrayDataset",
    "MLP",
    "create_dataloader",
    "create_optimizer",
    "get_task_loss",
    "make_mlp",
]
