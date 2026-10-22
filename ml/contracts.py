"""Immutable, versioned run settings. No ML or web libraries are imported here."""

from dataclasses import dataclass, field, fields
from enum import StrEnum


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Policy v1 ceilings; lower caps are allowed, increases require policy review.

    These are configuration contracts, not yet runtime enforcement. A future
    execution child applies the thread settings before importing numeric libraries.
    """

    policy_version: int = 1
    compute_threads: int = 4
    torch_interop_threads: int = 1
    estimator_n_jobs: int = 1
    dataloader_workers: int = 0
    active_jobs: int = 1
    cv_parallelism: int = 1
    upload_bytes: int = 20_000_000
    max_rows: int = 50_000
    max_columns: int = 200
    max_cell_bytes: int = 65_536
    max_encoded_features: int = 2048
    max_categories: int = 100
    preprocessing_bytes: int = 512 * 1024**2
    child_rss_bytes: int = 4 * 1024**3
    system_reserve_bytes: int = 4 * 1024**3
    sample_interval_seconds: float = 0.5
    memory_pressure_seconds: int = 10
    classical_fold_seconds: int = 180
    mlp_fold_seconds: int = 300
    cv_job_seconds: int = 45 * 60
    finalization_seconds: int = 10 * 60
    queue_size: int = 10
    max_epochs: int = 100
    max_diagnostic_points: int = 2000
    storage_bytes: int = 5 * 1024**3
    min_free_disk_bytes: int = 1024**3
    log_files: int = 5
    log_file_bytes: int = 5 * 1024**2

    def __post_init__(self) -> None:
        fixed = {"policy_version", "torch_interop_threads", "estimator_n_jobs",
                 "dataloader_workers", "active_jobs", "cv_parallelism",
                 "system_reserve_bytes", "sample_interval_seconds", "min_free_disk_bytes"}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "sample_interval_seconds":
                valid = type(value) in (int, float) and value == item.default
            elif item.name == "compute_threads":
                valid = type(value) is int and value in (2, 4, 6)
            elif item.name in fixed:
                valid = type(value) is int and value == item.default
            else:
                valid = type(value) is int and 0 < value <= item.default
            if not valid:
                raise ValueError(f"Invalid policy-v1 limit: {item.name}")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Shared run reproducibility settings; task-specific contracts arrive later."""

    schema_version: int = 1
    split_seed: int = 42
    model_seed: int = 42
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported run configuration schema")
        for name in ("split_seed", "model_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an unsigned 32-bit integer")
        if not isinstance(self.limits, ResourceLimits):
            raise ValueError("limits must be ResourceLimits")
