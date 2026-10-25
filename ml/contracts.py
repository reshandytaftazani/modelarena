"""Immutable, versioned run settings. No ML or web libraries are imported here."""

from dataclasses import dataclass, field, fields
from enum import StrEnum


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


class FeatureType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class FeatureColumn:
    name: str
    dtype: FeatureType

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Feature name must be nonempty")
        if not isinstance(self.dtype, FeatureType):
            raise ValueError("Feature type must be explicitly declared")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Ordered declared types, never inferred from the complete dataset."""

    columns: tuple[FeatureColumn, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.columns, tuple) or not self.columns:
            raise ValueError("Select at least one feature")
        if any(not isinstance(c, FeatureColumn) for c in self.columns):
            raise ValueError("Invalid feature declaration")
        if len({c.name for c in self.columns}) != len(self.columns):
            raise ValueError("Feature names must be unique")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    task: TaskType
    target: str
    features: FeatureSchema
    positive_label: str | None = None
    acknowledge_duplicates: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskType) or not isinstance(self.features, FeatureSchema):
            raise ValueError("Explicit task and feature schema required")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("Target name must be nonempty")
        if self.target in {c.name for c in self.features.columns}:
            raise ValueError("Target cannot also be a feature")
        if type(self.acknowledge_duplicates) is not bool:
            raise ValueError("Duplicate acknowledgement must be boolean")
        if self.task == TaskType.BINARY_CLASSIFICATION:
            if not isinstance(self.positive_label, str) or not self.positive_label.strip():
                raise ValueError("Explicit positive label required")
        elif self.positive_label is not None:
            raise ValueError("Regression does not have a positive label")


FeatureValue = float | str | bool | None


@dataclass(frozen=True, slots=True)
class ConfiguredDataset:
    """Private validated records; not an analytical response or profile input."""

    config: DatasetConfig
    rows: tuple[tuple[FeatureValue, ...], ...] = field(repr=False)
    targets: tuple[int | float, ...] = field(repr=False)
    label_mapping: tuple[tuple[str, int], ...]
    excluded_columns: tuple[str, ...]
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingView:
    """Only selected training records; no parent dataset or holdout reference.

    Internal trusted boundary. Day 4 will supply persisted training indices;
    arbitrary client indices must never be accepted by a transport endpoint.
    """

    config: DatasetConfig
    row_ids: tuple[int, ...] = field(repr=False)
    rows: tuple[tuple[FeatureValue, ...], ...] = field(repr=False)
    targets: tuple[int | float, ...] = field(repr=False)
    label_mapping: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, DatasetConfig):
            raise ValueError("Training view requires a dataset configuration")
        if not all(isinstance(v, tuple) for v in (self.row_ids, self.rows, self.targets, self.label_mapping)):
            raise ValueError("Training records must be immutable tuples")
        if not self.rows or len(self.rows) != len(self.targets) or len(self.rows) != len(self.row_ids):
            raise ValueError("Training records and identities must align and be nonempty")
        if any(type(i) is not int or i < 0 for i in self.row_ids) or len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("Training identities must be unique nonnegative integers")
        if any(not isinstance(row, tuple) or len(row) != len(self.config.features.columns) for row in self.rows):
            raise ValueError("Training records must match the feature schema")


@dataclass(frozen=True, slots=True)
class NumericSummary:
    count: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    quantiles: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class FeatureProfile:
    name: str
    dtype: FeatureType
    missing_count: int
    missing_fraction: float
    distinct_count: int
    numeric: NumericSummary | None
    top_counts: tuple[tuple[str | bool, int], ...]


@dataclass(frozen=True, slots=True)
class TargetProfile:
    task: TaskType
    class_counts: tuple[tuple[int, int], ...]
    minority_ratio: float | None
    numeric: NumericSummary | None


@dataclass(frozen=True, slots=True)
class DuplicateCounts:
    """Extra rows exclude the first occurrence; groups count repeated keys."""

    feature_groups: int
    feature_extra_rows: int
    full_row_groups: int
    full_row_extra_rows: int
    conflicting_label_groups: int


@dataclass(frozen=True, slots=True)
class ProfileWarning:
    code: str
    message: str
    feature: str | None = None
    heuristic: bool = True


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    partition: str
    row_count: int
    feature_count: int
    features: tuple[FeatureProfile, ...]
    target: TargetProfile
    duplicates: DuplicateCounts
    warnings: tuple[ProfileWarning, ...]
    warnings_truncated: bool


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
