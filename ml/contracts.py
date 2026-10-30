"""Immutable, versioned run settings. No ML or web libraries are imported here."""

from dataclasses import dataclass, field, fields
from enum import StrEnum
import math


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
class InnerSplit:
    """Persisted MLP early-stopping partition inside one outer training set.

    ``fit_row_ids`` and ``validation_row_ids`` are ascending, disjoint and cover
    exactly the outer fold's training rows. ``expanded`` records that rounding
    20% could not keep both binary labels, so the smallest usable size was used.
    """

    fit_row_ids: tuple[int, ...] = field(repr=False)
    validation_row_ids: tuple[int, ...] = field(repr=False)
    expanded: bool
    seed: int

    def __post_init__(self) -> None:
        for name in ("fit_row_ids", "validation_row_ids"):
            _check_row_ids(getattr(self, name))
        if len(self.fit_row_ids) < 2 or len(self.validation_row_ids) < 2:
            raise ValueError("Inner partitions need at least two rows each")
        if set(self.fit_row_ids) & set(self.validation_row_ids):
            raise ValueError("Inner partitions must be disjoint")
        if type(self.expanded) is not bool:
            raise ValueError("Inner expansion flag must be boolean")
        _check_seed(self.seed, "inner seed")


@dataclass(frozen=True, slots=True)
class FoldManifest:
    """One frozen outer fold with its persisted inner early-stopping split."""

    fold_index: int
    fit_row_ids: tuple[int, ...] = field(repr=False)
    validation_row_ids: tuple[int, ...] = field(repr=False)
    inner: InnerSplit

    def __post_init__(self) -> None:
        if type(self.fold_index) is not int or self.fold_index < 0:
            raise ValueError("Fold index must be a nonnegative integer")
        for name in ("fit_row_ids", "validation_row_ids"):
            _check_row_ids(getattr(self, name))
        if len(self.fit_row_ids) < 2 or len(self.validation_row_ids) < 2:
            raise ValueError("Fold partitions need at least two rows each")
        if set(self.fit_row_ids) & set(self.validation_row_ids):
            raise ValueError("Fold partitions must be disjoint")
        if not isinstance(self.inner, InnerSplit):
            raise ValueError("Fold requires a persisted inner split")
        if set(self.inner.fit_row_ids) | set(self.inner.validation_row_ids) != set(self.fit_row_ids):
            raise ValueError("Inner split must cover exactly the outer training rows")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Immutable 80/20 partition plus frozen outer and inner fold identities.

    Only row identities, digests and reasons are stored; no feature values and
    no holdout content. ``manifest_sha256`` is verified by ``verify_manifest``.
    """

    schema_version: int
    split_seed: int
    algorithm: str
    dataset_sha256: str
    config_sha256: str
    row_count: int
    train_row_ids: tuple[int, ...] = field(repr=False)
    holdout_row_ids: tuple[int, ...] = field(repr=False)
    train_sha256: str
    holdout_sha256: str
    fold_count: int
    fold_reason: str
    fold_warning: str
    folds: tuple[FoldManifest, ...] = field(repr=False)
    holdout_reason: str
    manifest_sha256: str
    splitter_version: str = "unknown"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported split manifest schema")
        _check_seed(self.split_seed, "split seed")
        if self.algorithm not in ("stratified_kfold", "kfold"):
            raise ValueError("Unsupported fold algorithm")
        if not isinstance(self.splitter_version, str) or not self.splitter_version or len(self.splitter_version) > 100:
            raise ValueError("Splitter library version must be recorded")
        for name in ("dataset_sha256", "config_sha256", "train_sha256", "holdout_sha256", "manifest_sha256"):
            _check_digest(getattr(self, name), name)
        if type(self.row_count) is not int or self.row_count < 4:
            raise ValueError("Split requires at least four rows")
        _check_row_ids(self.train_row_ids)
        _check_row_ids(self.holdout_row_ids)
        if not self.train_row_ids or len(self.holdout_row_ids) < 2:
            raise ValueError("Holdout requires at least two rows")
        if set(self.train_row_ids) & set(self.holdout_row_ids):
            raise ValueError("Training and holdout rows must be disjoint")
        all_ids = set(self.train_row_ids) | set(self.holdout_row_ids)
        # Validate the supplied identities; never allocate from an untrusted
        # persisted row_count before the checksum has been checked.
        if len(all_ids) != self.row_count or max(all_ids) >= self.row_count:
            raise ValueError("Partitions must be exhaustive and within the row count")
        if type(self.fold_count) is not int or not 2 <= self.fold_count <= 5:
            raise ValueError("Between two and five folds are required")
        if not isinstance(self.folds, tuple) or len(self.folds) != self.fold_count:
            raise ValueError("Persisted folds must match the effective fold count")
        if any(not isinstance(fold, FoldManifest) for fold in self.folds):
            raise ValueError("Invalid fold record")
        if tuple(fold.fold_index for fold in self.folds) != tuple(range(self.fold_count)):
            raise ValueError("Fold indices must be ordered and complete")
        train = set(self.train_row_ids)
        for fold in self.folds:
            if set(fold.fit_row_ids) | set(fold.validation_row_ids) != train:
                raise ValueError("Each fold must partition exactly the training rows")
        validation_ids = [i for fold in self.folds for i in fold.validation_row_ids]
        if len(validation_ids) != len(train) or set(validation_ids) != train:
            raise ValueError("Every training row must be validated exactly once")
        if self.fold_reason == "preferred_five_folds":
            if self.fold_count != 5 or self.fold_warning:
                raise ValueError("Preferred five folds cannot carry an adaptation warning")
        elif self.fold_reason == f"reduced_from_five_folds:{self.fold_count}":
            if not self.fold_warning:
                raise ValueError("Reduced fold count requires a recorded warning")
        else:
            raise ValueError("Fold reason must state preference or the reduced count")
        if self.holdout_reason != "default_20_percent":
            raise ValueError("Only the default 20 percent holdout is supported in policy v1")


def _check_row_ids(values: object) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError("Row identities must be a nonempty tuple")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Row identities must be nonnegative integers")
    if tuple(sorted(values)) != values:
        raise ValueError("Row identities must be persisted in ascending order")
    if len(set(values)) != len(values):
        raise ValueError("Row identities must be unique")


def _check_seed(value: object, name: str) -> None:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an unsigned 32-bit integer")


def _check_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


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
    family: str = "dummy"
    primary_metric: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported run configuration schema")
        for name in ("split_seed", "model_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an unsigned 32-bit integer")
        if not isinstance(self.limits, ResourceLimits):
            raise ValueError("limits must be ResourceLimits")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("Model family must be nonempty")
        if self.primary_metric is not None and self.primary_metric not in (
            "f1", "average_precision", "rmse", "mae",
        ):
            raise ValueError("Unsupported primary metric")


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    value: float | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MetricBundle:
    """Confusion matrix uses actual rows/predicted columns, label order (0, 1)."""

    task: TaskType
    sample_count: int
    metrics: tuple[MetricValue, ...]
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]] | None = None

    def get(self, name: str) -> MetricValue:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Training records and frozen split metadata only; no holdout contents/path.

    Construct the view with training_from_manifest at the trusted data boundary.
    The evaluator rechecks the manifest checksum, configuration and row IDs.
    """

    training: TrainingView
    split: SplitManifest


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold_index: int
    validation_row_ids: tuple[int, ...] = field(repr=False)
    status: str
    metrics: MetricBundle | None
    fit_seconds: float
    score_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    name: str
    mean: float | None
    std: float | None
    valid_folds: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticPoint:
    row_id: int
    fold_index: int
    truth: float
    prediction: float
    probability: float | None = None
    residual: float | None = None


@dataclass(frozen=True, slots=True)
class OofDiagnostics:
    """Training-only display data; scalar metrics cover all OOF predictions."""

    partition: str
    sample_count: int
    points: tuple[DiagnosticPoint, ...]
    metrics: MetricBundle
    roc_curve: tuple[tuple[float, float], ...]
    pr_curve: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class CandidateResult:
    task: TaskType
    family: str
    model_seed: int
    split_sha256: str
    primary_metric: str
    direction: str
    expected_fold_indices: tuple[int, ...]
    folds: tuple[FoldResult, ...]
    summary: tuple[MetricSummary, ...]
    total_seconds: float
    run_config: RunConfig | None = None
    diagnostics: OofDiagnostics | None = None

    @property
    def complete(self) -> bool:
        return (bool(self.expected_fold_indices)
                and len(set(self.expected_fold_indices)) == len(self.expected_fold_indices)
                and tuple(f.fold_index for f in self.folds) == self.expected_fold_indices
                and all(f.status == "completed" and f.metrics is not None for f in self.folds))

    @property
    def eligible(self) -> bool:
        """Derive eligibility from actual folds, never a cached success flag."""
        if not self.complete:
            return False
        for fold in self.folds:
            try:
                value = fold.metrics.get(self.primary_metric).value
            except KeyError:
                return False
            if value is None or not math.isfinite(value):
                return False
        return True


@dataclass(frozen=True, slots=True)
class FrozenSelection:
    finalization_id: str
    scope_id: str
    experiment_id: str
    candidate_id: str
    candidate_digest: str
    family: str
    task: TaskType
    primary_metric: str
    primary_score: float
    direction: str
    split_sha256: str
    config_sha256: str
    model_seed: int
    resource_limits: ResourceLimits
    created_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.finalization_id, str) or not self.finalization_id:
            raise ValueError("Finalization ID required")
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError("Scope ID required")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("Candidate ID required")
        if not isinstance(self.candidate_digest, str) or len(self.candidate_digest) != 64:
            raise ValueError("Valid 64-character candidate digest required")
        if not isinstance(self.task, TaskType):
            raise ValueError("TaskType required")
        if not math.isfinite(self.primary_score):
            raise ValueError("Primary score must be finite")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError("Direction must be maximize or minimize")
        if not isinstance(self.resource_limits, ResourceLimits):
            raise ValueError("ResourceLimits required")


_CAPABILITY_TOKEN_MIN_LEN = 32


@dataclass(frozen=True, slots=True)
class HoldoutCapability:
    scope_id: str
    holdout_path: str
    _auth_token: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError("Scope ID required")
        if not isinstance(self.holdout_path, str) or not self.holdout_path:
            raise ValueError("Holdout path required")
        if not isinstance(self._auth_token, str) or len(self._auth_token) < _CAPABILITY_TOKEN_MIN_LEN:
            raise PermissionError("HoldoutCapability cannot be instantiated without a valid authorization token")


@dataclass(frozen=True, slots=True)
class FinalResult:
    finalization_id: str
    scope_id: str
    candidate_id: str
    task: TaskType
    primary_metric: str
    metrics: MetricBundle
    evaluation_seconds: float
    evaluated_at: str
