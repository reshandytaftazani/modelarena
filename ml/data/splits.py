"""Frozen 80/20 partitions and fold identities; no feature values are stored.

Row identities are positional indices into the validated dataset, assigned
before splitting and persisted in ascending order, so no reorder drift is
possible. Binary classification uses a stratified holdout and stratified outer
folds; regression uses a plain shuffled split. Five folds are preferred and the
largest feasible lower count is recorded with a warning. Every inner MLP
early-stopping split is derived from one outer training partition only, so
inner and outer validation never overlap. Duplicate acknowledgement is an
integrity gate, not proof of row independence.
"""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

from ml.contracts import (
    ConfiguredDataset, DatasetConfig, FoldManifest, InnerSplit, RunConfig, SplitManifest,
    TaskType, TrainingView,
)
from ml.data.validation import CsvValidationError, configure_dataset, training_view
from ml.data.csv_io import ValidatedDataset

MANIFEST_VERSION = 1
HOLDOUT_FRACTION = 0.2
PREFERRED_FOLDS = 5
INNER_VALIDATION_FRACTION = 0.2
INNER_SEED_STEP = 101
_ALGORITHMS = {TaskType.BINARY_CLASSIFICATION: "stratified_kfold", TaskType.REGRESSION: "kfold"}


def _infeasible(message: str) -> CsvValidationError:
    # Messages must never quote labels, values or file paths.
    return CsvValidationError("split_infeasible", message)


def _inner_infeasible() -> CsvValidationError:
    return CsvValidationError(
        "inner_split_infeasible",
        "The training partition cannot provide a two-label inner early-stopping split.",
    )


def _seed(base: int, fold_index: int) -> int:
    return (base + INNER_SEED_STEP * (fold_index + 1)) % 2**32


def _partition_digest(data: ValidatedDataset, label: str, row_ids: tuple[int, ...]) -> str:
    payload = json.dumps(
        {"columns": data.columns, "partition": label,
         "records": [(i, data.rows[i]) for i in row_ids]},
        separators=(",", ":"), sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_digest(config: DatasetConfig) -> str:
    payload = json.dumps({
        "task": str(config.task),
        "target": config.target,
        "features": [[column.name, str(column.dtype)] for column in config.features.columns],
        "positive_label": config.positive_label,
        "acknowledge_duplicates": config.acknowledge_duplicates,
        "schema_version": MANIFEST_VERSION,
    }, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inner_labels(labels: list[int], size: int, seed: int) -> tuple[list[int], list[int]] | None:
    """Stratified inner positions keeping both labels on each side."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    try:
        fit, validation = next(splitter.split(np.zeros((len(labels), 1)), labels))
    except ValueError:
        return None
    fit_labels = {labels[i] for i in fit}
    validation_labels = {labels[i] for i in validation}
    if len(fit_labels) < 2 or len(validation_labels) < 2 or len(fit) < 2 or len(validation) < 2:
        return None
    return list(fit), list(validation)


def create_inner_split(
    fit_row_ids: tuple[int, ...], targets: tuple[int | float, ...], task: TaskType, seed: int,
) -> InnerSplit:
    """Freeze the inner early-stopping split of one outer training partition.

    20% is used when rounding keeps at least two rows and both binary labels on
    each side; otherwise the smallest usable validation size is chosen and
    ``expanded`` is set. Preprocessing fits these fit rows only.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Inner seed must be an unsigned 32-bit integer")
    if not isinstance(fit_row_ids, tuple) or len(fit_row_ids) < 4:
        raise _inner_infeasible()
    if len(set(fit_row_ids)) != len(fit_row_ids):
        raise _inner_infeasible()
    if any(type(i) is not int or not 0 <= i < len(targets) for i in fit_row_ids):
        raise _inner_infeasible()
    if task not in _ALGORITHMS:
        raise ValueError("Unsupported task")
    labels = [targets[i] for i in fit_row_ids]
    count = len(labels)
    preferred = math.ceil(INNER_VALIDATION_FRACTION * count)
    minimum = max(2, preferred)
    if task == TaskType.REGRESSION:
        if minimum > count - 2:
            raise _inner_infeasible()
        fit, validation = train_test_split(
            np.arange(count), test_size=minimum, random_state=seed, shuffle=True,
        )
        return InnerSplit(tuple(sorted(fit_row_ids[int(i)] for i in fit)),
                          tuple(sorted(fit_row_ids[int(i)] for i in validation)), minimum > preferred, seed)
    if set(labels) != {0, 1} or min(labels.count(0), labels.count(1)) < 2:
        raise _inner_infeasible()
    pair = _inner_labels(labels, minimum, seed)
    size = minimum
    if pair is None:
        # For two classes, stratified allocation is monotone up to half the
        # rows, where both labels are guaranteed given >=2 examples each.
        # Binary search avoids thousands of full shuffled allocations for a
        # 50k-row upload with a tiny minority. Persist the smallest valid size.
        lower, upper = minimum + 1, count // 2
        while lower <= upper:
            candidate = (lower + upper) // 2
            result = _inner_labels(labels, candidate, seed)
            if result is None:
                lower = candidate + 1
            else:
                pair, size = result, candidate
                upper = candidate - 1
    if pair is None:
        raise _inner_infeasible()
    fit, validation = pair
    return InnerSplit(tuple(sorted(fit_row_ids[i] for i in fit)),
                      tuple(sorted(fit_row_ids[i] for i in validation)),
                      size > preferred, seed)


def _feasible_folds(
    train_ids: tuple[int, ...], labels: list[int | float], task: TaskType, folds: int, seed: int,
) -> tuple[FoldManifest, ...] | None:
    targets = tuple(labels)
    labels = [targets[i] for i in train_ids]
    matrix = np.zeros((len(train_ids), 1))
    if task == TaskType.BINARY_CLASSIFICATION:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        minority = min(labels.count(0), labels.count(1))
        if minority < folds:
            return None
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    try:
        pairs = list(splitter.split(matrix, labels))
    except ValueError:
        return None
    manifests: list[FoldManifest] = []
    for fold_index, (fit, validation) in enumerate(pairs):
        fit_ids = tuple(sorted(train_ids[i] for i in fit))
        validation_ids = tuple(sorted(train_ids[i] for i in validation))
        if len(fit_ids) < 2 or len(validation_ids) < 2:
            return None
        if task == TaskType.BINARY_CLASSIFICATION:
            fit_labels = {labels[i] for i in fit}
            validation_labels = {labels[i] for i in validation}
            if len(fit_labels) < 2 or len(validation_labels) < 2:
                return None
        try:
            inner = create_inner_split(fit_ids, targets, task, _seed(seed, fold_index))
        except CsvValidationError:
            return None
        manifests.append(FoldManifest(fold_index, fit_ids, validation_ids, inner))
    return tuple(manifests)


def _holdout_indices(
    targets: tuple[int | float, ...], task: TaskType, seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = len(targets)
    positions = np.arange(count)
    holdout_size = max(2, math.ceil(HOLDOUT_FRACTION * count))
    if holdout_size >= count:
        raise _infeasible("The dataset is too small for a training and holdout partition.")
    if task == TaskType.BINARY_CLASSIFICATION:
        labels = list(targets)
        if min(labels.count(0), labels.count(1)) < 2:
            raise _infeasible("Both labels need at least two rows for a stratified holdout.")
        try:
            train, holdout = train_test_split(
                positions, test_size=holdout_size, random_state=seed, shuffle=True, stratify=labels,
            )
        except ValueError:
            raise _infeasible(
                "The holdout cannot contain both labels with enough rows; provide more minority rows."
            ) from None
        if len({labels[i] for i in holdout}) < 2 or len({labels[i] for i in train}) < 2:
            raise _infeasible("Every partition must contain both labels.")
    else:
        train, holdout = train_test_split(
            positions, test_size=holdout_size, random_state=seed, shuffle=True,
        )
        if len(train) < 4 or len(holdout) < 2:
            raise _infeasible("Regression needs at least two holdout rows and four training rows.")
    return tuple(sorted(int(i) for i in train)), tuple(sorted(int(i) for i in holdout))


def _manifest_digest(manifest: SplitManifest) -> str:
    payload = asdict(manifest)
    del payload["manifest_sha256"]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def create_split(data: ValidatedDataset, config: DatasetConfig, run: RunConfig = RunConfig()) -> SplitManifest:
    """Freeze the default 80/20 partition and every fold identity.

    Integrity and duplicate acknowledgement run first, so an unacknowledged
    repeat can never reach a partition. No feature values are copied.
    """
    configured: ConfiguredDataset = configure_dataset(data, config)
    task = configured.config.task
    if not isinstance(run, RunConfig):
        raise ValueError("A run configuration is required")
    seed = run.split_seed
    labels = list(configured.targets)
    if task == TaskType.BINARY_CLASSIFICATION:
        # Scope is content/target/task, independent of positive-label polarity.
        stable_mapping = {encoded: rank for rank, (_, encoded) in enumerate(sorted(configured.label_mapping))}
        labels = [stable_mapping[value] for value in configured.targets]
    train_ids, holdout_ids = _holdout_indices(tuple(labels), task, seed)
    folds = None
    for count in range(PREFERRED_FOLDS, 1, -1):
        folds = _feasible_folds(train_ids, labels, task, count, seed)
        if folds is not None:
            break
    if folds is None:
        raise _infeasible(
            "No feasible 5/4/3/2-fold scheme: every validation and inner fit needs at least "
            "two rows, and binary partitions need both labels. Provide more independent rows."
        )
    fold_count = len(folds)
    reason = ("preferred_five_folds" if fold_count == PREFERRED_FOLDS
              else f"reduced_from_five_folds:{fold_count}")
    warning = ("" if fold_count == PREFERRED_FOLDS else
               f"five folds infeasible for this data; largest feasible count {fold_count} recorded")
    manifest = SplitManifest(
        schema_version=MANIFEST_VERSION,
        split_seed=seed,
        algorithm=_ALGORITHMS[task],
        dataset_sha256=configured.canonical_sha256,
        config_sha256=_config_digest(configured.config),
        row_count=len(configured.targets),
        train_row_ids=train_ids,
        holdout_row_ids=holdout_ids,
        train_sha256=_partition_digest(data, "train", train_ids),
        holdout_sha256=_partition_digest(data, "holdout", holdout_ids),
        fold_count=fold_count,
        fold_reason=reason,
        fold_warning=warning,
        folds=folds,
        holdout_reason="default_20_percent",
        manifest_sha256="0" * 64,
        splitter_version=sklearn.__version__,
    )
    return replace(manifest, manifest_sha256=_manifest_digest(manifest))


def verify_manifest(manifest: SplitManifest) -> bool:
    """Recompute the digest so tampered indices or reasons cannot pass."""
    if not isinstance(manifest, SplitManifest):
        raise TypeError("A SplitManifest is required")
    return manifest.manifest_sha256 == _manifest_digest(manifest)


def manifest_json(manifest: SplitManifest) -> str:
    """Canonical, sorted JSON used for persisted split records."""
    if not isinstance(manifest, SplitManifest):
        raise TypeError("A SplitManifest is required")
    return json.dumps(asdict(manifest), separators=(",", ":"), sort_keys=True)


def write_manifest(manifest: SplitManifest, path: Path) -> Path:
    """Publish a complete private record without replacing any frozen record.

    A closed, flushed temporary file is linked atomically into place. Existing
    identical bytes are an idempotent retry; different bytes are a conflict.
    Paths must be service-owned on a local disk, never client filenames.
    """
    if not verify_manifest(manifest):
        raise CsvValidationError("invalid_manifest", "Split checksum does not match.")
    content = (manifest_json(manifest) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".split-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as source:
            source.write(content)
            source.flush()
            os.fsync(source.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open("rb") as existing:
                if existing.read(len(content) + 1) != content:
                    raise CsvValidationError("split_conflict", "A different frozen split already exists.") from None
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def read_manifest(path: Path) -> SplitManifest:
    """Read private JSON, enforcing bounds, structure and checksum before use."""
    with Path(path).open("rb") as source:
        content = source.read(16_000_001)
    try:
        if len(content) > 16_000_000:
            raise ValueError("Manifest size exceeded")
        payload = json.loads(content, object_pairs_hook=_unique_keys)
        folds = []
        for raw in payload.pop("folds"):
            raw["fit_row_ids"] = tuple(raw["fit_row_ids"])
            raw["validation_row_ids"] = tuple(raw["validation_row_ids"])
            inner = raw["inner"]
            inner["fit_row_ids"] = tuple(inner["fit_row_ids"])
            inner["validation_row_ids"] = tuple(inner["validation_row_ids"])
            raw["inner"] = InnerSplit(**inner)
            folds.append(FoldManifest(**raw))
        payload["folds"] = tuple(folds)
        for name in ("train_row_ids", "holdout_row_ids"):
            payload[name] = tuple(payload[name])
        manifest = SplitManifest(**payload)
        if not verify_manifest(manifest):
            raise ValueError("Checksum mismatch")
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise CsvValidationError("invalid_manifest", "Stored split is malformed or its checksum does not match.") from None
    return manifest


def training_from_manifest(data: ConfiguredDataset, manifest: SplitManifest) -> TrainingView:
    """Bind saved indices to the exact configured source before profiling/CV.

    Only training features are indexed. Hashes are integrity checks, not
    authentication; these records and paths are trusted private application data.
    """
    if (not verify_manifest(manifest) or data.canonical_sha256 != manifest.dataset_sha256
            or _config_digest(data.config) != manifest.config_sha256
            or len(data.rows) != manifest.row_count or len(data.targets) != manifest.row_count):
        raise CsvValidationError("split_identity_mismatch", "Split does not match the configured dataset.")
    return training_view(data, manifest.train_row_ids)
