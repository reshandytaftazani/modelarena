"""Structural errors and explicit task validation without analytical summaries."""

import math
from typing import TYPE_CHECKING

from ml.contracts import (
    ConfiguredDataset, DatasetConfig, FeatureType, FeatureValue, TaskType, TrainingView,
)

if TYPE_CHECKING:
    from ml.data.csv_io import ValidatedDataset


class CsvValidationError(ValueError):
    """A stable error code and safe message, without paths or record values."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _number(value: str, code: str) -> float:
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise CsvValidationError(code, "A declared numeric value is invalid.") from None
    if not math.isfinite(number):
        raise CsvValidationError(code, "Numeric values must be finite.")
    return number


def configure_dataset(data: "ValidatedDataset", config: DatasetConfig) -> ConfiguredDataset:
    """Check all integrity conditions before the acknowledgement gate.

    Empty/whitespace-only cells are missing. Other categorical spellings remain
    literal; numeric NaN/inf are invalid. Duplicate equality uses selected,
    typed features and target, so alternate numeric spellings compare equally.
    No counts, distributions, examples or inferred types are exposed here.
    """
    if not isinstance(config, DatasetConfig):
        raise ValueError("DatasetConfig required")
    names = {c.name for c in config.features.columns}
    if config.target not in data.columns or not names.issubset(data.columns):
        raise CsvValidationError("unknown_column", "Selected columns do not exist.")
    target_index = data.columns.index(config.target)
    indices = tuple(data.columns.index(c.name) for c in config.features.columns)
    labels: list[str] = []
    targets: list[int | float] = []
    rows: list[tuple[FeatureValue, ...]] = []
    for raw in data.rows:
        label = raw[target_index]
        if not label.strip():
            raise CsvValidationError("missing_target", "Target labels cannot be missing.")
        if config.task == TaskType.REGRESSION:
            targets.append(_number(label, "invalid_target"))
        else:
            # Numeric nonfinite spellings are forbidden even for string labels.
            try:
                nonfinite = not math.isfinite(float(label))
            except (ValueError, OverflowError):
                nonfinite = False
            if nonfinite:
                raise CsvValidationError("invalid_target", "Target labels must be finite.")
            labels.append(label)
        values: list[FeatureValue] = []
        for col, index in zip(config.features.columns, indices, strict=True):
            value = raw[index]
            if not value.strip():
                values.append(None)
            elif col.dtype == FeatureType.NUMERIC:
                values.append(_number(value, "invalid_numeric_feature"))
            elif col.dtype == FeatureType.BOOLEAN:
                boolean = {"true": True, "false": False, "1": True, "0": False}.get(value.strip().lower())
                if boolean is None:
                    raise CsvValidationError("invalid_boolean_feature", "Boolean values must be true/false or 1/0.")
                values.append(boolean)
            else:
                values.append(value)
        rows.append(tuple(values))
    mapping: tuple[tuple[str, int], ...] = ()
    if config.task == TaskType.BINARY_CLASSIFICATION:
        distinct = set(labels)
        if len(distinct) != 2:
            raise CsvValidationError("binary_target_required", "Exactly two target labels required.")
        if config.positive_label not in distinct:
            raise CsvValidationError("invalid_positive_label", "Positive label is not present.")
        negative = next(label for label in distinct if label != config.positive_label)
        mapping = ((negative, 0), (config.positive_label, 1))
        targets = [int(label == config.positive_label) for label in labels]
    elif len(set(targets)) < 2:
        raise CsvValidationError("constant_target", "Regression target must vary.")
    if len(set(rows)) < len(rows) and not config.acknowledge_duplicates:
        raise CsvValidationError(
            "duplicate_acknowledgement_required",
            "Repeated selected features require explicit risk acknowledgement; rows are retained. "
            "Random splitting assumes independent observations; grouped/time data are unsupported.",
        )
    return ConfiguredDataset(config, tuple(rows), tuple(targets), mapping,
                             tuple(c for c in data.columns if c not in names and c != config.target),
                             data.canonical_sha256)


def training_view(data: ConfiguredDataset, row_ids: tuple[int, ...]) -> TrainingView:
    """Materialize trusted training indices without retaining the full dataset."""
    if not isinstance(row_ids, tuple) or not row_ids:
        raise ValueError("Nonempty immutable training indices required")
    if any(type(i) is not int or not 0 <= i < len(data.rows) for i in row_ids):
        raise ValueError("Invalid training row identity")
    return TrainingView(data.config, row_ids, tuple(data.rows[i] for i in row_ids),
                        tuple(data.targets[i] for i in row_ids), data.label_mapping)
