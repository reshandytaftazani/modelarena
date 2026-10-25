"""Bounded analytical summaries of an already isolated training view.

No filesystem, full-data argument, category inference or holdout reader exists
here. Declared schema is preserved. Population std (ddof=0), linear quartiles,
five categorical top counts and at most twenty warnings are reported. Missing
cells are excluded from numeric/category counts. No row examples are returned.
These summaries are educational; they must never fit fold preprocessing.
"""

from collections import Counter, defaultdict
import math
import re

import numpy as np

from ml.contracts import (
    DatasetProfile, DuplicateCounts, FeatureProfile, FeatureType, NumericSummary,
    ProfileWarning, TargetProfile, TaskType, TrainingView,
)
from ml.data.validation import CsvValidationError


def _numeric(values: list[float]) -> NumericSummary:
    if not values:
        return NumericSummary(0, None, None, None, None, ())
    array = np.asarray(values, dtype=np.float64)
    # Scale variance, but preserve original magnitudes for order statistics.
    scale = float(np.max(np.abs(array))) or 1.0
    scaled = array / scale
    ordered = sorted(values)
    quantiles = []
    for probability in (0.25, 0.5, 0.75):
        position = (len(values) - 1) * probability
        lower = math.floor(position)
        left, right = ordered[lower], ordered[math.ceil(position)]
        weight = position - lower
        # Same-sign differences cannot overflow; opposite-sign convex sums
        # avoid subtracting extremes. Do not scale away small observations.
        value = (left + (right - left) * weight if (left < 0) == (right < 0)
                 else left * (1 - weight) + right * weight)
        quantiles.append((probability, value))
    try:
        mean = math.fsum(value / len(values) for value in values)
    except OverflowError:
        # Rounding near float max can overflow a sum of divided values.
        mean = float(np.mean(scaled) * scale)
    return NumericSummary(len(values), mean,
                          float(np.std(scaled, ddof=0) * scale),
                          float(np.min(array)), float(np.max(array)),
                          tuple(quantiles))


def _duplicates(data: TrainingView) -> DuplicateCounts:
    feature_counts = Counter(data.rows)
    full_counts = Counter(zip(data.rows, data.targets, strict=True))
    labels = defaultdict(set)
    for row, target in zip(data.rows, data.targets, strict=True):
        labels[row].add(target)
    return DuplicateCounts(
        sum(n > 1 for n in feature_counts.values()),
        sum(n - 1 for n in feature_counts.values()),
        sum(n > 1 for n in full_counts.values()),
        sum(n - 1 for n in full_counts.values()),
        sum(len(group) > 1 for group in labels.values()),
    )


def profile_training(data: TrainingView) -> DatasetProfile:
    """Profile only trusted training rows; Day 4 owns partition provenance."""
    if not isinstance(data, TrainingView):
        raise TypeError("Profiling requires an isolated TrainingView")
    count = len(data.rows)
    features: list[FeatureProfile] = []
    warnings = [ProfileWarning(
        "independence_assumption",
        "Random splitting assumes independent rows. Duplicate acknowledgement cannot "
        "establish independence; grouped and temporal evaluation are unsupported.",
    )]
    for index, column in enumerate(data.config.features.columns):
        values = [row[index] for row in data.rows if row[index] is not None]
        missing = count - len(values)
        distinct = len(set(values))
        numeric = _numeric(values) if column.dtype == FeatureType.NUMERIC else None
        top = () if numeric is not None else tuple(sorted(Counter(values).items(), key=lambda pair: (-pair[1], pair[0]))[:5])
        features.append(FeatureProfile(column.name, column.dtype, missing, missing / count, distinct, numeric, top))
        if not values:
            warnings.append(ProfileWarning("all_missing_feature", "No observed training values; fold-local empty-column policy required.", column.name))
        elif distinct == 1:
            warnings.append(ProfileWarning("constant_feature", "Feature is constant on observed training values.", column.name))
        normalized = re.sub(r"[^a-z0-9]+", "_", column.name.lower()).strip("_")
        tokens = set(normalized.split("_"))
        if "id" in tokens or "uuid" in tokens or (numeric is not None and len(values) >= 2 and distinct / len(values) >= 0.95):
            warnings.append(ProfileWarning("id_like_feature", "Name or near-unique numeric values suggest an identifier; review row independence.", column.name))
        if tokens & {"target", "label", "outcome", "duration"}:
            warnings.append(ProfileWarning("post_outcome_feature", "Name may indicate target or post-outcome information. Review availability at prediction time; exclude duration for Bank pre-call prediction.", column.name))
    if not any(feature.distinct_count for feature in features):
        raise CsvValidationError("no_usable_training_feature", "No observed training feature is available.")
    if data.config.task == TaskType.BINARY_CLASSIFICATION:
        counts = Counter(data.targets)
        class_counts = ((0, counts[0]), (1, counts[1]))
        ratio = min(counts[0], counts[1]) / count
        target = TargetProfile(data.config.task, class_counts, ratio, None)
        if ratio < 0.1:
            warnings.append(ProfileWarning("class_imbalance", "Training minority fraction is below 10%; review metric choice and split feasibility."))
    else:
        target = TargetProfile(data.config.task, (), None, _numeric(list(data.targets)))
    duplicates = _duplicates(data)
    if duplicates.feature_extra_rows:
        warnings.append(ProfileWarning("repeated_observations", "Repeated selected features remain in training, including any differing labels; random splits may overstate generalization."))
    # Global scientific warnings are retained even on wide datasets. Stable order
    # puts feature warnings after global ones without increasing response size.
    warnings.sort(key=lambda warning: warning.feature is not None)
    return DatasetProfile("training", count, len(features), tuple(features), target,
                          duplicates, tuple(warnings[:20]), len(warnings) > 20)
