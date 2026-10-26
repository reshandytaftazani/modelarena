"""Resource rejection must occur before encoding or dense allocation."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from sklearn.compose import ColumnTransformer

from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, TaskType, TrainingView
from ml.preprocessing.factory import FoldPreprocessor, estimate_resources, guard_allocation


def data(rows, types):
    config = DatasetConfig(TaskType.REGRESSION, "y", FeatureSchema(tuple(
        FeatureColumn(f"f{i}", kind) for i, kind in enumerate(types))))
    return TrainingView(config, tuple(range(len(rows))), tuple(rows), tuple(range(len(rows))), ())


def test_training_only_width_counts_missing_and_boolean_categories():
    train = data(((1., "a", True), (None, None, False), (3., "a", None)),
                 (FeatureType.NUMERIC, FeatureType.CATEGORICAL, FeatureType.BOOLEAN))
    estimate = estimate_resources(train)
    assert estimate.encoded_features == 6
    assert estimate.category_counts == (("f1", 2), ("f2", 3))


def test_budget_includes_fit_and_transform_float64_copies_and_float32_cast():
    # (10 fit + 5 validation) * 2 features * (8 + 8 + 4 bytes) * 3 = 1800
    assert guard_allocation(10, 5, 2, ResourceLimits(preprocessing_bytes=1800)) == 1800
    with pytest.raises(ValueError, match="memory"):
        guard_allocation(10, 5, 2, ResourceLimits(preprocessing_bytes=1799))


@pytest.mark.parametrize("rows,types,limits,reason", [
    (tuple((str(i),) for i in range(101)), (FeatureType.CATEGORICAL,), ResourceLimits(), "categor"),
    (((1., "a"), (2., "b")), (FeatureType.NUMERIC, FeatureType.CATEGORICAL), ResourceLimits(max_encoded_features=2), "width"),
    (((1.,), (2.,)), (FeatureType.NUMERIC,), ResourceLimits(preprocessing_bytes=100), "memory"),
])
def test_estimated_rejection_precedes_transformer_fit(monkeypatch, rows, types, limits, reason):
    def forbidden(*args, **kwargs):
        pytest.fail("encoder fit ran before preallocation rejection")
    monkeypatch.setattr(ColumnTransformer, "fit", forbidden)
    with pytest.raises(ValueError, match=reason):
        FoldPreprocessor(limits).fit_transform(data(rows, types))


def test_no_observed_training_feature_fails_before_fit():
    with pytest.raises(ValueError, match="usable"):
        estimate_resources(data(((None, None), (None, None)), (FeatureType.NUMERIC, FeatureType.CATEGORICAL)))


def test_realized_width_is_rechecked_before_dense_conversion(monkeypatch):
    train = data((("a",), ("b",)), (FeatureType.CATEGORICAL,))
    processor = FoldPreprocessor(ResourceLimits(max_encoded_features=2)).fit(train)
    monkeypatch.setattr(processor.transformer_, "transform", lambda frame: sparse.csr_matrix((2, 3)))
    monkeypatch.setattr(sparse.csr_matrix, "toarray", lambda *a, **k: pytest.fail("unsafe dense conversion"))
    with pytest.raises(ValueError, match="width"):
        processor.transform(train)


def test_transform_budget_checked_before_encoder_allocation(monkeypatch):
    train = data(((1., "a"), (2., "b")), (FeatureType.NUMERIC, FeatureType.CATEGORICAL))
    processor = FoldPreprocessor(ResourceLimits(preprocessing_bytes=1000)).fit(train)
    def forbidden(*args, **kwargs):
        pytest.fail("transform allocated before budget check")
    monkeypatch.setattr(processor.transformer_, "transform", forbidden)
    validation = replace(train, row_ids=tuple(range(20)), rows=train.rows * 10, targets=train.targets * 10)
    with pytest.raises(ValueError, match="memory"):
        processor.transform(validation)


def test_unknown_categories_do_not_expand_fit_width_and_float32_is_finite():
    train = data((("a",), ("b",)), (FeatureType.CATEGORICAL,))
    processor = FoldPreprocessor().fit(train)
    validation = replace(train, rows=(("c",), ("d",)))
    result = processor.transform(validation, dtype=np.float32)
    assert result.shape == (2, 2)
    assert result.dtype == np.float32
    assert not result.any()


def test_float32_overflow_is_rejected():
    train = data(((1.,), (1.,)), (FeatureType.NUMERIC,))
    processor = FoldPreprocessor().fit(train)
    with pytest.raises(ValueError, match="finite|float32"):
        processor.transform(replace(train, rows=((1.e100,), (1.,))), dtype=np.float32)
