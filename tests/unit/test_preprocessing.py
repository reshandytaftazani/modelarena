"""Regression checks for fitting outside the intended fold and unstable schemas."""

import numpy as np
import pytest

from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, TaskType, TrainingView
from ml.preprocessing.factory import FoldPreprocessor, make_preprocessor


def view(rows, types=(FeatureType.NUMERIC, FeatureType.CATEGORICAL), start=0):
    config = DatasetConfig(
        TaskType.REGRESSION, "target",
        FeatureSchema(tuple(FeatureColumn(f"x{i}", kind) for i, kind in enumerate(types))),
    )
    return TrainingView(config, tuple(range(start, start + len(rows))), tuple(rows),
                        tuple(float(i) for i in range(len(rows))), ())


def test_validation_values_never_change_fit_statistics_or_vocabulary():
    train = view(((1., "a"), (3., "b"), (None, None)))
    validation = view(((1000000., "VALIDATION_ONLY"), (None, "a")), start=10)
    processor = FoldPreprocessor()
    encoded = processor.fit_transform(train)
    numeric = processor.transformer_.named_transformers_["numeric"]
    categorical = processor.transformer_.named_transformers_["categorical"]
    assert numeric.named_steps["imputer"].statistics_.tolist() == [2.]
    assert numeric.named_steps["scaler"].mean_.tolist() == [2.]
    vocabulary = categorical.named_steps["encoder"].categories_[0].tolist()
    assert "value:VALIDATION_ONLY" not in vocabulary
    assert encoded.shape == (3, 4)  # numeric + a, b, missing
    output = processor.transform(validation)
    np.testing.assert_allclose(output[:, 0], [(1000000. - 2.) / np.sqrt(2. / 3.), 0.])
    assert output[0, 1:].tolist() == [0., 0., 0.]
    assert numeric.named_steps["scaler"].mean_.tolist() == [2.]
    assert np.isfinite(output).all()


def test_all_missing_and_constant_columns_keep_finite_stable_width():
    train = view(((None, None, 7.), (None, None, 7.)),
                 (FeatureType.NUMERIC, FeatureType.CATEGORICAL, FeatureType.NUMERIC))
    validation = view(((9., "new", 7.),),
                      (FeatureType.NUMERIC, FeatureType.CATEGORICAL, FeatureType.NUMERIC), start=9)
    processor = FoldPreprocessor()
    output = processor.fit_transform(train)
    assert output.shape == (2, 3)
    assert np.isfinite(output).all()
    assert processor.transform(validation).shape == (1, 3)
    np.testing.assert_array_equal(output[:, :2], np.zeros((2, 2)))


@pytest.mark.parametrize("types,rows", [
    ((FeatureType.NUMERIC,), ((1.,), (3.,))),
    ((FeatureType.CATEGORICAL,), (("a",), (None,))),
    ((FeatureType.BOOLEAN,), ((True,), (False,), (None,))),
])
def test_single_type_and_boolean_features(types, rows):
    output = FoldPreprocessor().fit_transform(view(rows, types))
    assert np.isfinite(output).all()
    assert output.shape[0] == len(rows)


def test_missing_sentinel_cannot_collide_with_user_category():
    output = FoldPreprocessor().fit_transform(view(((None,), ("__MISSING__",)), (FeatureType.CATEGORICAL,)))
    assert output.shape == (2, 2)
    assert not np.array_equal(output[0], output[1])


def test_factory_and_fold_instances_are_fresh_and_do_not_encode_targets():
    train = view(((1., "a"), (3., "b")))
    first, second = FoldPreprocessor(), FoldPreprocessor()
    one = first.fit_transform(train)
    changed_targets = TrainingView(train.config, train.row_ids, train.rows, (999., -999.), ())
    np.testing.assert_array_equal(one, second.fit_transform(changed_targets))
    assert first.transformer_ is not second.transformer_
    assert make_preprocessor(train.config.features) is not make_preprocessor(train.config.features)


def test_transform_requires_fit_and_matching_configuration():
    train = view(((1., "a"), (3., "b")))
    with pytest.raises(ValueError, match="fitted"):
        FoldPreprocessor().transform(train)
    processor = FoldPreprocessor().fit(train)
    with pytest.raises(ValueError, match="schema|configuration"):
        processor.transform(view(((1.,),), (FeatureType.NUMERIC,)))


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), "wrong"])
def test_invalid_numeric_cells_fail_safely(bad):
    with pytest.raises(ValueError, match="numeric|finite"):
        FoldPreprocessor().fit_transform(view(((bad, "a"), (1., "b"))))
