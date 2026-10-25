"""Task validation must reject invalid data before duplicate acknowledgement."""

from dataclasses import FrozenInstanceError
import pytest

from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, TaskType
from ml.data.csv_io import validate_csv
from ml.data.validation import CsvValidationError, configure_dataset, training_view


def configure(tmp_path, content, *, task=TaskType.BINARY_CLASSIFICATION,
              positive_label="yes", acknowledge=False, columns=None):
    path = tmp_path / "input.csv"
    path.write_text(content, encoding="utf-8")
    schema = FeatureSchema(columns=columns or (FeatureColumn("x", FeatureType.NUMERIC),))
    config = DatasetConfig(task=task, target="y", features=schema,
                           positive_label=positive_label,
                           acknowledge_duplicates=acknowledge)
    return configure_dataset(validate_csv(path, ResourceLimits()), config)


@pytest.mark.parametrize("target,code", [
    ("", "missing_target"), ("NaN", "invalid_target"),
    ("inf", "invalid_target"), ("-Infinity", "invalid_target"),
    ("yes", "binary_target_required"),
])
def test_rejects_invalid_classification_labels_even_with_acknowledgement(tmp_path, target, code):
    with pytest.raises(CsvValidationError) as caught:
        configure(tmp_path, f"x,y\n1,yes\n2,yes\n3,{target}\n", acknowledge=True)
    assert caught.value.code == code


def test_multiclass_target_is_rejected(tmp_path):
    with pytest.raises(CsvValidationError) as caught:
        configure(tmp_path, "x,y\n1,yes\n2,no\n3,maybe\n", acknowledge=True)
    assert caught.value.code == "binary_target_required"


@pytest.mark.parametrize("values,code", [
    ("1,nope", "invalid_target"), ("1,inf", "invalid_target"),
    ("1,NaN", "invalid_target"), ("1,", "missing_target"),
    ("2,2", "constant_target"),
])
def test_regression_requires_finite_variable_numbers(tmp_path, values, code):
    a, b = values.split(",")
    with pytest.raises(CsvValidationError) as caught:
        configure(tmp_path, f"x,y\n1,{a}\n2,{b}\n", task=TaskType.REGRESSION,
                  positive_label=None, acknowledge=True)
    assert caught.value.code == code


@pytest.mark.parametrize("value", ["inf", "-inf", "NaN", "1e999", "text"])
def test_selected_numeric_features_must_be_finite_or_empty(tmp_path, value):
    with pytest.raises(CsvValidationError) as caught:
        configure(tmp_path, f"x,y\n{value},yes\n2,no\n", acknowledge=True)
    assert caught.value.code == "invalid_numeric_feature"


def test_explicit_positive_mapping_preserves_order_and_missing_features(tmp_path):
    data = configure(tmp_path, "x,y\n,yes\n2,no\n3,yes\n")
    assert data.label_mapping == (("no", 0), ("yes", 1))
    view = training_view(data, (2, 0))
    assert view.row_ids == (2, 0)
    assert view.rows == ((3.0,), (None,))
    assert view.targets == (1, 1)
    with pytest.raises(FrozenInstanceError):
        data.config.target = "x"


@pytest.mark.parametrize("acknowledge", [False, True])
def test_differing_labels_on_identical_features_require_acknowledgement(tmp_path, acknowledge):
    if not acknowledge:
        with pytest.raises(CsvValidationError) as caught:
            configure(tmp_path, "x,y\n1,yes\n1,no\n")
        assert caught.value.code == "duplicate_acknowledgement_required"
        assert "yes" not in str(caught.value)
    else:
        data = configure(tmp_path, "x,y\n1,yes\n1,no\n", acknowledge=True)
        assert len(data.rows) == 2
        assert data.targets == (1, 0)


def test_identical_complete_rows_require_acknowledgement_without_deduplication(tmp_path):
    content = "x,y\n1,yes\n1,yes\n2,no\n"
    with pytest.raises(CsvValidationError, match="acknowledgement"):
        configure(tmp_path, content)
    assert len(configure(tmp_path, content, acknowledge=True).rows) == 3


def test_numeric_spelling_changes_cannot_bypass_duplicate_acknowledgement(tmp_path):
    with pytest.raises(CsvValidationError) as caught:
        configure(tmp_path, "x,y\n1,yes\n1.0,no\n")
    assert caught.value.code == "duplicate_acknowledgement_required"


def test_duplicate_acknowledgement_cannot_override_integrity_failure(tmp_path):
    with pytest.raises(CsvValidationError) as caught:
        configure(tmp_path, "x,y\n1,yes\n1,no\n2,inf\n", acknowledge=True)
    assert caught.value.code == "invalid_target"


def test_regression_and_reordered_features_preserve_explicit_schema(tmp_path):
    columns = (FeatureColumn("c", FeatureType.CATEGORICAL), FeatureColumn("x", FeatureType.NUMERIC))
    data = configure(tmp_path, "x,c,y\n1,b,4.5\n2,a,-2\n", columns=columns,
                     task=TaskType.REGRESSION, positive_label=None)
    assert data.rows == (("b", 1.0), ("a", 2.0))
    assert data.targets == (4.5, -2.0)
    assert data.label_mapping == ()


def test_feature_selection_is_explicit_and_excluded_columns_are_not_parsed(tmp_path):
    data = configure(tmp_path, "x,excluded,y\n1,inf,yes\n2,text,no\n")
    assert data.rows == ((1.0,), (2.0,))
    assert data.excluded_columns == ("excluded",)


@pytest.mark.parametrize("columns", [
    (FeatureColumn("y", FeatureType.CATEGORICAL),),
    (FeatureColumn("absent", FeatureType.NUMERIC),),
])
def test_target_or_unknown_feature_rejected(tmp_path, columns):
    with pytest.raises(ValueError):
        configure(tmp_path, "x,y\n1,yes\n2,no\n", columns=columns)


def test_unknown_target_and_positive_label_rejected(tmp_path):
    with pytest.raises(CsvValidationError):
        configure(tmp_path, "x,z\n1,yes\n2,no\n")
    with pytest.raises(CsvValidationError):
        configure(tmp_path, "x,y\n1,yes\n2,no\n", positive_label="other")


def test_categorical_and_boolean_declarations(tmp_path):
    cols = (FeatureColumn("b", FeatureType.BOOLEAN), FeatureColumn("c", FeatureType.CATEGORICAL))
    data = configure(tmp_path, "b,c,y\ntrue,NaN,yes\n0,,no\n", columns=cols)
    assert data.rows == ((True, "NaN"), (False, None))
    with pytest.raises(CsvValidationError):
        configure(tmp_path, "b,c,y\nmaybe,a,yes\n0,b,no\n", columns=cols)


@pytest.mark.parametrize("row_ids", [(), (0, 0), (-1,), (2,), (True,)])
def test_training_view_rejects_ambiguous_or_invalid_indices(tmp_path, row_ids):
    data = configure(tmp_path, "x,y\n1,yes\n2,no\n")
    with pytest.raises(ValueError):
        training_view(data, row_ids)


def test_invalid_configurations_fail_before_data_access():
    feature = FeatureColumn("x", FeatureType.NUMERIC)
    with pytest.raises(ValueError):
        FeatureSchema(columns=())
    with pytest.raises(ValueError):
        FeatureSchema(columns=(feature, feature))
    with pytest.raises(ValueError):
        DatasetConfig(task=TaskType.BINARY_CLASSIFICATION, target="y", features=FeatureSchema((feature,)))
    with pytest.raises(ValueError):
        DatasetConfig(task=TaskType.REGRESSION, target="y", features=FeatureSchema((feature,)), positive_label="yes")
