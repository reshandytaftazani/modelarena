"""Training-only summaries with independent literal expectations."""

import json
from dataclasses import asdict

import pytest

from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, TaskType
from ml.data.csv_io import validate_csv
from ml.data.profile import profile_training
from ml.data.validation import CsvValidationError, configure_dataset, training_view


def load(tmp_path, content, *, task=TaskType.BINARY_CLASSIFICATION, columns=None):
    path = tmp_path / "profile.csv"
    path.write_text(content, encoding="utf-8")
    data = validate_csv(path, ResourceLimits())
    config = DatasetConfig(task, "y", FeatureSchema(columns or (
        FeatureColumn("x", FeatureType.NUMERIC), FeatureColumn("c", FeatureType.CATEGORICAL))),
        positive_label="yes" if task == TaskType.BINARY_CLASSIFICATION else None,
        acknowledge_duplicates=True)
    return data, configure_dataset(data, config)


def test_dimensions_missingness_numeric_and_categorical_statistics(tmp_path):
    _, data = load(tmp_path, "x,c,y\n1,a,yes\n3,a,no\n,b,no\n5,,yes\n999,SECRET,yes\n")
    profile = profile_training(training_view(data, (0, 1, 2, 3)))
    assert profile.partition == "training"
    assert (profile.row_count, profile.feature_count) == (4, 2)
    numeric, category = profile.features
    assert (numeric.dtype, numeric.missing_count, numeric.missing_fraction) == (FeatureType.NUMERIC, 1, 0.25)
    assert numeric.numeric.count == 3
    assert numeric.numeric.mean == pytest.approx(3)
    assert numeric.numeric.std == pytest.approx((8 / 3) ** 0.5)
    assert (numeric.numeric.minimum, numeric.numeric.maximum) == (1, 5)
    assert numeric.numeric.quantiles == ((0.25, 2), (0.5, 3), (0.75, 4))
    assert category.top_counts == (("a", 2), ("b", 1))
    assert category.distinct_count == 2
    assert category.missing_count == 1
    assert profile.target.class_counts == ((0, 2), (1, 2))
    assert profile.target.minority_ratio == 0.5
    assert "SECRET" not in json.dumps(asdict(profile))


def test_holdout_sentinels_cannot_change_any_profile_field(tmp_path):
    prefix = "x,c,y\n1,a,yes\n2,b,no\n3,a,yes\n4,b,no\n"
    _, first = load(tmp_path, prefix + "999999,PRIVATE_A,yes\n1,a,no\n")
    _, second = load(tmp_path, prefix + "-999999,PRIVATE_B,no\n-999999,PRIVATE_B,yes\n")
    assert profile_training(training_view(first, (0, 1, 2, 3))) == profile_training(training_view(second, (0, 1, 2, 3)))


def test_full_dataset_and_structural_records_are_not_profile_inputs(tmp_path):
    raw, data = load(tmp_path, "x,c,y\n1,a,yes\n2,b,no\n")
    for invalid in (raw, data):
        with pytest.raises(TypeError):
            profile_training(invalid)
    assert not hasattr(raw, "target_distribution")
    assert not hasattr(data, "duplicate_counts")


def test_duplicate_details_distinguish_features_full_rows_and_conflicts(tmp_path):
    _, data = load(tmp_path, "x,c,y\n1,a,yes\n1,a,yes\n1,a,no\n2,b,no\n2,b,no\n2,b,no\n")
    counts = profile_training(training_view(data, (0, 1, 2, 3, 4))).duplicates
    assert counts.feature_groups == 2
    assert counts.feature_extra_rows == 3
    assert counts.full_row_groups == 2
    assert counts.full_row_extra_rows == 2
    assert counts.conflicting_label_groups == 1


def test_regression_target_summary_and_all_missing_constant_features(tmp_path):
    _, data = load(tmp_path, "x,c,y\n,a,2\n,a,4\n,a,6\n999,heldout,9000\n", task=TaskType.REGRESSION)
    profile = profile_training(training_view(data, (0, 1, 2)))
    assert profile.target.numeric.mean == 4
    assert profile.target.numeric.minimum == 2
    assert profile.target.numeric.maximum == 6
    assert profile.target.class_counts == ()
    assert profile.features[0].numeric.count == 0
    assert profile.features[0].numeric.mean is None
    assert {w.code for w in profile.warnings} >= {"all_missing_feature", "constant_feature"}


def test_no_usable_training_feature_fails_even_when_holdout_has_values(tmp_path):
    _, data = load(tmp_path, "x,c,y\n,,yes\n,,no\n3,a,yes\n")
    with pytest.raises(CsvValidationError) as caught:
        profile_training(training_view(data, (0, 1)))
    assert caught.value.code == "no_usable_training_feature"


def test_imbalance_id_post_outcome_and_independence_warnings_are_heuristic(tmp_path):
    cols = (FeatureColumn("customer_id", FeatureType.NUMERIC), FeatureColumn("duration", FeatureType.NUMERIC))
    rows = "".join(f"{i},{i + 1},{'yes' if i == 0 else 'no'}\n" for i in range(20))
    _, data = load(tmp_path, "customer_id,duration,y\n" + rows, columns=cols)
    profile = profile_training(training_view(data, tuple(range(20))))
    assert profile.target.minority_ratio == 0.05
    by_code = {w.code: w for w in profile.warnings}
    assert set(by_code) >= {"class_imbalance", "id_like_feature", "post_outcome_feature", "independence_assumption"}
    assert all(w.heuristic for w in by_code.values())
    assert by_code["post_outcome_feature"].feature == "duration"


def test_categorical_top_counts_are_bounded_and_ties_stable(tmp_path):
    content = "x,c,y\n" + "".join(f"{i},cat{i:02},{'yes' if i % 2 else 'no'}\n" for i in range(30))
    _, data = load(tmp_path, content)
    profile = profile_training(training_view(data, tuple(range(30))))
    assert profile.features[1].distinct_count == 30
    assert profile.features[1].top_counts == tuple((f"cat{i:02}", 1) for i in range(5))
    assert len(profile.warnings) <= 20


def test_boolean_profile_and_json_safe_extreme_numeric_statistics(tmp_path):
    cols = (FeatureColumn("x", FeatureType.NUMERIC), FeatureColumn("c", FeatureType.BOOLEAN))
    _, data = load(tmp_path, "x,c,y\n1e308,true,yes\n-1e308,false,no\n", columns=cols)
    profile = profile_training(training_view(data, (0, 1)))
    assert profile.features[0].numeric.mean == 0
    assert profile.features[0].numeric.std == pytest.approx(1e308)
    assert profile.features[1].top_counts == ((False, 1), (True, 1))
    json.dumps(asdict(profile), allow_nan=False)


def test_wide_profile_truncates_feature_warnings_but_retains_global_risks(tmp_path):
    columns = tuple(FeatureColumn(f"id_{i}", FeatureType.NUMERIC) for i in range(30))
    content = ",".join([c.name for c in columns] + ["y"]) + "\n"
    content += ",".join(["1"] * 30 + ["yes"]) + "\n"
    content += ",".join(["1"] * 30 + ["no"]) + "\n"
    _, data = load(tmp_path, content, columns=columns)
    profile = profile_training(training_view(data, (0, 1)))
    assert len(profile.warnings) == 20
    assert profile.warnings_truncated
    assert {w.code for w in profile.warnings} >= {"independence_assumption", "repeated_observations"}


def test_holdout_class_balance_cannot_trigger_training_imbalance_warning(tmp_path):
    _, data = load(tmp_path, "x,c,y\n1,a,yes\n2,b,no\n3,c,no\n4,d,no\n")
    profile = profile_training(training_view(data, (0, 1)))
    assert profile.target.class_counts == ((0, 1), (1, 1))
    assert "class_imbalance" not in {w.code for w in profile.warnings}


def test_mixed_magnitude_quantiles_preserve_small_representable_values(tmp_path):
    _, data = load(tmp_path, "x,c,y\n1e-100,a,yes\n2e-100,b,no\n3e-100,c,yes\n1e308,d,no\n")
    summary = profile_training(training_view(data, (0, 1, 2, 3))).features[0].numeric
    assert summary.quantiles[0][1] == pytest.approx(1.75e-100, rel=1e-12, abs=0)
    assert summary.quantiles[1][1] == pytest.approx(2.5e-100, rel=1e-12, abs=0)


def test_mean_preserves_finite_residual_after_large_value_cancellation(tmp_path):
    _, data = load(tmp_path, "x,c,y\n1e308,a,yes\n-1e308,b,no\n1e-100,c,yes\n")
    summary = profile_training(training_view(data, (0, 1, 2))).features[0].numeric
    assert summary.mean == pytest.approx(3.333333333333333e-101, rel=1e-12, abs=0)
