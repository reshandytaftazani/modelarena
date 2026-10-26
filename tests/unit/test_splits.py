"""Immutable partition and fold identities; no analytical or holdout leakage."""

import json
from dataclasses import replace

import pytest

from ml.contracts import (
    DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, FoldManifest, InnerSplit,
    ResourceLimits, RunConfig, SplitManifest, TaskType,
)
from ml.data.csv_io import validate_csv
from ml.data.splits import (
    create_inner_split, create_split, manifest_json, verify_manifest, write_manifest,
)
from ml.data.validation import CsvValidationError, configure_dataset, training_view

def write_dataset(tmp_path, labels, *, name="split.csv", overrides=None):
    overrides = overrides or {}
    lines = ["x,y"] + [f"{overrides.get(i, i + 1)},{label}" for i, label in enumerate(labels)]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return validate_csv(path, ResourceLimits())


def config_for(task, positive_label="yes"):
    return DatasetConfig(
        task=task, target="y",
        features=FeatureSchema((FeatureColumn("x", FeatureType.NUMERIC),)),
        positive_label=positive_label, acknowledge_duplicates=True,
    )


def split_binary(tmp_path, labels, seed=42):
    data = write_dataset(tmp_path, labels)
    config = config_for(TaskType.BINARY_CLASSIFICATION)
    return data, config, create_split(data, config, RunConfig(split_seed=seed))


def balanced_labels(positives, total):
    return ["yes"] * positives + ["no"] * (total - positives)


def test_default_split_is_disjoint_exhaustive_and_stratified(tmp_path):
    data, config, manifest = split_binary(tmp_path, balanced_labels(60, 100))
    train, holdout = set(manifest.train_row_ids), set(manifest.holdout_row_ids)
    assert not train & holdout
    assert train | holdout == set(range(100))
    assert (len(train), len(holdout)) == (80, 20)
    assert manifest.holdout_reason == "default_20_percent"
    targets = configure_dataset(data, config).targets
    assert {targets[i] for i in manifest.train_row_ids} == {0, 1}
    assert {targets[i] for i in manifest.holdout_row_ids} == {0, 1}


def test_manifest_is_byte_identical_for_repeated_creation(tmp_path):
    data, config, first = split_binary(tmp_path, balanced_labels(40, 80))
    second = create_split(data, config, RunConfig(split_seed=42))
    assert first == second
    assert manifest_json(first) == manifest_json(second)
    assert verify_manifest(first) is True
    payload = json.loads(manifest_json(first))
    assert payload["manifest_sha256"] == first.manifest_sha256
    assert list(payload) == sorted(payload)
    assert payload["schema_version"] == 1


def test_manifest_is_persisted_as_canonical_json(tmp_path):
    _, _, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    path = write_manifest(manifest, tmp_path / "splits" / "split.json")
    assert path.read_text(encoding="utf-8") == manifest_json(manifest) + "\n"
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_sha256"] == manifest.manifest_sha256
    write_manifest(manifest, path)
    assert verify_manifest(manifest) is True


def test_manifest_digest_detects_changed_fold_indices(tmp_path):
    _, _, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    fold = manifest.folds[0]
    with pytest.raises(ValueError):
        replace(manifest, folds=(replace(fold, validation_row_ids=fold.validation_row_ids[:-1]),)
                + manifest.folds[1:])
    assert verify_manifest(replace(manifest, split_seed=43)) is False


def test_changed_split_seed_creates_a_different_partition(tmp_path):
    data, config, first = split_binary(tmp_path, balanced_labels(40, 80))
    other = create_split(data, config, RunConfig(split_seed=43))
    assert first.holdout_row_ids != other.holdout_row_ids
    assert first.manifest_sha256 != other.manifest_sha256


def test_partition_hashes_track_partition_content(tmp_path):
    labels = balanced_labels(40, 80)
    data, config, manifest = split_binary(tmp_path, labels)
    train_id = manifest.train_row_ids[0]
    changed = write_dataset(tmp_path, labels, name="changed.csv", overrides={train_id: "999999"})
    other = create_split(changed, config, RunConfig(split_seed=42))
    assert other.train_row_ids == manifest.train_row_ids
    assert other.holdout_row_ids == manifest.holdout_row_ids
    assert other.train_sha256 != manifest.train_sha256
    assert other.holdout_sha256 == manifest.holdout_sha256
    assert other.dataset_sha256 != manifest.dataset_sha256


def test_reordered_rows_are_a_different_dataset_identity(tmp_path):
    labels = balanced_labels(40, 80)
    data, config, manifest = split_binary(tmp_path, labels)
    reordered = write_dataset(tmp_path, list(reversed(labels)), name="reordered.csv")
    other = create_split(reordered, config, RunConfig(split_seed=42))
    assert other.dataset_sha256 != manifest.dataset_sha256
    assert other.manifest_sha256 != manifest.manifest_sha256


def test_row_ids_are_persisted_in_ascending_order(tmp_path):
    data, config, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    assert manifest.train_row_ids == tuple(sorted(manifest.train_row_ids))
    assert manifest.holdout_row_ids == tuple(sorted(manifest.holdout_row_ids))
    for fold in manifest.folds:
        assert fold.fit_row_ids == tuple(sorted(fold.fit_row_ids))
        assert fold.validation_row_ids == tuple(sorted(fold.validation_row_ids))
        assert fold.inner.fit_row_ids == tuple(sorted(fold.inner.fit_row_ids))
        assert fold.inner.validation_row_ids == tuple(sorted(fold.inner.validation_row_ids))
    view = training_view(configure_dataset(data, config), manifest.train_row_ids)
    assert view.row_ids == manifest.train_row_ids


@pytest.mark.parametrize("labels", [["yes", "yes", "yes", "no"], ["yes", "yes", "no", "no"]])
def test_unsupported_counts_are_rejected_without_exposing_labels(tmp_path, labels):
    with pytest.raises(CsvValidationError) as caught:
        split_binary(tmp_path, labels)
    assert caught.value.code == "split_infeasible"
    assert "yes" not in str(caught.value)


def test_label_absent_from_a_partition_is_rejected(tmp_path):
    with pytest.raises(CsvValidationError) as caught:
        split_binary(tmp_path, ["yes"] + ["no"] * 39)
    assert caught.value.code == "split_infeasible"


def test_five_folds_are_preferred_when_feasible(tmp_path):
    data, config, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    assert (manifest.fold_count, manifest.fold_reason) == (5, "preferred_five_folds")
    assert manifest.algorithm == "stratified_kfold"
    train = set(manifest.train_row_ids)
    seen = set()
    for index, fold in enumerate(manifest.folds):
        assert fold.fold_index == index
        assert set(fold.fit_row_ids) | set(fold.validation_row_ids) == train
        assert not set(fold.fit_row_ids) & set(fold.validation_row_ids)
        assert not set(fold.validation_row_ids) & set(manifest.holdout_row_ids)
        assert len(fold.validation_row_ids) in (12, 13)
        seen |= set(fold.validation_row_ids)
    assert seen == train
    targets = configure_dataset(data, config).targets
    for fold in manifest.folds:
        assert {targets[i] for i in fold.validation_row_ids} == {0, 1}
        assert {targets[i] for i in fold.fit_row_ids} == {0, 1}


@pytest.mark.parametrize("positives,total,expected", [
    (8, 40, 5), (12, 60, 5), (40, 80, 5), (6, 54, 5), (5, 45, 4),
    (4, 40, 3),
])
def test_fold_count_is_the_largest_feasible_number(tmp_path, positives, total, expected):
    data, config, manifest = split_binary(tmp_path, balanced_labels(positives, total))
    targets = configure_dataset(data, config).targets
    assert manifest.fold_count == expected
    assert len(manifest.folds) == expected
    if expected == 5:
        assert manifest.fold_reason == "preferred_five_folds"
    else:
        assert manifest.fold_reason == f"reduced_from_five_folds:{expected}"
        assert manifest.fold_warning
    for fold in manifest.folds:
        assert {targets[i] for i in fold.validation_row_ids} == {0, 1}
        assert {targets[i] for i in fold.inner.fit_row_ids} == {0, 1}
        assert {targets[i] for i in fold.inner.validation_row_ids} == {0, 1}


def test_reduced_fold_count_keeps_every_partition_usable(tmp_path):
    data, config, manifest = split_binary(tmp_path, balanced_labels(5, 45))
    targets = configure_dataset(data, config).targets
    assert manifest.fold_count == 4
    for fold in manifest.folds:
        assert {targets[i] for i in fold.validation_row_ids} == {0, 1}
        assert {targets[i] for i in fold.inner.validation_row_ids} == {0, 1}
        assert {targets[i] for i in fold.inner.fit_row_ids} == {0, 1}


def test_inner_splits_stay_inside_each_outer_training_partition(tmp_path):
    data, config, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    targets = configure_dataset(data, config).targets
    for fold in manifest.folds:
        fit = set(fold.fit_row_ids)
        inner = fold.inner
        assert set(inner.fit_row_ids) | set(inner.validation_row_ids) == fit
        assert not set(inner.fit_row_ids) & set(inner.validation_row_ids)
        assert not set(inner.validation_row_ids) & set(fold.validation_row_ids)
        assert not set(inner.validation_row_ids) & set(manifest.holdout_row_ids)
        assert len(inner.validation_row_ids) >= 2
        assert {targets[i] for i in inner.fit_row_ids} == {0, 1}
        assert {targets[i] for i in inner.validation_row_ids} == {0, 1}
        assert inner.seed != manifest.split_seed


def test_inner_validation_grows_past_twenty_percent_when_rounding_is_too_small():
    targets = (1, 1, 1, 0, 0)
    inner = create_inner_split((0, 1, 2, 3, 4), targets, TaskType.BINARY_CLASSIFICATION, 143)
    assert inner.expanded is True
    assert len(inner.validation_row_ids) == 2
    assert set(inner.fit_row_ids) | set(inner.validation_row_ids) == {0, 1, 2, 3, 4}
    assert not set(inner.fit_row_ids) & set(inner.validation_row_ids)


def test_inner_split_keeps_twenty_percent_when_rounding_allows_it():
    targets = (1,) * 20 + (0,) * 20
    inner = create_inner_split(tuple(range(40)), targets, TaskType.BINARY_CLASSIFICATION, 143)
    assert inner.expanded is False
    assert len(inner.validation_row_ids) == 8
    assert len(inner.fit_row_ids) == 32


def test_inner_split_rejects_a_partition_without_both_labels():
    with pytest.raises(CsvValidationError) as caught:
        create_inner_split((0, 1, 2, 3), (1, 1, 1, 0), TaskType.BINARY_CLASSIFICATION, 143)
    assert caught.value.code == "inner_split_infeasible"


def test_regression_folds_keep_two_validation_rows(tmp_path):
    labels = [f"{(i * 7) % 23 + 1}.5" for i in range(60)]
    data = write_dataset(tmp_path, labels)
    config = config_for(TaskType.REGRESSION, positive_label=None)
    manifest = create_split(data, config, RunConfig(split_seed=42))
    assert manifest.algorithm == "kfold"
    assert len(manifest.holdout_row_ids) == 12
    assert manifest.fold_count == 5
    for fold in manifest.folds:
        assert len(fold.validation_row_ids) >= 2
        assert len(fold.inner.validation_row_ids) >= 2
        assert len(fold.inner.fit_row_ids) >= 2
        assert not set(fold.inner.validation_row_ids) & set(fold.validation_row_ids)


def test_regression_rejects_a_dataset_without_usable_holdout(tmp_path):
    labels = [f"{i}.5" for i in range(5)]
    data = write_dataset(tmp_path, labels)
    config = config_for(TaskType.REGRESSION, positive_label=None)
    with pytest.raises(CsvValidationError) as caught:
        create_split(data, config, RunConfig(split_seed=42))
    assert caught.value.code == "split_infeasible"


def test_duplicate_rows_still_require_acknowledgement_before_splitting(tmp_path):
    data = write_dataset(tmp_path, balanced_labels(4, 8), overrides={1: "1"})
    config = DatasetConfig(
        task=TaskType.BINARY_CLASSIFICATION, target="y",
        features=FeatureSchema((FeatureColumn("x", FeatureType.NUMERIC),)),
        positive_label="yes",
    )
    with pytest.raises(CsvValidationError) as caught:
        create_split(data, config, RunConfig())
    assert caught.value.code == "duplicate_acknowledgement_required"


def valid_manifest():
    first = InnerSplit((0, 1), (2, 3), False, 143)
    second = InnerSplit((4, 5), (6, 7), False, 144)
    return SplitManifest(
        schema_version=1, split_seed=42, algorithm="stratified_kfold",
        dataset_sha256="a" * 64, config_sha256="b" * 64, row_count=10,
        train_row_ids=(0, 1, 2, 3, 4, 5, 6, 7), holdout_row_ids=(8, 9), train_sha256="c" * 64,
        holdout_sha256="d" * 64, fold_count=2, fold_reason="reduced_from_five_folds:2",
        fold_warning="five folds infeasible; largest feasible count 2 recorded",
        folds=(FoldManifest(0, (0, 1, 2, 3), (4, 5, 6, 7), first),
               FoldManifest(1, (4, 5, 6, 7), (0, 1, 2, 3), second)),
        holdout_reason="default_20_percent", manifest_sha256="e" * 64,
    )


@pytest.mark.parametrize("change", [
    {"holdout_row_ids": (5, 4, 3, 2, 1, 0)},
    {"holdout_row_ids": (4, 5)},
    {"train_row_ids": (0, 1, 2, 3)},
    {"fold_count": 3},
    {"holdout_row_ids": (6, 8), "row_count": 9},
    {"algorithm": "unsupported"},
    {"manifest_sha256": "not-a-digest"},
    {"fold_reason": "preferred_five_folds"},
])
def test_manifest_contract_rejects_inconsistent_partitions(change):
    with pytest.raises(ValueError):
        replace(valid_manifest(), **change)


def test_manifest_contract_rejects_inconsistent_folds():
    inner = InnerSplit((0, 1), (2, 3), False, 143)
    with pytest.raises(ValueError):
        FoldManifest(0, (0, 1, 2, 3), (3, 4), inner)
    with pytest.raises(ValueError):
        FoldManifest(0, (0, 1, 2, 3, 4), (5,), inner)
    with pytest.raises(ValueError):
        FoldManifest(0, (0, 1, 2, 3), (2,), inner)
    with pytest.raises(ValueError):
        InnerSplit((0, 1), (1, 2), False, 143)
    with pytest.raises(ValueError):
        InnerSplit((0,), (2, 3), False, 143)


def test_positive_label_change_cannot_change_locked_row_assignments(tmp_path):
    data, config, first = split_binary(tmp_path, balanced_labels(40, 80))
    second = create_split(data, replace(config, positive_label="no"), RunConfig())
    assert first.train_row_ids == second.train_row_ids
    assert first.holdout_row_ids == second.holdout_row_ids
    assert first.folds == second.folds


def test_manifest_rejects_repeated_validation_fold_with_uncovered_rows():
    manifest = valid_manifest()
    duplicate = replace(manifest.folds[0], fold_index=1)
    with pytest.raises(ValueError):
        replace(manifest, folds=(manifest.folds[0], duplicate))


def test_split_writer_never_replaces_different_frozen_manifest(tmp_path):
    data, config, first = split_binary(tmp_path, balanced_labels(40, 80))
    second = create_split(data, config, RunConfig(split_seed=43))
    path = write_manifest(first, tmp_path / "frozen.json")
    before = path.read_bytes()
    with pytest.raises(CsvValidationError):
        write_manifest(second, path)
    assert path.read_bytes() == before


def test_tampered_manifest_cannot_be_persisted(tmp_path):
    _, _, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    path = tmp_path / "tampered.json"
    with pytest.raises(CsvValidationError):
        write_manifest(replace(manifest, split_seed=43), path)
    assert not path.exists()


@pytest.mark.parametrize("rows,folds", [(10, 4), (8, 3)])
def test_regression_adapts_fold_count_with_inner_minimums(tmp_path, rows, folds):
    data = write_dataset(tmp_path, [str(i) for i in range(rows)])
    manifest = create_split(data, config_for(TaskType.REGRESSION, None), RunConfig())
    assert manifest.fold_count == folds
    for fold in manifest.folds:
        assert len(fold.validation_row_ids) >= 2
        assert len(fold.inner.fit_row_ids) >= 2
        assert len(fold.inner.validation_row_ids) >= 2


def test_two_outer_folds_are_supported_when_both_inner_partitions_fit():
    from ml.data.splits import _feasible_folds

    folds = _feasible_folds(tuple(range(8)), list(range(8)), TaskType.REGRESSION, 2, 42)
    assert len(folds) == 2
    assert all(len(fold.inner.fit_row_ids) == len(fold.inner.validation_row_ids) == 2 for fold in folds)


def test_no_feasible_outer_inner_scheme_rejects_before_experiment(tmp_path):
    # Three positives leave two in training; CV necessarily leaves at most one
    # in an outer fit, so an inner two-label split is impossible.
    with pytest.raises(CsvValidationError) as caught:
        split_binary(tmp_path, balanced_labels(3, 40))
    assert caught.value.code == "split_infeasible"


def test_fractional_holdout_rounds_up(tmp_path):
    _, _, manifest = split_binary(tmp_path, balanced_labels(20, 42))
    assert len(manifest.holdout_row_ids) == 9


def test_manifest_contract_rejects_more_than_five_folds():
    manifest = valid_manifest()
    with pytest.raises(ValueError):
        replace(manifest, fold_count=6, folds=tuple(replace(manifest.folds[0], fold_index=i) for i in range(6)),
                fold_reason="reduced_from_five_folds:6")


def test_imbalanced_inner_search_is_bounded_and_keeps_both_labels(monkeypatch):
    from ml.data import splits

    original = splits._inner_labels
    attempts = 0

    def counted(*args):
        nonlocal attempts
        attempts += 1
        assert attempts <= 20, "Inner split must not scan every candidate size"
        return original(*args)

    monkeypatch.setattr(splits, "_inner_labels", counted)
    targets = (1, 1) + (0,) * 9998
    inner = create_inner_split(tuple(range(10000)), targets, TaskType.BINARY_CLASSIFICATION, 143)
    assert {targets[i] for i in inner.fit_row_ids} == {0, 1}
    assert {targets[i] for i in inner.validation_row_ids} == {0, 1}
    assert inner.expanded
    assert 2500 <= len(inner.validation_row_ids) <= 2501


def test_persisted_manifest_round_trips_with_splitter_version(tmp_path):
    import sklearn
    from ml.data import splits

    _, _, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    path = write_manifest(manifest, tmp_path / "saved.json")
    assert splits.read_manifest(path) == manifest
    assert json.loads(path.read_bytes())["splitter_version"] == sklearn.__version__


@pytest.mark.parametrize("content", [b"not-json", b"[]", b'{"schema_version":1,"schema_version":2}'])
def test_manifest_reader_rejects_malformed_records(tmp_path, content):
    from ml.data import splits

    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(CsvValidationError):
        splits.read_manifest(path)


def test_manifest_reader_rejects_modified_checksum(tmp_path):
    from ml.data import splits

    _, _, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    payload = json.loads(manifest_json(manifest))
    payload["split_seed"] = 123
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CsvValidationError):
        splits.read_manifest(path)


def test_profile_uses_persisted_training_rows_without_reading_holdout(tmp_path):
    from ml.data import splits
    from ml.data.profile import profile_training

    data, config, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    configured = configure_dataset(data, config)

    class GuardedRows(tuple):
        def __getitem__(self, index):
            if index in manifest.holdout_row_ids:
                raise AssertionError("Holdout accessed")
            return super().__getitem__(index)

    guarded = replace(configured, rows=GuardedRows(configured.rows))
    path = write_manifest(manifest, tmp_path / "saved.json")
    view = splits.training_from_manifest(guarded, splits.read_manifest(path))
    profile = profile_training(view)
    assert view.row_ids == manifest.train_row_ids
    assert profile.row_count == 64
    assert profile.features[0].numeric.mean == pytest.approx(sum(i + 1 for i in manifest.train_row_ids) / 64)


def test_saved_indices_cannot_be_applied_to_reordered_data_or_changed_config(tmp_path):
    from ml.data import splits

    data, config, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    configured = configure_dataset(data, config)
    with pytest.raises(CsvValidationError):
        splits.training_from_manifest(replace(configured, canonical_sha256="f" * 64), manifest)
    with pytest.raises(CsvValidationError):
        splits.training_from_manifest(replace(configured, config=replace(config, positive_label="no")), manifest)


def test_publication_failure_leaves_no_partial_final_or_temporary_file(tmp_path, monkeypatch):
    from ml.data import splits

    _, _, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    destination = tmp_path / "private" / "frozen.json"

    def fail_link(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(splits.os, "link", fail_link)
    with pytest.raises(OSError):
        write_manifest(manifest, destination)
    assert list(destination.parent.iterdir()) == []


def test_corrupt_row_count_is_rejected_without_allocating_declared_range(tmp_path, monkeypatch):
    import ml.contracts as contracts
    from ml.data import splits

    _, _, manifest = split_binary(tmp_path, balanced_labels(40, 80))
    payload = json.loads(manifest_json(manifest))
    payload["row_count"] = 10**12
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    builtin_range = range

    def guarded_range(stop):
        assert stop < 100_000, "Untrusted count must not drive allocation"
        return builtin_range(stop)

    monkeypatch.setattr(contracts, "range", guarded_range, raising=False)
    with pytest.raises(CsvValidationError):
        splits.read_manifest(path)
