"""G1: CSV -> frozen metadata -> training-only profile -> queued Dummy CV.

This invokes CV explicitly as test code, not from an API or queue operation.
The real execution process, result publication and MLflow are later milestones.
"""

from dataclasses import asdict, replace
import json
import sqlite3

import pytest

from apps.api.database.repository import JobRepository, WorkerIdentity
from apps.api.services.jobs import prepare_experiment
from ml.contracts import (
    DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits,
    RunConfig, TaskType, TrainingContext,
)
from ml.data.csv_io import validate_csv
from ml.data.splits import read_manifest, training_from_manifest
from ml.data.validation import CsvValidationError, configure_dataset
from ml.training.cv import evaluate_candidate


def source(tmp_path, task, *, duplicates=False):
    path = tmp_path / "mixed.csv"
    rows = [f"{i if i else ''},group-{i % 3},{'yes' if i % 5 == 0 else 'no'}"
            if task == TaskType.BINARY_CLASSIFICATION else f"{i if i else ''},group-{i % 3},{i * 1.5}"
            for i in range(100)]
    if duplicates:
        rows[1] = rows[0]  # full-row repeat
        rows[2] = rows[0].rsplit(",", 1)[0] + ",no"  # same features, differing label
    path.write_text("x,category,target\n" + "\n".join(rows), encoding="utf-8")
    config = DatasetConfig(task, "target", FeatureSchema((
        FeatureColumn("x", FeatureType.NUMERIC), FeatureColumn("category", FeatureType.CATEGORICAL),
    )), positive_label="yes" if task == TaskType.BINARY_CLASSIFICATION else None)
    return validate_csv(path, ResourceLimits()), config


@pytest.mark.parametrize("task", list(TaskType))
def test_data_to_baseline_roundtrip_uses_persisted_split_and_no_holdout(tmp_path, task):
    data, config = source(tmp_path, task)
    repo = JobRepository(tmp_path / "private" / "metadata.sqlite3")
    prepared = prepare_experiment(repo, data, config, RunConfig())
    job = repo.enqueue_cv(prepared.experiment_id, "cv-once")
    with sqlite3.connect(repo.path) as db:
        saved_split, = db.execute("SELECT manifest_json FROM splits").fetchone()
        saved_profile, = db.execute("SELECT profile_json FROM configurations").fetchone()
        saved_run, = db.execute("SELECT run_json FROM experiments").fetchone()
    split_path = tmp_path / "persisted-split.json"
    split_path.write_text(saved_split, encoding="utf-8")
    split = read_manifest(split_path)
    assert split == prepared.context.split
    assert json.loads(saved_profile) == json.loads(json.dumps(asdict(prepared.profile)))
    raw_run = json.loads(saved_run)
    raw_run["limits"] = ResourceLimits(**raw_run["limits"])
    run = RunConfig(**raw_run)
    configured = configure_dataset(data, config)

    class PoisonHoldout(tuple):
        def __getitem__(self, row_id):
            if row_id in split.holdout_row_ids:
                raise AssertionError("Holdout accessed by data-to-Dummy flow")
            return super().__getitem__(row_id)

    guarded = replace(configured, rows=PoisonHoldout(configured.rows), targets=PoisonHoldout(configured.targets))
    context = TrainingContext(training_from_manifest(guarded, split), split)
    claimed = JobRepository(repo.path).claim_next(WorkerIdentity("test-worker", 1))
    assert claimed.job_id == job.job_id and claimed.state == "running"
    result = evaluate_candidate(context, run)
    assert result.eligible and result.complete and len(result.folds) == 5
    assert prepared.profile.partition == "training" and prepared.profile.row_count == 80
    assert set(context.training.row_ids).isdisjoint(split.holdout_row_ids)
    assert result.summary == evaluate_candidate(context, run).summary
    assert result.split_sha256 == split.manifest_sha256
    assert repo.enqueue_cv(prepared.experiment_id, "cv-once").job_id == claimed.job_id


def test_duplicate_acknowledgement_is_required_before_persistence_and_retained(tmp_path):
    data, config = source(tmp_path, TaskType.BINARY_CLASSIFICATION, duplicates=True)
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    with pytest.raises(CsvValidationError):
        prepare_experiment(repo, data, config, RunConfig())
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM datasets").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM experiments").fetchone()[0] == 0
    prepared = prepare_experiment(repo, data, replace(config, acknowledge_duplicates=True), RunConfig())
    assert any(warning.code == "independence_assumption" for warning in prepared.profile.warnings)
    with sqlite3.connect(repo.path) as db:
        saved, = db.execute("SELECT config_json FROM configurations").fetchone()
        assert json.loads(saved)["acknowledge_duplicates"] is True
    assert evaluate_candidate(prepared.context, RunConfig()).eligible


def test_invalid_run_or_infeasible_data_never_creates_queueable_metadata(tmp_path):
    data, config = source(tmp_path, TaskType.BINARY_CLASSIFICATION)
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    for run in (RunConfig(primary_metric="mae"), RunConfig(family="unknown")):
        with pytest.raises(ValueError):
            prepare_experiment(repo, data, config, run)
    tiny = replace(data, rows=data.rows[:4])
    with pytest.raises(CsvValidationError):
        prepare_experiment(repo, tiny, config, RunConfig())
    with sqlite3.connect(repo.path) as db:
        assert db.execute("SELECT count(*) FROM experiments").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
