"""Prepare validated configurations and experiments without fitting models."""

from dataclasses import asdict
import json

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError
from apps.api.services.datasets import load_dataset
from apps.api.services.jobs import PreparedExperiment, prepare_experiment
from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, RunConfig, TaskType, TrainingContext
from ml.data.splits import parse_manifest, training_from_manifest
from ml.data.validation import configure_dataset
from ml.data.profile import profile_training


def parse_config(raw: dict) -> DatasetConfig:
    return DatasetConfig(TaskType(raw["task"]), raw["target"],
                         FeatureSchema(tuple(FeatureColumn(c["name"], FeatureType(c["dtype"]))
                                             for c in raw["features"]["columns"])),
                         raw["positive_label"], raw["acknowledge_duplicates"])


def configure(repository: JobRepository, dataset_id: str, config: DatasetConfig,
              split_seed: int, limits: ResourceLimits) -> dict:
    data = load_dataset(repository, dataset_id, limits)
    prepared = prepare_experiment(repository, data, config, RunConfig(split_seed=split_seed, limits=limits))
    with repository._connection() as db:
        row = db.execute("SELECT configuration_id FROM experiments WHERE experiment_id=?", (prepared.experiment_id,)).fetchone()
    return {"configuration_id": row[0], "profile": asdict(prepared.profile)}


def configuration(repository: JobRepository, configuration_id: str) -> dict:
    with repository._connection() as db:
        row = db.execute("SELECT c.dataset_id,c.config_json,c.profile_json,s.manifest_json, "
                         "EXISTS(SELECT 1 FROM datasets d2 JOIN deletion_intents i ON i.dataset_id=d2.dataset_id "
                         "WHERE d2.canonical_sha256=d.canonical_sha256) AS pending "
                         "FROM configurations c JOIN splits s ON s.split_id=c.split_id "
                         "JOIN datasets d ON d.dataset_id=c.dataset_id WHERE c.configuration_id=?",
                         (configuration_id,)).fetchone()
    if row is None:
        raise NotFoundError("Configuration not found")
    if row["pending"]:
        raise ConflictError("Dataset deletion is pending")
    return {"dataset_id": row["dataset_id"], "config": parse_config(json.loads(row["config_json"])),
            "profile": json.loads(row["profile_json"]), "split_seed": json.loads(row["manifest_json"])["split_seed"]}


def create_experiment(repository: JobRepository, configuration_id: str, primary_metric: str | None,
                      model_seed: int, limits: ResourceLimits) -> str:
    saved = configuration(repository, configuration_id)
    data = load_dataset(repository, saved["dataset_id"], limits)
    run = RunConfig(split_seed=saved["split_seed"], model_seed=model_seed,
                    primary_metric=primary_metric, limits=limits)
    prepared = prepare_experiment(repository, data, saved["config"], run)
    with repository._connection() as db:
        row = db.execute("SELECT configuration_id FROM experiments WHERE experiment_id=?", (prepared.experiment_id,)).fetchone()
    if row[0] != configuration_id:
        raise ConflictError("Experiment does not match frozen configuration")
    return prepared.experiment_id


def load_experiment(repository: JobRepository, experiment_id: str) -> tuple[PreparedExperiment, RunConfig]:
    """Rebuild the trusted training-only context using persisted source and folds."""
    with repository._connection() as db:
        row = db.execute("SELECT e.run_json,c.dataset_id,c.config_json,s.manifest_json "
                         "FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
                         "JOIN splits s ON s.split_id=c.split_id WHERE e.experiment_id=?", (experiment_id,)).fetchone()
    if row is None:
        raise NotFoundError("Experiment not found")
    raw_run = json.loads(row["run_json"])
    raw_run["limits"] = ResourceLimits(**raw_run["limits"])
    run = RunConfig(**raw_run)
    data = load_dataset(repository, row["dataset_id"], run.limits)
    config = parse_config(json.loads(row["config_json"]))
    split = parse_manifest(row["manifest_json"].encode("utf-8"))
    view = training_from_manifest(configure_dataset(data, config), split)
    context = TrainingContext(view, split)
    prepared = PreparedExperiment(experiment_id, context, profile_training(view))
    repository.verify_classical_input(experiment_id, context, run)
    return prepared, run
