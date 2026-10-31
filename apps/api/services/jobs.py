"""Prepare immutable experiments; enqueue never executes model training."""

from dataclasses import dataclass

from apps.api.database.repository import JobRepository
from ml.contracts import DatasetConfig, DatasetProfile, RunConfig, TrainingContext
from ml.data.csv_io import ValidatedDataset
from ml.data.profile import profile_training
from ml.data.splits import create_split, training_from_manifest
from ml.data.validation import configure_dataset
from ml.models.classical import make_classical


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    experiment_id: str
    context: TrainingContext
    profile: DatasetProfile


def prepare_experiment(repository: JobRepository, data: ValidatedDataset,
                       config: DatasetConfig, run: RunConfig) -> PreparedExperiment:
    """Trusted data boundary, not an arbitrary-path or model-execution endpoint.

    Task integrity, duplicate acknowledgement and all-fold/inner feasibility
    must pass before metadata is created. Analytical summaries use training only.
    """
    configured = configure_dataset(data, config)
    split = create_split(data, config, run)
    context = TrainingContext(training_from_manifest(configured, split), split)
    profile = profile_training(context.training)
    make_classical(config.task, run.family, run.model_seed)
    experiment_id = repository.register_experiment(data, config, split, run, profile)
    from apps.api.services.partitions import save_training_partition
    save_training_partition(repository, experiment_id, data, split)
    return PreparedExperiment(experiment_id, context, profile)
