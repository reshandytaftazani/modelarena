from dataclasses import FrozenInstanceError, asdict, replace
import pytest

from apps.api.settings import Settings, load_settings
from ml.contracts import ResourceLimits, RunConfig


def test_initial_policy_is_frozen_and_serializable():
    limits = ResourceLimits()
    assert limits.policy_version == 1
    assert limits.compute_threads == 4
    assert limits.child_rss_bytes == 4 * 1024**3
    assert (limits.torch_interop_threads, limits.estimator_n_jobs,
            limits.dataloader_workers, limits.active_jobs, limits.cv_parallelism) == (1, 1, 0, 1, 1)
    assert (limits.upload_bytes, limits.max_rows, limits.max_columns,
            limits.max_cell_bytes) == (20_000_000, 50_000, 200, 65_536)
    assert (limits.max_encoded_features, limits.max_categories,
            limits.preprocessing_bytes) == (2048, 100, 512 * 1024**2)
    assert (limits.classical_fold_seconds, limits.mlp_fold_seconds,
            limits.cv_job_seconds, limits.finalization_seconds) == (180, 300, 2700, 600)
    assert limits.sample_interval_seconds == 0.5
    assert limits.system_reserve_bytes == 4 * 1024**3
    assert limits.memory_pressure_seconds == 10
    assert asdict(limits)["compute_threads"] == 4
    with pytest.raises(FrozenInstanceError):
        limits.compute_threads = 6


@pytest.mark.parametrize("threads", [2, 4, 6])
def test_benchmark_thread_candidates(threads):
    assert ResourceLimits(compute_threads=threads).compute_threads == threads


@pytest.mark.parametrize("changes", [
    {"compute_threads": 3}, {"compute_threads": True},
    {"compute_threads": 4.0}, {"policy_version": 2},
    {"child_rss_bytes": 4 * 1024**3 + 1}, {"child_rss_bytes": 0},
    {"upload_bytes": 20_000_001}, {"max_rows": -1},
    {"max_columns": 201}, {"max_cell_bytes": 65537},
    {"preprocessing_bytes": 512 * 1024**2 + 1},
    {"max_encoded_features": 2049}, {"max_categories": 101},
    {"estimator_n_jobs": -1}, {"torch_interop_threads": 2},
    {"dataloader_workers": 1}, {"active_jobs": 2}, {"cv_parallelism": 2},
    {"classical_fold_seconds": 181}, {"mlp_fold_seconds": 301},
    {"cv_job_seconds": 2701}, {"finalization_seconds": 601},
    {"sample_interval_seconds": 1}, {"sample_interval_seconds": float("nan")},
    {"system_reserve_bytes": 1}, {"memory_pressure_seconds": 11},
    {"queue_size": 11}, {"storage_bytes": 5 * 1024**3 + 1},
    {"min_free_disk_bytes": 0}, {"max_diagnostic_points": 2001},
])
def test_invalid_or_relaxed_policy_is_rejected(changes):
    with pytest.raises(ValueError):
        ResourceLimits(**changes)


def test_limits_can_be_reduced_for_guard_tests():
    assert replace(ResourceLimits(), child_rss_bytes=1024).child_rss_bytes == 1024


def test_run_config_is_frozen_with_distinct_seed_roles():
    config = RunConfig()
    assert (config.split_seed, config.model_seed) == (42, 42)
    assert replace(config, model_seed=43).split_seed == 42
    with pytest.raises(FrozenInstanceError):
        config.model_seed = 44
    for seed in (-1, 2**32, True, 1.5):
        with pytest.raises(ValueError):
            RunConfig(model_seed=seed)


def test_windows_home_uses_local_appdata_without_creating_it(tmp_path):
    root = tmp_path / "local"
    settings = load_settings({"LOCALAPPDATA": str(root)}, platform="win32")
    assert settings.app_home == root / "ModelArena"
    assert not root.exists()
    assert settings.metadata_path != settings.mlflow_database_path
    assert settings.mlflow_tracking_uri.startswith("sqlite:///")
    assert settings.mlflow_artifacts_path.is_relative_to(settings.app_home)


def test_posix_home_uses_xdg_or_private_default(tmp_path):
    assert load_settings({"XDG_DATA_HOME": str(tmp_path)}, platform="linux").app_home == tmp_path / "modelarena"
    assert load_settings({}, platform="linux", user_home=tmp_path).app_home == tmp_path / ".local/share/modelarena"


def test_explicit_home_and_benchmark_threads(tmp_path):
    settings = load_settings({"MODELARENA_HOME": str(tmp_path / "private"),
                              "MODELARENA_COMPUTE_THREADS": "6"})
    assert settings.app_home == tmp_path / "private"
    assert settings.limits.compute_threads == 6
    with pytest.raises(FrozenInstanceError):
        settings.app_home = tmp_path


@pytest.mark.parametrize("value", ["", "relative", "../outside", "C:relative", "\\\\server\\share\\home", "//server/share/home"])
def test_home_rejects_relative_or_network_paths(value):
    with pytest.raises(ValueError):
        load_settings({"MODELARENA_HOME": value})


def test_home_rejects_synchronized_path_and_existing_file(tmp_path):
    with pytest.raises(ValueError):
        Settings(app_home=tmp_path / "OneDrive" / "ModelArena")
    file = tmp_path / "file"
    file.touch()
    with pytest.raises(ValueError):
        Settings(app_home=file)


@pytest.mark.parametrize("folder", ["OneDrive", "OneDrive - Personal", "Dropbox", "Google Drive", "GoogleDrive", "iCloudDrive"])
def test_known_sync_folders_cannot_host_active_storage(tmp_path, folder):
    with pytest.raises(ValueError, match="synchronized"):
        Settings(app_home=tmp_path / folder / "ModelArena")


@pytest.mark.parametrize("value", ["3", "0", "many", "4.0", ""])
def test_environment_rejects_invalid_threads(tmp_path, value):
    with pytest.raises(ValueError):
        load_settings({"MODELARENA_HOME": str(tmp_path), "MODELARENA_COMPUTE_THREADS": value})


def test_windows_missing_localappdata_is_explicit():
    with pytest.raises(ValueError, match="LOCALAPPDATA"):
        load_settings({}, platform="win32")
