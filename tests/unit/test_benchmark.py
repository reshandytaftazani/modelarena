"""Probe policy, measured sampling, and training-only scratch isolation."""

import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ml.contracts import ResourceLimits
from scripts.benchmark import ResourceRecorder, TrialConfig, normalized_cpu_percent, run_initial_probe


def test_cpu_normalization_uses_machine_capacity():
    assert normalized_cpu_percent(2.0, 1.0, 4) == 50.0
    assert normalized_cpu_percent(0.0, 1.0, 4) == 0.0


def test_recorder_handles_work_shorter_than_a_clock_tick():
    recorder = ResourceRecorder()
    with recorder:
        pass
    assert recorder.summary()["wall_seconds"] > 0


@pytest.mark.parametrize("cpu,wall,cores", [(-1, 1, 4), (1, 0, 4), (1, 1, 0),
                                                     (math.nan, 1, 4), (1, math.inf, 4)])
def test_invalid_observations_do_not_produce_metrics(cpu, wall, cores):
    with pytest.raises(ValueError):
        normalized_cpu_percent(cpu, wall, cores)


def test_recorder_samples_real_process_and_stops_on_failure():
    recorder = ResourceRecorder()
    with pytest.raises(RuntimeError, match="workload failed"):
        with recorder:
            time.sleep(0.6)
            raise RuntimeError("workload failed")
    result = recorder.summary()
    assert result["wall_seconds"] >= 0.6
    assert len(result["samples"]) >= 3
    assert result["peak_rss_bytes"] > 0
    assert result["minimum_available_bytes"] > 0
    assert all(a["elapsed_seconds"] <= b["elapsed_seconds"]
               for a, b in zip(result["samples"], result["samples"][1:]))
    count = len(result["samples"])
    time.sleep(0.55)
    assert len(recorder.summary()["samples"]) == count


@pytest.mark.parametrize("threads", [2, 4, 6])
def test_trial_configuration_retains_caps_for_later_comparison(threads):
    config = TrialConfig(limits=ResourceLimits(compute_threads=threads))
    assert config.to_manifest()["resource_limits"]["child_rss_bytes"] == 4 * 1024**3
    assert config.to_manifest()["resource_limits"]["cv_parallelism"] == 1
    assert config.to_manifest()["scope"] == "initial_probe_only"


def test_initial_probe_uses_only_training_rows_and_no_holdout_metrics(tmp_path):
    report = run_initial_probe(TrialConfig(), tmp_path)
    assert report["status"] == "completed"
    assert report["parser"]["row_count"] == 455
    split = report["scratch_partition"]
    assert split["training_rows"] == 455
    assert split["reserved_rows"] == 114
    assert split["holdout_scored"] is False
    assert len(report["dummy_cv"]["folds"]) == 5
    assert all(fold["fit_rows"] + fold["validation_rows"] == 455
               for fold in report["dummy_cv"]["folds"])
    assert report["resources"]["wall_seconds"] > 0
    assert report["resources"]["peak_rss_bytes"] > 0
    assert all(pool["num_threads"] == 1 for pool in report["observed_threadpools"]
               if pool["user_api"] == "openmp")
    assert not list(tmp_path.iterdir())  # Scratch CSV is never retained.
    json.dumps(report, allow_nan=False)


def test_cli_rejects_unplanned_stage_and_invalid_thread_count():
    for args in (["--stage", "full"], ["--threads", "8"]):
        result = subprocess.run([sys.executable, "-m", "scripts.benchmark", *args],
                                capture_output=True, text=True)
        assert result.returncode == 2
