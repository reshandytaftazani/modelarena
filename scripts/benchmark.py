"""Initial parser/Dummy probe and seed study. Run as a fresh process, outside the API.

This is instrumentation preparation for Day 21, not the full comparison. The
sampler observes the current process only; owned-child sampling and termination
now live in ``apps.api.workers.monitor``, which this module reuses for the single
normalized-CPU definition. No network access, model export, production split or
final holdout evaluation.
"""

import argparse
from dataclasses import asdict, dataclass, field, replace as dc_replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
import uuid

import psutil

from apps.api.settings import load_settings
from apps.api.workers.monitor import normalized_cpu_percent
from ml.contracts import ResourceLimits


class ResourceRecorder:
    """Monotonic wall time with start/end and 0.5-second process samples."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._samples: list[dict[str, float | int]] = []
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    def _capture(self) -> None:
        cpu = self._process.cpu_times()
        self._samples.append({
            "elapsed_seconds": time.perf_counter() - self._started,
            "rss_bytes": self._process.memory_info().rss,
            "process_cpu_seconds": cpu.user + cpu.system,
            "available_bytes": psutil.virtual_memory().available,
        })

    def _sample_loop(self) -> None:
        try:
            while not self._stop.wait(0.5):
                self._capture()
        except Exception as error:
            self._error = error

    def __enter__(self) -> "ResourceRecorder":
        if self._thread is not None:
            raise RuntimeError("A recorder cannot be reused")
        self._started = time.perf_counter()
        self._capture()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self._capture()
        if self._error is not None and exc_type is None:
            raise RuntimeError("Resource sampling failed") from self._error

    def summary(self) -> dict[str, object]:
        if not self._stop.is_set() or len(self._samples) < 2:
            raise RuntimeError("Recorder has not finished")
        first, last = self._samples[0], self._samples[-1]
        wall = last["elapsed_seconds"] - first["elapsed_seconds"]
        cpu = max(0.0, last["process_cpu_seconds"] - first["process_cpu_seconds"])
        return {
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "normalized_cpu_percent": normalized_cpu_percent(cpu, wall, psutil.cpu_count() or 1),
            "logical_cores": psutil.cpu_count() or 1,
            "peak_rss_bytes": max(item["rss_bytes"] for item in self._samples),
            "minimum_available_bytes": min(item["available_bytes"] for item in self._samples),
            "sample_interval_seconds": 0.5,
            "peak_note": "Sampled current-process RSS; transient peaks and descendants may be missed.",
            "samples": list(self._samples),
        }


@dataclass(frozen=True, slots=True)
class TrialConfig:
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    split_seed: int = 42
    model_seed: int = 42

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1, "scope": "initial_probe_only",
            "split_seed": self.split_seed, "model_seed": self.model_seed,
            "resource_limits": asdict(self.limits),
            "workload": "Breast Cancer training-only CSV parse and serial five-fold DummyClassifier",
            "dummy_parameters": {"strategy": "most_frequent", "random_state": self.model_seed},
            "primary_metric": "f1", "cv_folds": 5, "holdout_fraction": 0.2,
            "resource_policy_status": "provisional; no final thread or RSS cap selection",
            "enforcement": "thread pools bounded; RSS/time termination deferred to worker integration",
        }


def _index_hash(indices) -> str:
    payload = json.dumps([int(index) for index in indices], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def run_initial_probe(config: TrialConfig, scratch_parent: Path) -> dict[str, object]:
    """Use bundled data; only reserved-training rows enter parsing or CV."""
    recorder = ResourceRecorder()
    with recorder:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_breast_cancer
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold, train_test_split
        from threadpoolctl import threadpool_info, threadpool_limits

        from ml.data.csv_io import validate_csv

        with (
            threadpool_limits(limits=config.limits.compute_threads, user_api="blas"),
            threadpool_limits(limits=1, user_api="openmp"),
        ):
            source = load_breast_cancer()
            train_ids, reserved_ids = train_test_split(
                np.arange(len(source.target)), test_size=0.2,
                random_state=config.split_seed, stratify=source.target,
            )
            # Discard the reserved indices without indexing source.data by them.
            reserved_count = len(reserved_ids)
            del reserved_ids
            x, y = source.data[train_ids].copy(), source.target[train_ids].copy()
            del source
            with tempfile.TemporaryDirectory(prefix="initial-probe-", dir=scratch_parent) as folder:
                path = Path(folder) / "training.csv"
                frame = pd.DataFrame(x)
                frame["target"] = y
                frame.to_csv(path, index=False, lineterminator="\n")
                started = time.perf_counter()
                parsed = validate_csv(path, config.limits, expected_root=Path(folder))
                parser_seconds = time.perf_counter() - started
                # Fit from the validated snapshot, not a separately loaded data path.
                matrix = np.asarray(parsed.rows, dtype=np.float64)
                x, y = matrix[:, :-1], matrix[:, -1].astype(np.int64)
                folds = []
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.split_seed)
                for fit_ids, validation_ids in cv.split(x, y):
                    started = time.perf_counter()
                    model = DummyClassifier(strategy="most_frequent", random_state=config.model_seed)
                    model.fit(x[fit_ids], y[fit_ids])
                    prediction = model.predict(x[validation_ids])
                    folds.append({
                        "fit_rows": len(fit_ids), "validation_rows": len(validation_ids),
                        "fit_row_ids_sha256": _index_hash(train_ids[fit_ids]),
                        "validation_row_ids_sha256": _index_hash(train_ids[validation_ids]),
                        "f1": float(f1_score(y[validation_ids], prediction, zero_division=0)),
                        "wall_seconds": time.perf_counter() - started,
                    })
                parser = {"wall_seconds": parser_seconds, "row_count": parsed.row_count,
                          "byte_count": parsed.byte_count, "raw_sha256": parsed.raw_sha256,
                          "canonical_sha256": parsed.canonical_sha256,
                          "parse_policy_version": parsed.parse_policy_version}
            pools = threadpool_info()
    resources = recorder.summary()
    return {
        "schema_version": 1, "status": "completed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config.to_manifest(), "parser": parser,
        "scratch_partition": {"training_rows": len(train_ids), "reserved_rows": reserved_count,
                              "training_row_ids_sha256": _index_hash(train_ids),
                              "holdout_scored": False,
                              "note": "Scratch probe only; production immutable split is scheduled for Day 4."},
        "dummy_cv": {"folds": folds, "f1_mean": float(np.mean([f["f1"] for f in folds]))},
        "resources": resources, "observed_threadpools": pools,
        "rss_soft_cap_exceeded": resources["peak_rss_bytes"] > config.limits.child_rss_bytes,
        "system_reserve_preserved_in_samples": resources["minimum_available_bytes"] >= config.limits.system_reserve_bytes,
        "performance_decision": "none; single initial probe, no comparative or responsiveness evidence",
    }


def run_seed_study(config: TrialConfig, scratch_parent: Path) -> dict[str, object]:
    """Execute fixed-fold three-model-seed study (42/43/44) on both-task fixtures at provisional T=4.

    Distinguishes model seeds from split seeds:
    - Split seed is frozen at 42 (defining fixed fold partition identities).
    - Model seeds 42, 43, 44 are evaluated on the exact same folds.
    - Zero holdout access guarded by PoisonHoldout.
    - Prepares frozen Day 21 workloads and policy assertions.
    """
    recorder = ResourceRecorder()
    with recorder:
        import numpy as np
        import torch
        from threadpoolctl import threadpool_info, threadpool_limits

        from ml.contracts import (
            DatasetConfig,
            FeatureColumn,
            FeatureSchema,
            FeatureType,
            RunConfig,
            TaskType,
            TrainingContext,
        )
        from ml.data.csv_io import validate_csv
        from ml.data.splits import create_split, training_from_manifest
        from ml.data.validation import configure_dataset
        from ml.training.cv import evaluate_candidate

        orig_torch_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(config.limits.compute_threads)
            with (
                threadpool_limits(limits=config.limits.compute_threads, user_api="blas"),
                threadpool_limits(limits=1, user_api="openmp"),
            ):
                study_results: dict[str, dict[str, Any]] = {}
                seed_list = (42, 43, 44)
                split_seed = 42

                with tempfile.TemporaryDirectory(prefix="seed-study-", dir=scratch_parent) as folder:
                    scratch_dir = Path(folder)
                    tasks = (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION)
                    for task in tasks:
                        task_key = task.value
                        study_results[task_key] = {}

                        # Synthetic deterministic fixture (100 rows, 5 folds)
                        rng = np.random.RandomState(split_seed)
                        row_count = 100
                        x_num = rng.uniform(-3.0, 3.0, size=row_count)
                        categories = [f"cat_{i % 4}" for i in range(row_count)]

                        if task == TaskType.BINARY_CLASSIFICATION:
                            logits = 0.8 * x_num - 0.3 * (x_num**2) + rng.uniform(-0.5, 0.5, size=row_count)
                            targets = ["yes" if val > 0.0 else "no" for val in logits]
                            dataset_cfg = DatasetConfig(
                                task=task,
                                target="target",
                                features=FeatureSchema((
                                    FeatureColumn("num_feat", FeatureType.NUMERIC),
                                    FeatureColumn("cat_feat", FeatureType.CATEGORICAL),
                                )),
                                positive_label="yes",
                            )
                        else:
                            targets = [float(2.5 * x - 0.5 * (x**2) + rng.normal(0, 0.2)) for x in x_num]
                            dataset_cfg = DatasetConfig(
                                task=task,
                                target="target",
                                features=FeatureSchema((
                                    FeatureColumn("num_feat", FeatureType.NUMERIC),
                                    FeatureColumn("cat_feat", FeatureType.CATEGORICAL),
                                )),
                            )

                        csv_path = scratch_dir / f"{task_key}.csv"
                        lines = ["num_feat,cat_feat,target"]
                        for x, cat, y in zip(x_num, categories, targets, strict=True):
                            lines.append(f"{x:.6f},{cat},{y}")
                        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                        parsed = validate_csv(csv_path, config.limits, expected_root=scratch_dir)
                        split = create_split(parsed, dataset_cfg, RunConfig(split_seed=split_seed))
                        configured = configure_dataset(parsed, dataset_cfg)

                        class PoisonHoldout(tuple):
                            def __getitem__(self, index):
                                if index in split.holdout_row_ids:
                                    raise AssertionError("CV attempted forbidden holdout access")
                                return super().__getitem__(index)

                        guarded = dc_replace(
                            configured,
                            rows=PoisonHoldout(configured.rows),
                            targets=PoisonHoldout(configured.targets),
                        )
                        training = training_from_manifest(guarded, split)
                        context = TrainingContext(training, split)

                        # Representative classical estimator: random_forest
                        # Representative neural estimator: mlp
                        families = ("random_forest", "mlp")
                        for family in families:
                            family_records: dict[str, Any] = {}
                            seed_means: dict[str, float] = {}

                            for model_seed in seed_list:
                                run_cfg = RunConfig(
                                    family=family,
                                    split_seed=split_seed,
                                    model_seed=model_seed,
                                    limits=config.limits,
                                )
                                candidate = evaluate_candidate(context, run_cfg)
                                if (not candidate.complete or not candidate.eligible
                                        or candidate.split_sha256 != split.manifest_sha256
                                        or candidate.run_config != run_cfg
                                        or candidate.expected_fold_indices != tuple(f.fold_index for f in split.folds)
                                        or any(actual.validation_row_ids != frozen.validation_row_ids
                                               for actual, frozen in zip(candidate.folds, split.folds, strict=True))):
                                    raise RuntimeError("Seed study candidate failed frozen split/config or fold coverage checks")

                                primary_val = next(
                                    s.mean for s in candidate.summary if s.name == candidate.primary_metric
                                )
                                primary_std = next(
                                    s.std for s in candidate.summary if s.name == candidate.primary_metric
                                )
                                seed_means[str(model_seed)] = float(primary_val)

                                fold_records = []
                                for fold in candidate.folds:
                                    metric_item = fold.metrics.get(candidate.primary_metric) if fold.metrics else None
                                    primary_fold_val = metric_item.value if metric_item else None
                                    fold_records.append({
                                        "fold_index": fold.fold_index,
                                        "validation_ids_sha256": hashlib.sha256(json.dumps(fold.validation_row_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
                                        "validation_rows": len(fold.validation_row_ids),
                                        candidate.primary_metric: float(primary_fold_val) if primary_fold_val is not None else None,
                                        "fit_seconds": fold.fit_seconds,
                                        "score_seconds": fold.score_seconds,
                                        "best_epoch": fold.best_epoch,
                                        "epochs_trained": fold.epochs_trained,
                                        "stop_reason": fold.stop_reason,
                                    })

                                family_records[str(model_seed)] = {
                                    "model_seed": model_seed,
                                    "split_sha256": candidate.split_sha256,
                                    "dataset_config_sha256": split.config_sha256,
                                    "run_config": asdict(run_cfg),
                                    "primary_metric": candidate.primary_metric,
                                    "mean": float(primary_val),
                                    "std": float(primary_std),
                                    "total_seconds": candidate.total_seconds,
                                    "folds": fold_records,
                                }

                            mean_vals = list(seed_means.values())
                            study_results[task_key][family] = {
                                "seeds": family_records,
                                "variation": {
                                    "mean_of_means": float(np.mean(mean_vals)),
                                    "std_of_means": float(np.std(mean_vals, ddof=0)),
                                    "min_mean": float(np.min(mean_vals)),
                                    "max_mean": float(np.max(mean_vals)),
                                    "spread": float(np.max(mean_vals) - np.min(mean_vals)),
                                },
                            }
                pools = threadpool_info()
        finally:
            torch.set_num_threads(orig_torch_threads)

    resources = recorder.summary()
    return {
        "schema_version": 1,
        "stage": "seeds",
        "status": "completed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "split_seed": split_seed,
            "model_seeds": list(seed_list),
            "threads": config.limits.compute_threads,
            "representative_classical": "random_forest",
            "representative_neural": "mlp",
            "tasks": ["binary_classification", "regression"],
        },
        "seed_interpretation": {
            "split_seed": split_seed,
            "split_seed_role": "Defines row partitioning into holdout and CV folds; held constant across all trials.",
            "model_seeds": list(seed_list),
            "model_seed_role": "Controls pseudo-random number generation within the estimator (initial neural weights, tree splits/subsampling, batch shuffle); evaluated on identical frozen folds.",
            "holdout_access": "Training view created using index-guarded rows and targets; CV receives no holdout values or path. Fixture generation, validation and split construction precede this boundary.",
            "timing_policy": "Timings excluded from deterministic metric tolerance checks; CPU execution time varies naturally.",
        },
        "study_results": study_results,
        "frozen_day_21_preparation": {
            "workloads": {
                "BC-classical": "Breast Cancer training partition, 5-fold CV for DummyClassifier, LogisticRegression, RandomForestClassifier and HistGradientBoostingClassifier.",
                "Housing-classical": "California Housing training partition, 5-fold CV for DummyRegressor, Ridge, RandomForestRegressor and HistGradientBoostingRegressor.",
                "BC-small-MLP": "One frozen outer-training fold of Breast Cancer, default 64/32 network, batch 32, Adam 1e-3, max 100 epochs, patience 10.",
            },
            "policy_assertions": {
                "thread_trials_model_seed": 42,
                "thread_trials_fixed_seed_rule": "Thread scaling trials across T=2, 4, 6 MUST use fixed model seed 42 to prevent confounding model seed variance with thread performance.",
                "deterministic_metric_tolerance": "abs=1e-6 or rel=1e-5 on identical environment, seeds, threads, and configs with timings excluded.",
                "timing_exclusion": "Wall and CPU execution timings are excluded from deterministic metric tolerance checks.",
                "provisional_thread_cap": config.limits.compute_threads,
                "provisional_rss_cap_bytes": 4 * 1024**3,
                "provisional_system_reserve_bytes": 4 * 1024**3,
                "day_21_execution_status": "unexecuted; frozen preparation only",
            },
        },
        "resources": resources,
        "observed_threadpools": pools,
        "rss_soft_cap_exceeded": resources["peak_rss_bytes"] > config.limits.child_rss_bytes,
        "system_reserve_preserved_in_samples": resources["minimum_available_bytes"] >= config.limits.system_reserve_bytes,
        "performance_decision": "none; educational seed study on fixed folds, Day 21 full matrix benchmarks remain unexecuted",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["initial", "seeds"], default="initial")
    parser.add_argument("--threads", type=int, choices=[2, 4, 6], default=4)
    args = parser.parse_args()
    # Applied before any numerical library import in this fresh command process.
    os.environ["OMP_NUM_THREADS"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    settings = load_settings()
    config = TrialConfig(limits=ResourceLimits(compute_threads=args.threads))
    output = settings.app_home / "benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != output.absolute():
        raise ValueError("Benchmark directory cannot be redirected")
    lock_name = f"{args.stage}.lock"
    lock = output / lock_name
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        parser.exit(1, f"A {args.stage} probe lock exists. Verify the previous process has exited before removing it.\n")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()}, handle)
        from scripts.profile_hardware import _probe, build_inventory

        repository = Path(__file__).resolve().parents[1]
        inventory = build_inventory(repository)
        dirty = _probe(["git", "status", "--porcelain", "--untracked-files=normal"], repository)

        if args.stage == "seeds":
            report = run_seed_study(config, output)
            report_prefix = "seeds"
        else:
            report = run_initial_probe(config, output)
            report_prefix = "initial"

        from ml.tracking.manifest import execution_provenance
        report["execution_provenance"] = execution_provenance()
        report["inventory"] = inventory
        report["code_dirty"] = bool(dirty.get("value")) if dirty["status"] == "observed" else None
        report["measurement_note"] = "Includes numerical imports, CSV generation, parsing and CV; inventory excluded."
        destination = output / f"{report_prefix}-{uuid.uuid4().hex}.json"
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")

        if args.stage == "seeds":
            summary = {
                "stage": report["stage"],
                "status": report["status"],
                "report_path": str(destination),
                "configuration": report["configuration"],
                "seed_interpretation": report["seed_interpretation"],
                "study_summary": {
                    task: {
                        fam: data["variation"]
                        for fam, data in fams.items()
                    }
                    for task, fams in report["study_results"].items()
                },
                "frozen_day_21_preparation": report["frozen_day_21_preparation"],
                "resources": {
                    "wall_seconds": report["resources"]["wall_seconds"],
                    "cpu_seconds": report["resources"]["cpu_seconds"],
                    "peak_rss_bytes": report["resources"]["peak_rss_bytes"],
                    "normalized_cpu_percent": report["resources"]["normalized_cpu_percent"],
                },
            }
            print(json.dumps(summary, indent=2))
        else:
            print(json.dumps({"report_path": str(destination), "resources": report["resources"],
                              "parser": report["parser"], "performance_decision": report["performance_decision"]}, indent=2))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
