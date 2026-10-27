"""Initial parser/Dummy probe. Run as a fresh process, outside the API.

This is instrumentation preparation for Day 21, not the full comparison. The
sampler observes the current process only; owned-child sampling and termination
now live in ``apps.api.workers.monitor``, which this module reuses for the single
normalized-CPU definition. No network access, model export, production split or
final holdout evaluation.
"""

import argparse
import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["initial"], default="initial")
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
    lock = output / "initial.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        parser.exit(1, "An initial probe lock exists. Verify the previous process has exited before removing it.\n")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()}, handle)
        from scripts.profile_hardware import _probe, build_inventory

        repository = Path(__file__).resolve().parents[1]
        inventory = build_inventory(repository)
        dirty = _probe(["git", "status", "--porcelain", "--untracked-files=normal"], repository)
        report = run_initial_probe(config, output)
        report["inventory"] = inventory
        report["code_dirty"] = bool(dirty.get("value")) if dirty["status"] == "observed" else None
        report["measurement_note"] = "Includes numerical imports, CSV generation, parsing and CV; inventory excluded."
        destination = output / f"initial-{uuid.uuid4().hex}.json"
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"report_path": str(destination), "resources": report["resources"],
                          "parser": report["parser"], "performance_decision": report["performance_decision"]}, indent=2))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
