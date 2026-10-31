"""Canonical JSON and honest execution provenance for private CV bundles."""

from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
import sys


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def execution_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                                capture_output=True, text=True, check=True, timeout=5).stdout
        dirty = bool(status)
        diff = subprocess.run(["git", "diff", "--binary", "HEAD", "--"], cwd=root,
                              capture_output=True, check=True, timeout=5).stdout
        tracked_diff_sha256 = digest_bytes(diff)
    except (OSError, subprocess.SubprocessError):
        revision, dirty, tracked_diff_sha256 = "unversioned", None, None
    packages = {}
    for name in ("mlflow", "scikit-learn", "numpy", "pandas", "torch", "threadpoolctl"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    lock = root / "uv.lock"
    return {"code": {"revision": revision or "unversioned", "dirty": dirty,
                     "tracked_diff_sha256": tracked_diff_sha256},
            "python": sys.version, "platform": platform.platform(), "packages": packages,
            "lock_sha256": file_digest(lock) if lock.is_file() else None}


def make_manifest(job, context, run, dataset: dict, candidates: list, telemetry: dict | None,
                  artifact_sha256: dict, run_ids: dict) -> dict:
    split = context.split
    # Identities are hashed here: the private split store owns the complete indices.
    folds = [{"fold_index": f.fold_index,
              "fit_row_ids_sha256": digest_bytes(canonical_bytes(f.fit_row_ids)),
              "validation_row_ids_sha256": digest_bytes(canonical_bytes(f.validation_row_ids)),
              "inner_fit_row_ids_sha256": digest_bytes(canonical_bytes(f.inner.fit_row_ids)),
              "inner_validation_row_ids_sha256": digest_bytes(canonical_bytes(f.inner.validation_row_ids)),
              "inner_seed": f.inner.seed}
             for f in split.folds]
    return {"schema_version": 1, "scope": "training_cv_only", "job_id": job.job_id,
            "attempt_id": job.attempt_id, "experiment_id": job.experiment_id,
            "dataset": dataset, "dataset_config": asdict(context.training.config),
            "split": {"manifest_sha256": split.manifest_sha256, "split_seed": split.split_seed,
                      "train_sha256": split.train_sha256, "holdout_sha256": split.holdout_sha256,
                      "algorithm": split.algorithm, "fold_reason": split.fold_reason, "folds": folds},
            "preprocessing": {"numeric": "median imputation with empty features retained, standard scaling",
                              "categorical": "constant imputation, one-hot encoding with unknown categories ignored",
                              "fit_scope": "classical: outer fit; MLP: inner fit only", "dense_output_guarded": True},
            "run_config": asdict(run),
            "candidates": candidates, "resource_telemetry": telemetry,
            "artifact_sha256": artifact_sha256, "mlflow_run_ids": run_ids,
            **execution_provenance()}
