"""Capability-gated access to the private held-out partition.

Reading is not publication: a worker must still stage and atomically publish a
complete result. No finalization worker is connected to this reader yet.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from apps.api.database.repository import JobRepository
from apps.api.services.partitions import _folder
from apps.api.services.workflow import parse_config
from ml.contracts import DatasetConfig, FeatureValue, HoldoutCapability, MetricBundle, ResourceLimits, TaskType, TrainingView
from ml.data.csv_io import ValidatedDataset
from ml.data.splits import parse_manifest
from ml.data.validation import configure_dataset
from ml.evaluation.metrics import score_predictions
from ml.tracking.manifest import canonical_bytes, digest_bytes


@dataclass(frozen=True, slots=True)
class HoldoutView:
    config: DatasetConfig
    row_ids: tuple[int, ...] = field(repr=False)
    rows: tuple[tuple[FeatureValue, ...], ...] = field(repr=False)
    targets: tuple[int | float, ...] = field(repr=False)
    label_mapping: tuple[tuple[str, int], ...]


def _read_verified(path, digest: str, limit: int, partition: str, ids: tuple[int, ...]) -> dict:
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("Unsafe partition file")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Partition checksum mismatch")
    payload = json.loads(content)
    if (set(payload) != {"columns", "partition", "records"} or payload["partition"] != partition
            or tuple(record[0] for record in payload["records"]) != ids):
        raise ValueError("Invalid partition")
    return payload


def read_holdout(repository: JobRepository, capability: HoldoutCapability) -> HoldoutView:
    """Consume authorization before opening the real holdout, at most once.

    A crash or read error after authorization consumes the opportunity. The
    repository checks the active owner, generation, job and attempt first.
    """
    if not isinstance(capability, HoldoutCapability):
        raise PermissionError("Holdout capability required")
    fin = repository.consume_holdout_authorization(capability)
    with repository._connection() as db:
        row = db.execute(
            "SELECT e.run_json,c.config_json,s.manifest_json FROM experiments e "
            "JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN splits s ON s.split_id=c.split_id WHERE e.experiment_id=?",
            (fin["experiment_id"],),
        ).fetchone()
    if row is None:
        raise ValueError("Frozen experiment unavailable")
    split = parse_manifest(row["manifest_json"].encode("utf-8"))
    config = parse_config(json.loads(row["config_json"]))
    limits = ResourceLimits(**json.loads(row["run_json"])["limits"])
    limit = limits.upload_bytes * 6 + limits.max_rows * 32 + 65536
    folder = _folder(repository, fin["experiment_id"])
    train = _read_verified(folder / "train.json", split.train_sha256, limit, "train", split.train_row_ids)
    held = _read_verified(folder / "holdout.json", split.holdout_sha256, limit, "holdout", split.holdout_row_ids)
    if train["columns"] != held["columns"]:
        raise ValueError("Partition schemas differ")
    records = train["records"] + held["records"]
    data = ValidatedDataset(tuple(train["columns"]), tuple(tuple(record[1]) for record in records),
                            0, "", split.dataset_sha256)
    configured = configure_dataset(data, config)
    offset = len(train["records"])
    return HoldoutView(config, split.holdout_row_ids, configured.rows[offset:],
                       configured.targets[offset:], configured.label_mapping)


def score_holdout(model, view: HoldoutView, task: TaskType) -> MetricBundle:
    """Score the frozen model once on holdout records, using fixed probability threshold for classification."""
    training_view = TrainingView(view.config, view.row_ids, view.rows, view.targets, view.label_mapping)
    dtype = np.float32 if model.plan.epoch_budget else np.float64
    encoded = model.preprocessor.transform(training_view, dtype=dtype)
    y_true = np.asarray(view.targets)
    if task == TaskType.BINARY_CLASSIFICATION:
        if model.plan.epoch_budget is not None:
            model.model.eval()
            with torch.no_grad():
                tensor_x = torch.from_numpy(encoded.astype(np.float32))
                logits = model.model(tensor_x).squeeze(-1)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                predictions = (probabilities >= 0.5).astype(int)
        else:
            if hasattr(model.model, "predict_proba"):
                probabilities = model.model.predict_proba(encoded)[:, 1]
                predictions = (probabilities >= 0.5).astype(int)
            else:
                predictions = model.model.predict(encoded)
                probabilities = None
        return score_predictions(task, y_true, predictions, probabilities)
    else:
        if model.plan.epoch_budget is not None:
            model.model.eval()
            with torch.no_grad():
                tensor_x = torch.from_numpy(encoded.astype(np.float32))
                logits = model.model(tensor_x).squeeze(-1)
                predictions = logits.cpu().numpy()
        else:
            predictions = model.model.predict(encoded)
        return score_predictions(task, y_true, predictions)


def stage_final_bundle(home: Path, finalization_id: str, scope_id: str, attempt_id: str,
                       candidate_id: str, final_result: dict) -> tuple[Path, str]:
    """Atomically stage the complete checksummed final result bundle."""
    root = Path(home).resolve() / "final" / scope_id
    root.mkdir(parents=True, exist_ok=True)
    staging = root / ("staging-" + attempt_id)
    final = root / attempt_id
    staging.mkdir(parents=True, exist_ok=True)

    results_bytes = canonical_bytes(final_result)
    results_sha = digest_bytes(results_bytes)
    with (staging / "results.json").open("xb") as s:
        s.write(results_bytes)
        s.flush()
        os.fsync(s.fileno())

    manifest = {
        "schema_version": 1,
        "origin": "modelarena-final-v1",
        "finalization_id": finalization_id,
        "scope_id": scope_id,
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "artifact_sha256": {"results.json": results_sha},
    }
    manifest_bytes = canonical_bytes(manifest)
    manifest_sha = digest_bytes(manifest_bytes)
    with (staging / "manifest.json").open("xb") as s:
        s.write(manifest_bytes)
        s.flush()
        os.fsync(s.fileno())

    staging.rename(final)
    return final, manifest_sha


_REQUIRED_FINAL_RESULT_KEYS = frozenset({
    "finalization_id",
    "scope_id",
    "candidate_id",
    "task",
    "primary_metric",
    "metrics",
    "evaluation_seconds",
    "evaluated_at",
})


def validate_final_result_payload(payload: object, expected: object = None) -> bool:
    """Validate that the final result payload is structurally sound and matches expected identities."""
    if not isinstance(payload, dict):
        return False
    if not _REQUIRED_FINAL_RESULT_KEYS.issubset(payload.keys()):
        return False
    if not isinstance(payload["finalization_id"], str) or not payload["finalization_id"].strip():
        return False
    if not isinstance(payload["scope_id"], str) or not payload["scope_id"].strip():
        return False
    if not isinstance(payload["candidate_id"], str) or not payload["candidate_id"].strip():
        return False
    task_val = payload["task"].value if hasattr(payload["task"], "value") else str(payload["task"])
    if task_val not in (TaskType.BINARY_CLASSIFICATION.value, TaskType.REGRESSION.value):
        return False
    if not isinstance(payload["primary_metric"], str) or not payload["primary_metric"].strip():
        return False
    if (type(payload["evaluation_seconds"]) not in (int, float)
            or isinstance(payload["evaluation_seconds"], bool)
            or payload["evaluation_seconds"] < 0):
        return False
    if not isinstance(payload["evaluated_at"], str) or not payload["evaluated_at"].strip():
        return False
    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or "task" not in metrics or "metrics" not in metrics:
        return False
    if not isinstance(metrics.get("metrics"), (list, tuple)):
        return False

    if expected is not None:
        exp_dict = dict(expected) if hasattr(expected, "keys") else expected.__dict__ if hasattr(expected, "__dict__") else {}
        for key in ("finalization_id", "scope_id", "candidate_id", "candidate_digest"):
            if key in exp_dict and exp_dict[key] is not None:
                if payload.get(key) != exp_dict[key]:
                    return False
        if "evidence_sha256" in exp_dict and exp_dict["evidence_sha256"] is not None:
            if payload.get("evidence_sha256") is not None and payload.get("evidence_sha256") != exp_dict["evidence_sha256"]:
                return False
    return True
