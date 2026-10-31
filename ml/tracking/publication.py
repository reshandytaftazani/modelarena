"""Stage complete private bundles, then atomically rename before DB commit."""

import os
from pathlib import Path
from uuid import uuid4

from ml.contracts import TaskType
from ml.models.classical import make_classical
from ml.tracking import mlflow_store
from ml.tracking.manifest import canonical_bytes, digest_bytes, make_manifest


def _clean_curve(curve):
    if curve is None:
        return None
    res = []
    for point in curve:
        if isinstance(point, dict):
            res.append({"epoch": int(point["epoch"]), "train_loss": float(point["train_loss"]),
                         "val_loss": float(point["val_loss"]) if point.get("val_loss") is not None else None})
        else:
            res.append({"epoch": int(point.epoch), "train_loss": float(point.train_loss),
                         "val_loss": float(point.val_loss) if point.val_loss is not None else None})
    return res


def _clean_payload(payload: dict) -> dict:
    if "candidates" not in payload:
        return payload
    new_candidates = []
    for c in payload["candidates"]:
        cand = dict(c)
        if "folds" in cand:
            cleaned_folds = []
            for f in cand["folds"]:
                fold = dict(f)
                if "loss_curve" in fold and fold["loss_curve"] is not None:
                    fold["loss_curve"] = _clean_curve(fold["loss_curve"])
                cleaned_folds.append(fold)
            cand["folds"] = cleaned_folds
        new_candidates.append(cand)
    return {**payload, "candidates": new_candidates}


def _write(path: Path, value: object) -> str:
    data = canonical_bytes(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return digest_bytes(data)


def stage_attempt(home: Path, job, context, run, dataset: dict, payload: dict, telemetry: dict | None) -> dict:
    root = Path(home).resolve() / "results"
    root.mkdir(parents=True, exist_ok=True)
    # Both staging and final directories are siblings on the same filesystem.
    staging = root / ("staging-" + job.attempt_id)
    final = root / job.attempt_id
    staging.mkdir()
    payload = _clean_payload(payload)
    results_sha = _write(staging / "results.json", payload)
    candidates = payload["candidates"]
    ids, client = mlflow_store.track_attempt(home, job, candidates)
    summaries = []
    task = context.training.config.task
    for candidate in candidates:
        if candidate["family"] == "mlp":
            model_parameters = {
                "architecture": "MLP(in_features->64->ReLU->32->ReLU->1)",
                "hidden_dims": [64, 32],
                "lr": 1e-3,
                "batch_size": 32,
                "max_epochs": candidate["run_config"]["limits"]["max_epochs"],
                "patience": 10,
                "min_delta": 1e-6,
                "loss": "BCEWithLogitsLoss" if task == TaskType.BINARY_CLASSIFICATION else "MSELoss",
            }
        else:
            model = make_classical(task, candidate["family"], candidate["model_seed"])
            model_parameters = model.get_params(deep=False)
        summaries.append({
            "family": candidate["family"],
            "model_parameters": model_parameters,
            "primary_metric": candidate["primary_metric"],
            "direction": candidate["direction"],
            "eligible": candidate["eligible"],
            "summary": candidate["summary"],
            "total_seconds": candidate["total_seconds"],
            "folds": [{
                k: fold[k] for k in (
                    "fold_index", "status", "metrics", "fit_seconds", "score_seconds", "error",
                    "best_epoch", "epochs_trained", "stop_reason", "inner_fit_row_count", "loss_curve",
                ) if k in fold
            } for fold in candidate["folds"]],
        })
    manifest = make_manifest(job, context, run, dataset, summaries, telemetry,
                             {"results.json": results_sha}, ids)
    manifest["observed_threadpools"] = payload.get("observed_threadpools", [])
    manifest_sha = _write(staging / "manifest.json", manifest)
    mlflow_store.finish_attempt(client, ids, [staging / "results.json", staging / "manifest.json"])
    staging.rename(final)
    return {"path": str(final), "manifest_sha256": manifest_sha, "results_sha256": results_sha,
            "candidates": [{"candidate_id": str(uuid4()), "family": c["family"],
                            "eligible": c["eligible"], "run_id": ids["candidates"][c["family"]]["run_id"],
                            "config_sha256": digest_bytes(canonical_bytes(c["run_config"]))}
                           for c in candidates]}
