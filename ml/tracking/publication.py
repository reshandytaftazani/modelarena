"""Stage complete private bundles, then atomically rename before DB commit."""

import os
from pathlib import Path
from uuid import uuid4

from ml.models.classical import make_classical
from ml.tracking import mlflow_store
from ml.tracking.manifest import canonical_bytes, digest_bytes, make_manifest


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
    results_sha = _write(staging / "results.json", payload)
    candidates = payload["candidates"]
    ids, client = mlflow_store.track_attempt(home, job, candidates)
    summaries = []
    for candidate in candidates:
        model = make_classical(context.training.config.task, candidate["family"], candidate["model_seed"])
        summaries.append({"family": candidate["family"], "model_parameters": model.get_params(deep=False),
                          "primary_metric": candidate["primary_metric"], "direction": candidate["direction"],
                          "eligible": candidate["eligible"], "summary": candidate["summary"],
                          "total_seconds": candidate["total_seconds"],
                          "folds": [{k: fold[k] for k in ("fold_index", "status", "metrics", "fit_seconds", "score_seconds", "error")}
                                    for fold in candidate["folds"]]})
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
