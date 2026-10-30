"""Explicit local MLflow runs; no global tracking URI or autologging."""

from pathlib import Path
import math

from mlflow.tracking import MlflowClient

from ml.tracking.manifest import file_digest


def track_attempt(home: Path, job, candidates: list[dict]) -> tuple[dict, MlflowClient]:
    home = Path(home).resolve()
    artifacts = home / "mlflow-artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri="sqlite:///" + (home / "mlflow.sqlite3").as_posix())
    name = "modelarena-" + job.experiment_id
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(name, artifact_location=artifacts.as_uri())
    common = {"modelarena.experiment_id": job.experiment_id, "modelarena.job_id": job.job_id,
              "modelarena.attempt_id": job.attempt_id}

    def create(kind, parent=None, extra=None):
        tags = {**common, "modelarena.kind": kind, **(extra or {})}
        if parent:
            tags["mlflow.parentRunId"] = parent
        return client.create_run(experiment_id, tags=tags).info.run_id

    parent = create("attempt")
    ids = {"parent": parent, "candidates": {}}
    for candidate in candidates:
        family = candidate["family"]
        cid = create("candidate", parent, {"modelarena.family": family})
        ids["candidates"][family] = {"run_id": cid, "folds": {}}
        client.log_param(cid, "model_seed", candidate["model_seed"])
        client.log_param(cid, "primary_metric", candidate["primary_metric"])
        for summary in candidate["summary"]:
            for key in ("mean", "std"):
                value = summary[key]
                if value is not None and math.isfinite(value):
                    client.log_metric(cid, summary["name"] + "_" + key, value)
        for fold in candidate["folds"]:
            fid = create("fold", cid, {"modelarena.family": family, "modelarena.fold_index": str(fold["fold_index"])})
            ids["candidates"][family]["folds"][str(fold["fold_index"])] = fid
            client.log_param(fid, "status", fold["status"])
            client.log_metric(fid, "fit_seconds", fold["fit_seconds"])
            client.log_metric(fid, "score_seconds", fold["score_seconds"])
            if fold["metrics"] is not None:
                for metric in fold["metrics"]["metrics"]:
                    value = metric["value"]
                    if value is not None and math.isfinite(value):
                        client.log_metric(fid, metric["name"], value)
            client.set_terminated(fid, status="FINISHED")
        client.set_terminated(cid, status="FINISHED")
    return ids, client


def finish_attempt(client: MlflowClient, run_ids: dict, files: list[Path]) -> None:
    parent = run_ids["parent"]
    for path in files:
        client.log_artifact(parent, str(path))
        downloaded = Path(client.download_artifacts(parent, path.name))
        if file_digest(downloaded) != file_digest(path):
            raise ValueError("MLflow artifact checksum mismatch")
    client.set_terminated(parent, status="FINISHED")
    if client.get_run(parent).info.status != "FINISHED":
        raise ValueError("MLflow parent did not finish")
