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
        if family == "mlp":
            client.log_param(cid, "max_epochs", candidate["run_config"]["limits"]["max_epochs"])
            client.log_param(cid, "patience", 10)
            client.log_param(cid, "batch_size", 32)
            client.log_param(cid, "lr", 1e-3)
            client.log_param(cid, "hidden_dims", "64,32")
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
            if fold.get("best_epoch") is not None:
                client.log_metric(fid, "best_epoch", fold["best_epoch"])
            if fold.get("epochs_trained") is not None:
                client.log_metric(fid, "epochs_trained", fold["epochs_trained"])
            if fold.get("inner_fit_row_count") is not None:
                client.log_metric(fid, "inner_fit_row_count", fold["inner_fit_row_count"])
            if fold.get("stop_reason") is not None:
                client.log_param(fid, "stop_reason", fold["stop_reason"])
            if fold.get("loss_curve"):
                for point in fold["loss_curve"]:
                    epoch = point["epoch"] if isinstance(point, dict) else point.epoch
                    train_loss = point["train_loss"] if isinstance(point, dict) else point.train_loss
                    val_loss = point["val_loss"] if isinstance(point, dict) else point.val_loss
                    if train_loss is not None and math.isfinite(train_loss):
                        client.log_metric(fid, "train_loss", train_loss, step=epoch)
                    if val_loss is not None and math.isfinite(val_loss):
                        client.log_metric(fid, "val_loss", val_loss, step=epoch)
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
