"""Identify terminal-attempt orphans without deleting private evidence."""

from pathlib import Path


def reconcile_orphans(repository) -> dict:
    """Mark MLflow runs and list unreferenced bundles belonging to terminal jobs.

    Active or unknown attempts are never touched. Marking is idempotent; physical
    deletion is deferred to the explicitly acknowledged retention workflow.
    """
    with repository._connection() as db:
        jobs = {row["attempt_id"]: row["state"] for row in db.execute(
            "SELECT attempt_id,state FROM jobs WHERE attempt_id IS NOT NULL")}
        references = {str(Path(row[0]).resolve()) for row in db.execute(
            "SELECT DISTINCT bundle_path FROM candidates WHERE bundle_path IS NOT NULL")}
    terminal = {attempt for attempt, state in jobs.items() if state in ("failed", "interrupted", "cancelled")}
    root = repository.path.parent / "results"
    bundles = []
    if root.exists():
        for path in root.iterdir():
            attempt = path.name.removeprefix("staging-")
            if path.is_dir() and attempt in terminal and str(path.resolve()) not in references:
                bundles.append(str(path))
    marked = []
    mlflow_db = repository.path.parent / "mlflow.sqlite3"
    if mlflow_db.exists() and terminal:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri="sqlite:///" + mlflow_db.as_posix())
        experiment_token = None
        while True:
            experiments = client.search_experiments(max_results=1000, page_token=experiment_token)
            for experiment in experiments:
                run_token = None
                while True:
                    runs = client.search_runs([experiment.experiment_id], max_results=1000, page_token=run_token)
                    for run in runs:
                        if run.data.tags.get("modelarena.attempt_id") in terminal:
                            client.set_tag(run.info.run_id, "modelarena.orphaned", "true")
                            if run.info.status == "RUNNING":
                                client.set_terminated(run.info.run_id, status="KILLED")
                            marked.append(run.info.run_id)
                    run_token = runs.token
                    if not run_token:
                        break
            experiment_token = experiments.token
            if not experiment_token:
                break
    return {"bundle_paths": sorted(bundles), "mlflow_run_ids": sorted(marked)}
