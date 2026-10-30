"""Fail-closed deletion intent and partial private-file cleanup.

Final metadata/MLflow purge and durable scope-lock tombstones are not available;
therefore this service deliberately never reports confirmed deletion.
"""

import os
from pathlib import Path
import shutil
from uuid import UUID

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError, _now


def _remove_file(path: Path) -> None:
    if path.is_symlink() or (path.exists() and path.resolve() != path.absolute()):
        raise OSError("Unsafe private file")
    path.unlink(missing_ok=True)


def _remove_directory(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or os.path.isjunction(path) or path.resolve() != path.absolute() or not path.is_dir():
        raise OSError("Unsafe private directory")
    for root, directories, files in os.walk(path, followlinks=False):
        for name in directories + files:
            child = Path(root) / name
            if child.is_symlink() or os.path.isjunction(child):
                raise OSError("Unsafe private directory entry")
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def request_deletion(repository: JobRepository, dataset_id: str) -> dict:
    """Persist an intent atomically with active-job checks before removing files."""
    try:
        if str(UUID(dataset_id)) != dataset_id:
            raise ValueError
    except ValueError:
        raise NotFoundError("Dataset not found") from None
    with repository._transaction() as db:
        row = db.execute("SELECT canonical_sha256 FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if row is None:
            raise NotFoundError("Dataset not found")
        active = db.execute("SELECT 1 FROM jobs j JOIN experiments e ON e.experiment_id=j.experiment_id "
                            "JOIN configurations c ON c.configuration_id=e.configuration_id "
                            "JOIN datasets d ON d.dataset_id=c.dataset_id "
                            "WHERE d.canonical_sha256=? AND j.state IN ('queued','running') LIMIT 1", (row[0],)).fetchone()
        if active:
            raise ConflictError("Dataset has active jobs")
        db.execute("INSERT OR IGNORE INTO deletion_intents(dataset_id,created_at) VALUES(?,?)", (dataset_id, _now()))
        attempts = [r[0] for r in db.execute(
            "SELECT j.attempt_id FROM jobs j JOIN experiments e ON e.experiment_id=j.experiment_id "
            "JOIN configurations c ON c.configuration_id=e.configuration_id "
            "WHERE c.dataset_id=? AND j.attempt_id IS NOT NULL", (dataset_id,))]
    failed = False
    home = repository.path.parent
    try:
        _remove_file(home / "uploads" / dataset_id / "source.csv")
        for attempt in attempts:
            if str(UUID(attempt)) != attempt:
                raise OSError("Unsafe attempt identity")
            _remove_directory(home / "results" / attempt)
            _remove_directory(home / "results" / ("staging-" + attempt))
            _remove_directory(home / "jobs" / attempt / "staging")
    except (OSError, ValueError):
        failed = True
    return {"dataset_id": dataset_id, "state": "deletion_pending", "file_cleanup_failed": failed,
            "disclosure": "Deletion is not complete: metadata, tracking copies and evaluation-lock protection require further cleanup."}
