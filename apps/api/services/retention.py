"""Fail-closed deletion intent and partial private-file cleanup.

Final metadata/MLflow purge and finalization integration are not available;
therefore this service deliberately never reports confirmed deletion.
"""

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
from uuid import UUID

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError, _now


def _key(repository: JobRepository, *, create: bool) -> bytes | None:
    path = repository.path.parent / "scope-tombstone.key"
    if not path.exists() and not path.is_symlink():
        with repository._connection() as db:
            existing = db.execute("SELECT 1 FROM scope_tombstones LIMIT 1").fetchone()
        if existing:
            raise RuntimeError("Scope protection key is unavailable")
        if not create:
            return None
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
    if path.is_symlink() or path.resolve() != path.absolute():
        raise RuntimeError("Scope protection key is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise RuntimeError("Scope protection key is unsafe")
        key = stream.read(33)
    if len(key) != 32:
        raise RuntimeError("Scope protection key is invalid")
    return key


def _scope_hmac(key: bytes, canonical_sha256: str, target: str, task: str) -> str:
    if (len(canonical_sha256) != 64 or any(c not in "0123456789abcdef" for c in canonical_sha256)
            or not isinstance(target, str) or not target or task not in ("regression", "binary_classification")):
        raise ValueError("Invalid scope identity")
    encoded = json.dumps(["modelarena-scope-v1", canonical_sha256, target, task],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def find_tombstone(repository: JobRepository, canonical_sha256: str, target: str, task: str, *, db=None) -> str | None:
    """Look up a deleted scope without storing its dataset or target identity."""
    key = _key(repository, create=False)
    if key is None:
        return None
    digest = _scope_hmac(key, canonical_sha256, target, task)
    if db is not None:
        row = db.execute("SELECT state FROM scope_tombstones WHERE scope_hmac=? AND version=1", (digest,)).fetchone()
        return None if row is None else row[0]
    with repository._connection() as db:
        row = db.execute("SELECT state FROM scope_tombstones WHERE scope_hmac=? AND version=1", (digest,)).fetchone()
    return None if row is None else row[0]


def _save_tombstones(repository: JobRepository, db, canonical_sha256: str) -> None:
    scopes = list(db.execute("SELECT s.canonical_sha256,s.target,s.task,m.state FROM evaluation_scopes s "
                             "JOIN scope_lock_markers m ON m.scope_id=s.scope_id WHERE s.canonical_sha256=?",
                             (canonical_sha256,)))
    if not scopes:
        return
    key = _key(repository, create=True)
    assert key is not None
    for scope in scopes:
        digest = _scope_hmac(key, scope[0], scope[1], scope[2])
        db.execute("INSERT INTO scope_tombstones(scope_hmac,version,state) VALUES(?,1,?) "
                   "ON CONFLICT(scope_hmac) DO UPDATE SET state=CASE WHEN state='consumed' OR excluded.state='consumed' "
                   "THEN 'consumed' ELSE 'reserved' END", (digest, scope[3]))


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


def deletion_disclosure(repository: JobRepository, dataset_id: str) -> dict:
    with repository._connection() as db:
        row = db.execute("SELECT canonical_sha256 FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if row is None:
            raise NotFoundError("Dataset not found")
        experiments = db.execute("SELECT count(*) FROM experiments e JOIN configurations c "
                                 "ON e.configuration_id=c.configuration_id WHERE c.dataset_id=?", (dataset_id,)).fetchone()[0]
        active = db.execute("SELECT count(*) FROM jobs j JOIN experiments e ON e.experiment_id=j.experiment_id "
                            "JOIN configurations c ON c.configuration_id=e.configuration_id JOIN datasets d ON d.dataset_id=c.dataset_id "
                            "WHERE d.canonical_sha256=? AND j.state IN ('queued','running')", (row[0],)).fetchone()[0]
        pending = db.execute("SELECT 1 FROM deletion_intents WHERE dataset_id=?", (dataset_id,)).fetchone() is not None
    return {"dataset_id": dataset_id, "linked_experiments": experiments, "active_jobs": active,
            "deletion_pending": pending, "confirmed_deletion_supported": False,
            "disclosure": ("A request removes known private uploads, partitions, split copies, results and reports, "
                           "including files shared by equivalent scopes. Metadata and MLflow copies remain. "
                           "Reserved or consumed scope markers produce a keyed tombstone; finalization integration is pending. "
                           "Deletion cannot yet be confirmed and does not cover backups or downloaded files.")}


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
        _save_tombstones(repository, db, row[0])
        db.execute("INSERT OR IGNORE INTO deletion_intents(dataset_id,created_at) VALUES(?,?)", (dataset_id, _now()))
        attempts = [r[0] for r in db.execute(
            "SELECT j.attempt_id FROM jobs j JOIN experiments e ON e.experiment_id=j.experiment_id "
            "JOIN configurations c ON c.configuration_id=e.configuration_id "
            "WHERE c.dataset_id=? AND j.attempt_id IS NOT NULL", (dataset_id,))]
        linked = list(db.execute("SELECT DISTINCT c.scope_id,c.split_id,e.experiment_id FROM configurations c "
                                 "LEFT JOIN experiments e ON e.configuration_id=c.configuration_id "
                                 "WHERE c.dataset_id=?", (dataset_id,)))
        shared_scopes = [r[0] for r in db.execute(
            "SELECT scope_id FROM evaluation_scopes WHERE canonical_sha256=?", (row[0],))]
    failed = False
    home = repository.path.parent
    try:
        _remove_file(home / "uploads" / dataset_id / "source.csv")
        for scope_id in shared_scopes:
            if str(UUID(scope_id)) != scope_id:
                raise OSError("Unsafe linked identity")
            _remove_directory(home / "datasets" / scope_id)
        for scope_id, split_id, experiment_id in linked:
            for value in (scope_id, split_id, experiment_id):
                if value is not None and str(UUID(value)) != value:
                    raise OSError("Unsafe linked identity")
            _remove_file(home / "splits" / (split_id + ".json"))
            _remove_directory(home / "final" / scope_id)
            if experiment_id is not None:
                _remove_directory(home / "reports" / experiment_id)
                _remove_file(home / "reports" / (experiment_id + ".zip"))
        for attempt in attempts:
            if str(UUID(attempt)) != attempt:
                raise OSError("Unsafe attempt identity")
            _remove_directory(home / "results" / attempt)
            _remove_directory(home / "results" / ("staging-" + attempt))
            _remove_directory(home / "jobs" / attempt / "staging")
    except (OSError, ValueError):
        failed = True
    return {"dataset_id": dataset_id, "state": "deletion_pending", "file_cleanup_failed": failed,
            "disclosure": "Deletion is not complete: metadata and tracking copies remain. Shared scope files may be removed; finalization lock integration is pending."}
