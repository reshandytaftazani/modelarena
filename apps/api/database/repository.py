"""Short, bounded SQLite transactions; no model fitting or automatic takeover.

The first claim binds the singleton lease. Process identity, lease acquisition,
heartbeats, completion and restart reconciliation belong to the worker milestone.
A stale heartbeat alone never frees this slot. Paths here are trusted app-owned
local paths, not transport input. Connections are per operation and always close.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
from typing import Iterator
from uuid import uuid4

from apps.api.database.schema import SCHEMA_SQL, SCHEMA_VERSION
from ml.contracts import DatasetConfig, DatasetProfile, RunConfig, SplitManifest, TaskType
from ml.data.csv_io import ValidatedDataset
from ml.data.splits import _config_digest, manifest_json, verify_manifest
from ml.evaluation.metrics import metric_direction

logger = logging.getLogger(__name__)


class ConflictError(ValueError):
    """409-equivalent immutable identity/idempotency conflict."""


class NotFoundError(LookupError):
    """An opaque persisted identity does not exist."""


class QueueFullError(ConflictError):
    """The bounded waiting queue is full."""


class StorageBusyError(RuntimeError):
    """All bounded BEGIN attempts encountered a locked database."""


@dataclass(frozen=True, slots=True)
class WorkerIdentity:
    owner_token: str
    generation: int

    def __post_init__(self) -> None:
        _text(self.owner_token)
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("Worker generation must be positive")


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    experiment_id: str
    kind: str
    state: str
    owner_token: str | None
    owner_generation: int | None
    attempt_id: str | None
    cancel_requested: bool
    stage: str
    created_at: str
    updated_at: str
    heartbeat_at: str | None
    safe_error: str | None


def _text(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("Expected nonempty text of at most 200 characters")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(**{name: bool(row[name]) if name == "cancel_requested" else row[name]
                        for name in JobRecord.__dataclass_fields__})


def _audit(db: sqlite3.Connection, actor: str, entity: str, event: str) -> None:
    db.execute("INSERT INTO audit_events(occurred_at,actor,entity_id,event) VALUES(?,?,?,?)",
               (_now(), actor, entity, event))


class JobRepository:
    def __init__(self, path: Path, *, busy_timeout_ms: int = 5000, begin_retries: int = 2):
        if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 5000:
            raise ValueError("Busy timeout must be between 1 and 5000 ms")
        if type(begin_retries) is not int or not 0 <= begin_retries <= 2:
            raise ValueError("At most two bounded retries allowed")
        self.path = Path(path).absolute()
        self.busy_timeout_ms = busy_timeout_ms
        self.begin_retries = begin_retries
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def _begin(self, db: sqlite3.Connection) -> None:
        for attempt in range(self.begin_retries + 1):
            try:
                db.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                if getattr(exc, "sqlite_errorcode", None) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    raise
                if attempt == self.begin_retries:
                    raise StorageBusyError("Metadata store is busy; retry later") from None

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as db:
            self._begin(db)
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def _initialize(self) -> None:
        with self._connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError("Unsupported metadata schema version")
            if version == SCHEMA_VERSION:
                return
            db.execute("PRAGMA journal_mode=WAL")
            self._begin(db)
            try:
                # Recheck after acquiring the lock: another initializer may win.
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    # executescript implicitly commits; execute complete statements
                    # individually so DDL, triggers and version share one commit.
                    statement = ""
                    for line in SCHEMA_SQL.splitlines():
                        statement += line + "\n"
                        if sqlite3.complete_statement(statement):
                            db.execute(statement)
                            statement = ""
                    db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif version != SCHEMA_VERSION:
                    raise ValueError("Unsupported metadata schema version")
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def register_experiment(self, data: ValidatedDataset, config: DatasetConfig,
                            split: SplitManifest, run: RunConfig, profile: DatasetProfile) -> str:
        """Persist validated service metadata, never feature/target records.

        Dataset/config feasibility and profiling are completed before entering
        this transaction. No private upload path exists until upload services.
        """
        if (not verify_manifest(split) or split.dataset_sha256 != data.canonical_sha256
                or split.row_count != data.row_count or split.config_sha256 != _config_digest(config)
                or run.split_seed != split.split_seed or profile.partition != "training"
                or profile.row_count != len(split.train_row_ids)):
            raise ValueError("Metadata does not match the frozen training split")
        primary = run.primary_metric or ("f1" if config.task == TaskType.BINARY_CLASSIFICATION else "rmse")
        direction = metric_direction(config.task, primary)
        config_json, run_json, profile_json = _json(asdict(config)), _json(asdict(run)), _json(asdict(profile))
        partition_hash = _digest(_json([split.split_seed, split.train_row_ids, split.holdout_row_ids]))
        payload_hash = _digest(_json([split.manifest_sha256, config_json, run_json, primary]))
        now = _now()
        with self._transaction() as db:
            dataset = db.execute("SELECT dataset_id FROM datasets WHERE raw_sha256=?", (data.raw_sha256,)).fetchone()
            dataset_id = dataset[0] if dataset else str(uuid4())
            if not dataset:
                db.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?)", (
                    dataset_id, data.raw_sha256, data.canonical_sha256, data.parse_policy_version,
                    data.byte_count, data.row_count, len(data.columns), now))
            scope = db.execute("SELECT scope_id,partition_sha256 FROM evaluation_scopes WHERE canonical_sha256=? AND target=? AND task=?",
                               (data.canonical_sha256, config.target, str(config.task))).fetchone()
            if scope and scope[1] != partition_hash:
                raise ConflictError("Evaluation scope already has a different frozen split")
            scope_id = scope[0] if scope else str(uuid4())
            if not scope:
                db.execute("INSERT INTO evaluation_scopes VALUES(?,?,?,?,?,?)",
                           (scope_id, data.canonical_sha256, config.target, str(config.task), partition_hash, now))
            saved_split = db.execute("SELECT split_id FROM splits WHERE manifest_sha256=?", (split.manifest_sha256,)).fetchone()
            split_id = saved_split[0] if saved_split else str(uuid4())
            if not saved_split:
                db.execute("INSERT INTO splits VALUES(?,?,?,?,?)", (split_id, scope_id, split.manifest_sha256, manifest_json(split), now))
            configuration = db.execute("SELECT configuration_id FROM configurations WHERE dataset_id=? AND split_id=? AND config_sha256=?",
                                       (dataset_id, split_id, split.config_sha256)).fetchone()
            configuration_id = configuration[0] if configuration else str(uuid4())
            if not configuration:
                db.execute("INSERT INTO configurations VALUES(?,?,?,?,?,?,?,?)", (
                    configuration_id, dataset_id, scope_id, split_id, split.config_sha256, config_json, profile_json, now,
                ))
            existing = db.execute("SELECT experiment_id FROM experiments WHERE configuration_id=? AND payload_sha256=?",
                                  (configuration_id, payload_hash)).fetchone()
            if existing:
                return existing[0]
            experiment_id = str(uuid4())
            db.execute("INSERT INTO experiments VALUES(?,?,?,?,?,?,?,?)", (
                experiment_id, configuration_id, run_json, payload_hash, primary, direction, run.limits.queue_size, now,
            ))
            _audit(db, "service", experiment_id, "experiment_registered")
        logger.info("Experiment registered: %s", experiment_id)
        return experiment_id

    def enqueue_cv(self, experiment_id: str, idempotency_key: str) -> JobRecord:
        _text(experiment_id)
        _text(idempotency_key)
        with self._transaction() as db:
            experiment = db.execute("SELECT payload_sha256,queue_limit FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
            if experiment is None:
                raise NotFoundError("Experiment not found")
            payload = _digest(_json(["cv", experiment_id, experiment[0]]))
            existing = db.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing:
                if existing["payload_sha256"] != payload:
                    raise ConflictError("Idempotency key was used with a different payload")
                return _record(existing)
            if db.execute("SELECT count(*) FROM jobs WHERE state='queued'").fetchone()[0] >= experiment[1]:
                raise QueueFullError("Waiting queue is full")
            job_id, now = str(uuid4()), _now()
            db.execute("""INSERT INTO jobs(job_id,experiment_id,kind,idempotency_key,payload_sha256,state,created_at,updated_at)
                          VALUES(?,?,'cv',?,?,'queued',?,?)""", (job_id, experiment_id, idempotency_key, payload, now, now))
            _audit(db, "service", job_id, "job_enqueued")
            result = _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        logger.info("Job queued: %s", result.job_id)
        return result

    def get_job(self, job_id: str) -> JobRecord:
        _text(job_id)
        with self._connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError("Job not found")
            return _record(row)

    def claim_next(self, owner: WorkerIdentity) -> JobRecord | None:
        if not isinstance(owner, WorkerIdentity):
            raise TypeError("WorkerIdentity required")
        with self._transaction() as db:
            lease = db.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            if (lease["generation"] != owner.generation or lease["active_job_id"] is not None
                    or lease["owner_token"] not in (None, owner.owner_token)):
                return None
            if db.execute("SELECT 1 FROM jobs WHERE state='running'").fetchone():
                return None
            row = db.execute("SELECT * FROM jobs WHERE state='queued' AND cancel_requested=0 ORDER BY sequence LIMIT 1").fetchone()
            if row is None:
                return None
            now, attempt_id = _now(), str(uuid4())
            db.execute("""UPDATE jobs SET state='running',owner_token=?,owner_generation=?,attempt_id=?,
                          stage='claimed',updated_at=?,heartbeat_at=? WHERE job_id=? AND state='queued'""",
                       (owner.owner_token, owner.generation, attempt_id, now, now, row["job_id"]))
            db.execute("UPDATE worker_lease SET owner_token=?,active_job_id=?,heartbeat_at=? WHERE singleton=1",
                       (owner.owner_token, row["job_id"], now))
            _audit(db, "worker", row["job_id"], "job_claimed")
            result = _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone())
        logger.info("Job claimed: %s", result.job_id)
        return result

    def request_cancel(self, job_id: str) -> JobRecord:
        _text(job_id)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError("Job not found")
            if row["state"] not in ("queued", "running") or row["cancel_requested"]:
                return _record(row)
            state = "cancelled" if row["state"] == "queued" else "running"
            db.execute("UPDATE jobs SET cancel_requested=1,state=?,stage=?,updated_at=? WHERE job_id=? AND state=?",
                       (state, "cancelled" if state == "cancelled" else row["stage"], _now(), job_id, row["state"]))
            _audit(db, "service", job_id, "job_cancel_requested")
            result = _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        logger.info("Cancellation requested: %s", result.job_id)
        return result
