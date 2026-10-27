"""Short, bounded SQLite transactions; no model fitting or automatic takeover.

The first claim binds the singleton lease. Worker identity, heartbeat, completion
and verified-death reconciliation use guarded transactions. A stale heartbeat
alone never frees this slot. Paths here are trusted app-owned
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

from apps.api.database.schema import SCHEMA_SQL, SCHEMA_VERSION, WORKER_PROCESSES_SQL
from apps.api.workers.monitor import process_liveness
from apps.api.workers.control import check_safe_error, check_stage, check_state
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


def _process_identity(pid: int, created: float) -> None:
    import math
    if (type(pid) is not int or pid <= 0 or type(created) not in (int, float)
            or not math.isfinite(created) or created <= 0):
        raise ValueError("Invalid process identity")


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(**{name: bool(row[name]) if name == "cancel_requested" else row[name]
                        for name in JobRecord.__dataclass_fields__})


def _audit(db: sqlite3.Connection, actor: str, entity: str, event: str) -> None:
    db.execute("INSERT INTO audit_events(occurred_at,actor,entity_id,event) VALUES(?,?,?,?)",
               (_now(), actor, entity, event))


#: Only these numeric keys may be persisted as resource telemetry. Anything else
#: (a path, an exception string, a feature value) is dropped rather than stored.
_TELEMETRY_KEYS = frozenset({
    "wall_seconds", "cpu_seconds", "normalized_cpu_percent", "logical_cores",
    "peak_rss_bytes", "minimum_available_bytes", "sample_count",
    "sample_interval_seconds", "child_pid", "child_exit_code",
})


def render_telemetry_event(telemetry: dict[str, object]) -> str:
    """Render a bounded, sorted, numbers-only audit event.

    Public so a caller can validate telemetry *before* opening the terminal
    transaction; a rejected value must never abort a publication.
    """
    if not isinstance(telemetry, dict):
        raise ValueError("Telemetry must be a mapping")
    safe: dict[str, int | float] = {}
    for key in sorted(_TELEMETRY_KEYS & set(telemetry)):
        value = telemetry[key]
        if value is None:
            continue
        if type(value) not in (int, float) or isinstance(value, bool):
            raise ValueError("Telemetry values must be numbers")
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError("Telemetry values must be finite")
        safe[key] = value
    return "job_telemetry:" + json.dumps(safe, sort_keys=True, separators=(",", ":"))


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
            if version not in (0, 1, SCHEMA_VERSION):
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
                elif version == 1:
                    db.execute(WORKER_PROCESSES_SQL)
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

    def lease_generation(self) -> int:
        """Current fencing generation; a supervisor adopts it before claiming."""
        with self._connection() as db:
            return int(db.execute("SELECT generation FROM worker_lease WHERE singleton=1").fetchone()[0])

    def cancel_requested(self, job_id: str) -> bool:
        """Cheap read of user intent; the worker decides how to act on it."""
        _text(job_id)
        with self._connection() as db:
            row = db.execute("SELECT cancel_requested FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError("Job not found")
            return bool(row[0])

    def claim_next(self, owner: WorkerIdentity, *, supervisor_process: tuple[int, float] | None = None) -> JobRecord | None:
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
            if supervisor_process is not None:
                pid, created = supervisor_process
                _process_identity(pid, created)
                db.execute("INSERT INTO worker_processes(attempt_id,supervisor_pid,supervisor_created) VALUES(?,?,?)",
                           (attempt_id, pid, created))
            _audit(db, "worker", row["job_id"], "job_claimed")
            result = _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone())
        logger.info("Job claimed: %s", result.job_id)
        return result

    def record_child(self, owner: WorkerIdentity, job_id: str, attempt_id: str,
                     pid: int, created: float) -> None:
        """Persist identity before the child's start gate is opened."""
        _process_identity(pid, created)
        with self._transaction() as db:
            row = self._owned_running_row(db, owner, job_id)
            if row["attempt_id"] != attempt_id:
                raise ConflictError("Attempt no longer owns the job")
            result = db.execute("UPDATE worker_processes SET child_pid=?,child_created=? "
                                "WHERE attempt_id=? AND child_pid IS NULL", (pid, created, attempt_id))
            if result.rowcount != 1:
                raise ConflictError("Child identity cannot be replaced")

    def reconcile_worker(self) -> bool:
        """Release only a slot whose supervisor AND registered child are dead.

        Unknown ownership (including legacy running jobs or a spawn-registration
        crash gap) blocks. Heartbeat age is deliberately irrelevant. No process
        is killed here and no new fit overlaps a still-live orphan.
        """
        with self._transaction() as db:
            lease = db.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            if lease["active_job_id"] is None:
                return True
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (lease["active_job_id"],)).fetchone()
            identity = db.execute("SELECT * FROM worker_processes WHERE attempt_id=?",
                                  (row["attempt_id"],)).fetchone()
            if identity is None or identity["child_pid"] is None:
                return False
            for prefix in ("supervisor", "child"):
                if process_liveness(identity[f"{prefix}_pid"], identity[f"{prefix}_created"]) != "dead":
                    return False
            now = _now()
            db.execute("UPDATE jobs SET state='interrupted',stage='interrupted',safe_error='owner_lost',updated_at=? "
                       "WHERE job_id=?", (now, row["job_id"]))
            db.execute("UPDATE candidates SET status='interrupted' WHERE attempt_id=?", (row["attempt_id"],))
            db.execute("UPDATE worker_lease SET generation=generation+1,owner_token=NULL,active_job_id=NULL,heartbeat_at=?",
                       (now,))
            _audit(db, "recovery", row["job_id"], "job_interrupted")
            return True

    def retry_cv(self, job_id: str, idempotency_key: str) -> JobRecord:
        """Explicit whole-CV retry; immutable experiment/splits are reused."""
        previous = self.get_job(job_id)
        if previous.kind != "cv" or previous.state not in ("failed", "interrupted", "cancelled"):
            raise ConflictError("Only unsuccessful terminal CV jobs can be retried")
        return self.enqueue_cv(previous.experiment_id, idempotency_key)

    def heartbeat(self, owner: WorkerIdentity, job_id: str) -> JobRecord:
        """Refresh the lease and job heartbeat for the verified attempt owner."""
        if not isinstance(owner, WorkerIdentity):
            raise TypeError("WorkerIdentity required")
        _text(job_id)
        with self._transaction() as db:
            row = self._owned_running_row(db, owner, job_id)
            now = _now()
            db.execute("UPDATE jobs SET heartbeat_at=?,updated_at=? WHERE job_id=?", (now, now, job_id))
            db.execute("UPDATE worker_lease SET heartbeat_at=? WHERE singleton=1 AND owner_token=?",
                       (now, owner.owner_token))
            return _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def record_progress(self, owner: WorkerIdentity, job_id: str, stage: str) -> JobRecord:
        """Persist one allowlisted stage; free text can never reach the database."""
        if not isinstance(owner, WorkerIdentity):
            raise TypeError("WorkerIdentity required")
        stage = check_stage(stage)
        _text(job_id)
        with self._transaction() as db:
            self._owned_running_row(db, owner, job_id)
            now = _now()
            db.execute("UPDATE jobs SET stage=?,updated_at=?,heartbeat_at=? WHERE job_id=?",
                       (stage, now, now, job_id))
            return _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def finish_job(self, owner: WorkerIdentity, job_id: str, attempt_id: str, *,
                   state: str, stage: str, safe_error: str | None = None,
                   telemetry: dict[str, object] | None = None) -> JobRecord:
        """Publish one terminal transition and release the singleton slot atomically.

        Only the verified attempt owner may finish. Resource telemetry is an
        allowlisted, numbers-only record appended to the audit trail; no path,
        exception text or row value can be persisted through it.
        """
        if not isinstance(owner, WorkerIdentity):
            raise TypeError("WorkerIdentity required")
        state, stage = check_state(state), check_stage(stage)
        safe_error = check_safe_error(safe_error)
        _text(job_id)
        _text(attempt_id)
        event = render_telemetry_event(telemetry) if telemetry is not None else None
        with self._transaction() as db:
            row = self._owned_running_row(db, owner, job_id)
            if row["attempt_id"] != attempt_id:
                raise ConflictError("Attempt does not own this job")
            identity = db.execute("SELECT child_pid,child_created FROM worker_processes WHERE attempt_id=?",
                                  (attempt_id,)).fetchone()
            if identity is not None and identity["child_pid"] is not None:
                if process_liveness(identity["child_pid"], identity["child_created"]) != "dead":
                    raise ConflictError("Child death must be verified before releasing its slot")
            if row["cancel_requested"] and state == "completed":
                state, stage, safe_error = "cancelled", "cancelled", "cancelled"
            now = _now()
            db.execute("""UPDATE jobs SET state=?,stage=?,safe_error=?,updated_at=?,heartbeat_at=?
                          WHERE job_id=?""", (state, stage, safe_error, now, now, job_id))
            if state != "completed":
                db.execute("UPDATE candidates SET status=? WHERE attempt_id=?", (state, attempt_id))
            db.execute("UPDATE worker_lease SET owner_token=NULL,active_job_id=NULL,heartbeat_at=? "
                       "WHERE singleton=1 AND owner_token=?", (now, owner.owner_token))
            _audit(db, "worker", job_id, f"job_{state}")
            if event is not None:
                _audit(db, "worker", job_id, event)
            result = _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        logger.info("Job %s: %s", job_id, state)
        return result

    def _owned_running_row(self, db: sqlite3.Connection, owner: WorkerIdentity, job_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError("Job not found")
        if (row["state"] != "running" or row["owner_token"] != owner.owner_token
                or row["owner_generation"] != owner.generation):
            raise ConflictError("Worker no longer owns this job")
        lease = db.execute("SELECT owner_token,active_job_id,generation FROM worker_lease WHERE singleton=1").fetchone()
        if (lease["owner_token"] != owner.owner_token or lease["active_job_id"] != job_id
                or lease["generation"] != owner.generation):
            raise ConflictError("Worker no longer holds the singleton slot")
        return row

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
