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

import secrets
from apps.api.database.schema import (DELETION_INTENTS_SQL, EVALUATION_AUTH_SQL, FINALIZATIONS_SQL, JOB_PROGRESS_SQL, SCHEMA_SQL, SCHEMA_VERSION,
                                      SCOPE_LOCK_MARKERS_SQL, SCOPE_TOMBSTONES_SQL,
                                      WORKER_PRESENCE_SQL, WORKER_PROCESSES_SQL)
from apps.api.workers.monitor import process_liveness
from apps.api.workers.control import check_safe_error, check_stage, check_state
from ml.contracts import DatasetConfig, DatasetProfile, FrozenSelection, ResourceLimits, RunConfig, SplitManifest, TaskType, TrainingContext
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
            if version not in (0, 1, 2, 3, 4, 5, 6, SCHEMA_VERSION):
                raise ValueError("Unsupported metadata schema version")
            if version == SCHEMA_VERSION:
                return
            db.execute("PRAGMA journal_mode=WAL")
            self._begin(db)
            try:
                # Recheck after acquiring the lock: another initializer may win.
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    statement = ""
                    for line in SCHEMA_SQL.splitlines():
                        statement += line + "\n"
                        if sqlite3.complete_statement(statement):
                            db.execute(statement)
                            statement = ""
                    db.execute(EVALUATION_AUTH_SQL)
                    db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif version == 5:
                    db.execute(EVALUATION_AUTH_SQL)
                    cols = {row[1] for row in db.execute("PRAGMA table_info(finalizations)").fetchall()}
                    if "evidence_sha256" not in cols:
                        db.execute("ALTER TABLE finalizations ADD COLUMN evidence_sha256 TEXT")
                    if "epoch_budget" not in cols:
                        db.execute("ALTER TABLE finalizations ADD COLUMN epoch_budget INTEGER")
                    if "plan_json" not in cols:
                        db.execute("ALTER TABLE finalizations ADD COLUMN plan_json TEXT")
                    db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif version in (1, 2, 3, 4):
                    if version < 2:
                        db.execute(WORKER_PROCESSES_SQL)
                    if version < 3:
                        db.execute(DELETION_INTENTS_SQL)
                    if version < 4:
                        db.execute(WORKER_PRESENCE_SQL)
                        db.execute(SCOPE_LOCK_MARKERS_SQL)
                        db.execute(SCOPE_TOMBSTONES_SQL)
                        db.execute(JOB_PROGRESS_SQL)
                    statement = ""
                    for line in FINALIZATIONS_SQL.splitlines():
                        statement += line + "\n"
                        if sqlite3.complete_statement(statement):
                            db.execute(statement)
                            statement = ""
                    db.execute("PRAGMA foreign_keys=OFF")
                    db.execute("""CREATE TABLE IF NOT EXISTS jobs_v5 (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL UNIQUE,
                        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                        kind TEXT NOT NULL CHECK(kind IN ('cv', 'finalization')),
                        idempotency_key TEXT NOT NULL UNIQUE,
                        payload_sha256 TEXT NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('queued','running','completed','failed','interrupted','cancelled')),
                        owner_token TEXT,
                        owner_generation INTEGER,
                        attempt_id TEXT UNIQUE,
                        cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
                        stage TEXT NOT NULL DEFAULT 'queued',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        heartbeat_at TEXT,
                        safe_error TEXT,
                        CHECK(state != 'running' OR
                              (owner_token IS NOT NULL AND owner_generation > 0 AND attempt_id IS NOT NULL))
                    )""")
                    db.execute("INSERT INTO jobs_v5 SELECT * FROM jobs")
                    db.execute("DROP TABLE jobs")
                    db.execute("ALTER TABLE jobs_v5 RENAME TO jobs")
                    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_running_job ON jobs((1)) WHERE state='running'")
                    db.execute("CREATE INDEX IF NOT EXISTS queued_jobs ON jobs(state, sequence)")
                    db.execute("PRAGMA foreign_keys=ON")
                    db.execute(EVALUATION_AUTH_SQL)
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
            if db.execute("SELECT 1 FROM deletion_intents i JOIN datasets d ON d.dataset_id=i.dataset_id "
                          "WHERE d.canonical_sha256=?", (data.canonical_sha256,)).fetchone():
                raise ConflictError("Dataset deletion is pending")
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

    def verify_classical_input(self, experiment_id: str, context: TrainingContext, run: RunConfig) -> None:
        """Match trusted, already prepared training input to immutable job metadata.

        This is not a source-content verifier: callers must obtain the context
        from the trusted prepare_experiment/training_from_manifest boundary.
        """
        if not isinstance(context, TrainingContext) or not isinstance(run, RunConfig):
            raise ValueError("Trusted training context and run required")
        with self._connection() as db:
            row = db.execute(
                "SELECT e.run_json, c.config_json, s.manifest_json FROM experiments e "
                "JOIN configurations c ON c.configuration_id=e.configuration_id "
                "JOIN splits s ON s.split_id=c.split_id WHERE e.experiment_id=?", (experiment_id,),
            ).fetchone()
        if (row is None or row[0] != _json(asdict(run))
                or row[1] != _json(asdict(context.training.config))
                or row[2] != manifest_json(context.split)):
            raise ConflictError("Classical input does not match frozen experiment")

    def tracking_dataset(self, experiment_id: str) -> dict:
        """Return stored input identities without exposing any dataset rows."""
        with self._connection() as db:
            row = db.execute("SELECT d.raw_sha256,d.canonical_sha256,d.parse_policy_version "
                             "FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
                             "JOIN datasets d ON d.dataset_id=c.dataset_id WHERE e.experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            raise NotFoundError("Experiment not found")
        return dict(row)

    def read_published_results(self, job_id: str) -> dict:
        """Read only committed bundles, checking both manifest and contents each time."""
        from ml.tracking.manifest import file_digest
        with self._connection() as db:
            row = db.execute("SELECT c.bundle_path,c.bundle_sha256 FROM candidates c "
                             "JOIN jobs j ON j.job_id=c.job_id WHERE j.job_id=? AND j.state='completed' LIMIT 1", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError("No published results")
        folder = Path(row["bundle_path"]).resolve()
        if folder.parent != (self.path.parent / "results").resolve():
            raise ValueError("Invalid bundle location")
        if file_digest(folder / "manifest.json") != row["bundle_sha256"]:
            raise ValueError("Manifest checksum mismatch")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if file_digest(folder / "results.json") != manifest["artifact_sha256"]["results.json"]:
            raise ValueError("Results checksum mismatch")
        payload = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        return {**payload, "manifest": manifest}

    def enqueue_cv(self, experiment_id: str, idempotency_key: str) -> JobRecord:
        _text(experiment_id)
        _text(idempotency_key)
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
                          "JOIN datasets source ON source.dataset_id=c.dataset_id JOIN datasets d ON d.canonical_sha256=source.canonical_sha256 "
                          "JOIN deletion_intents i ON i.dataset_id=d.dataset_id WHERE e.experiment_id=?",
                          (experiment_id,)).fetchone():
                raise ConflictError("Dataset deletion is pending")
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

    def claim_next(self, owner: WorkerIdentity, *, supervisor_process: tuple[int, float] | None = None, kind: str | None = None) -> JobRecord | None:
        if not isinstance(owner, WorkerIdentity):
            raise TypeError("WorkerIdentity required")
        with self._transaction() as db:
            lease = db.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            if (lease["generation"] != owner.generation or lease["active_job_id"] is not None
                    or lease["owner_token"] not in (None, owner.owner_token)):
                return None
            if db.execute("SELECT 1 FROM jobs WHERE state='running'").fetchone():
                return None
            if kind is not None:
                row = db.execute("SELECT * FROM jobs WHERE state='queued' AND kind=? AND cancel_requested=0 ORDER BY sequence LIMIT 1", (kind,)).fetchone()
            else:
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
            if row["kind"] == "finalization":
                fin = db.execute("SELECT * FROM finalizations WHERE refit_job_id=?", (row["job_id"],)).fetchone()
                if fin and fin["evaluation_started_at"] is not None:
                    recovered = not row["cancel_requested"] and self._try_recover_staged_final_bundle(db, fin, row["attempt_id"], now)
                    if not recovered:
                        db.execute("UPDATE finalizations SET state='locked_failed',updated_at=? "
                                   "WHERE refit_job_id=? AND evaluation_started_at IS NOT NULL AND state='evaluating'", (now, row["job_id"]))
                        db.execute("UPDATE jobs SET state='interrupted',stage='interrupted',safe_error='owner_lost',updated_at=? "
                                   "WHERE job_id=?", (now, row["job_id"]))
                        _audit(db, "recovery", fin["finalization_id"], "finalization_locked_failed")
                elif fin:
                    db.execute("UPDATE finalizations SET state='retryable_pre_evaluation',updated_at=? "
                               "WHERE refit_job_id=? AND evaluation_started_at IS NULL AND state='refitting'", (now, row["job_id"]))
                    db.execute("UPDATE jobs SET state='interrupted',stage='interrupted',safe_error='owner_lost',updated_at=? "
                               "WHERE job_id=?", (now, row["job_id"]))
                    _audit(db, "recovery", fin["finalization_id"], "finalization_interrupted_pre_evaluation")
            else:
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

    def record_work_progress(self, owner: WorkerIdentity, job_id: str, family: str, fold_index: int,
                             finished_units: int, total_units: int) -> None:
        """Persist bounded fold work counts only for the current fenced owner."""
        if family not in {"dummy", "logistic", "ridge", "random_forest", "hist_gradient_boosting", "mlp"}:
            raise ValueError("Unsupported candidate family")
        if (type(fold_index) is not int or type(finished_units) is not int or type(total_units) is not int
                or total_units not in (8, 12, 16, 20, 10, 15, 25)
                or not 0 <= fold_index < (5 if total_units == 20 else
                                          total_units // 5 if total_units in (10, 15, 25) else total_units // 4)
                or not 0 <= finished_units <= total_units):
            raise ValueError("Invalid progress units")
        with self._transaction() as db:
            self._owned_running_row(db, owner, job_id)
            old = db.execute("SELECT finished_units,total_units FROM job_progress WHERE job_id=?", (job_id,)).fetchone()
            if old is not None and (finished_units < old[0] or total_units != old[1]):
                raise ValueError("Progress cannot go backwards")
            db.execute("INSERT INTO job_progress VALUES(?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET "
                       "family=excluded.family,fold_index=excluded.fold_index,finished_units=excluded.finished_units, "
                       "total_units=excluded.total_units", (job_id, family, fold_index, finished_units, total_units))

    def get_work_progress(self, job_id: str) -> dict | None:
        with self._connection() as db:
            row = db.execute("SELECT family,fold_index,finished_units,total_units FROM job_progress WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row is not None else None

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
                   telemetry: dict[str, object] | None = None,
                   publication: dict | None = None) -> JobRecord:
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
            # CV/probe completion cannot publish a final result. Until a fenced
            # final-bundle transaction exists, never report finalization success
            # through this generic job transition, even with a CV publication.
            if row["kind"] == "finalization" and state == "completed":
                raise ConflictError("Finalization requires a dedicated result publication")
            if state == "completed" and publication is not None:
                from ml.tracking.manifest import file_digest
                folder = Path(publication["path"]).resolve()
                if (folder.parent != (self.path.parent / "results").resolve() or folder.name != attempt_id
                        or file_digest(folder / "manifest.json") != publication["manifest_sha256"]
                        or file_digest(folder / "results.json") != publication["results_sha256"]):
                    raise ValueError("Bundle checksum or identity mismatch")
                for candidate in publication["candidates"]:
                    db.execute("INSERT INTO candidates(candidate_id,experiment_id,job_id,attempt_id,family,status,config_sha256,bundle_path,bundle_sha256,mlflow_run_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (candidate["candidate_id"], row["experiment_id"], job_id, attempt_id,
                                candidate["family"], "completed" if candidate["eligible"] else "failed",
                                candidate["config_sha256"], str(folder), publication["manifest_sha256"], candidate["run_id"]))
            now = _now()
            db.execute("""UPDATE jobs SET state=?,stage=?,safe_error=?,updated_at=?,heartbeat_at=?
                          WHERE job_id=?""", (state, stage, safe_error, now, now, job_id))
            if state != "completed":
                db.execute("UPDATE candidates SET status=? WHERE attempt_id=?", (state, attempt_id))
                db.execute("UPDATE finalizations SET state='retryable_pre_evaluation',updated_at=? "
                           "WHERE refit_job_id=? AND evaluation_started_at IS NULL AND state='refitting'", (now, job_id))
                db.execute("UPDATE finalizations SET state='locked_failed',updated_at=? "
                           "WHERE refit_job_id=? AND evaluation_started_at IS NOT NULL AND state='evaluating'", (now, job_id))
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
            if state == "cancelled":
                db.execute("UPDATE finalizations SET state='retryable_pre_evaluation',updated_at=? "
                           "WHERE refit_job_id=? AND evaluation_started_at IS NULL AND state='refitting'", (_now(), job_id))
            _audit(db, "service", job_id, "job_cancel_requested")
            result = _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        logger.info("Cancellation requested: %s", result.job_id)
        return result

    def reserve_finalization(self, scope_id: str, candidate_id: str, digest: str, *, experiment_id: str) -> FrozenSelection:
        import math
        _text(scope_id)
        _text(candidate_id)
        _text(digest)
        _text(experiment_id)
        with self._transaction() as db:
            belongs = db.execute("SELECT 1 FROM candidates c JOIN experiments e ON e.experiment_id=c.experiment_id "
                                 "JOIN configurations cfg ON cfg.configuration_id=e.configuration_id "
                                 "WHERE c.candidate_id=? AND e.experiment_id=? AND cfg.scope_id=?",
                                 (candidate_id, experiment_id, scope_id)).fetchone()
            if not belongs:
                raise ConflictError("Candidate does not belong to the requested experiment and scope")
            scope = db.execute(
                "SELECT scope_id, canonical_sha256, target, task FROM evaluation_scopes WHERE scope_id=?",
                (scope_id,),
            ).fetchone()
            if scope is None:
                raise NotFoundError("Evaluation scope not found")

            # Serialize this lookup with deletion's tombstone write. A reserved
            # deleted scope is locked too: its original selection is gone.
            from apps.api.services.retention import find_tombstone
            if find_tombstone(self, scope["canonical_sha256"], scope["target"], scope["task"], db=db) is not None:
                raise ConflictError("Deleted holdout scope is locked")

            if db.execute("SELECT 1 FROM deletion_intents i JOIN datasets d ON d.dataset_id=i.dataset_id WHERE d.canonical_sha256=?",
                          (scope["canonical_sha256"],)).fetchone():
                raise ConflictError("Dataset deletion is pending")

            marker = db.execute("SELECT state FROM scope_lock_markers WHERE scope_id=?", (scope_id,)).fetchone()
            existing = db.execute("SELECT * FROM finalizations WHERE scope_id=?", (scope_id,)).fetchone()
            if marker and marker["state"] == "consumed":
                if not (existing is not None and existing["state"] == "evaluated"
                        and existing["candidate_id"] == candidate_id and existing["candidate_digest"] == digest):
                    raise ConflictError("Holdout already consumed")
            if marker and existing is None:
                # The marker proves a reservation existed but its selection is
                # unavailable. Never reinterpret this as an unselected scope.
                raise ConflictError("Holdout reservation identity is unavailable")
            if existing is not None:
                if existing["candidate_id"] != candidate_id or existing["candidate_digest"] != digest:
                    raise ConflictError("Evaluation scope already has a reserved candidate")
                cand_row = db.execute(
                    "SELECT c.*, e.primary_metric, e.direction, e.run_json, s.manifest_sha256 "
                    "FROM candidates c JOIN experiments e ON e.experiment_id=c.experiment_id "
                    "JOIN configurations cfg ON cfg.configuration_id=e.configuration_id "
                    "JOIN splits s ON s.split_id=cfg.split_id WHERE c.candidate_id=?",
                    (candidate_id,),
                ).fetchone()
                bundle = self.read_published_results(cand_row["job_id"])
                cand_data = next((c for c in bundle.get("candidates", []) if c.get("family") == cand_row["family"]), {})
                summary = cand_data.get("summary", [])
                score = next((s["mean"] for s in summary if s.get("name") == cand_row["primary_metric"]), 0.0)
                run_dict = json.loads(cand_row["run_json"])
                limits = ResourceLimits(**run_dict.get("limits", {}))
                sel = FrozenSelection(
                    finalization_id=existing["finalization_id"],
                    scope_id=scope_id,
                    experiment_id=existing["experiment_id"],
                    candidate_id=candidate_id,
                    candidate_digest=digest,
                    family=cand_row["family"],
                    task=TaskType(scope["task"]),
                    primary_metric=cand_row["primary_metric"],
                    primary_score=float(score if score is not None else 0.0),
                    direction=cand_row["direction"],
                    split_sha256=cand_row["manifest_sha256"],
                    config_sha256=cand_row["config_sha256"],
                    model_seed=cand_data.get("model_seed", 42),
                    resource_limits=limits,
                    created_at=existing["created_at"],
                )
                if "evidence_sha256" in existing.keys() and existing["evidence_sha256"] is not None:
                    if cand_data.get("folds"):
                        from ml.training.refit import candidate_from_dict, freeze_refit
                        from apps.api.services.workflow import load_refit_experiment
                        prep, _ = load_refit_experiment(self, existing["experiment_id"])
                        plan = freeze_refit(prep.context, sel, candidate_from_dict(cand_data))
                        if plan.evidence_sha256 != existing["evidence_sha256"] or plan.epoch_budget != existing["epoch_budget"]:
                            raise ConflictError("Frozen selection no longer matches its evidence")
                return sel

            cand_row = db.execute(
                "SELECT c.*, e.primary_metric, e.direction, e.run_json, cfg.scope_id as cfg_scope_id, s.manifest_sha256 "
                "FROM candidates c JOIN experiments e ON e.experiment_id=c.experiment_id "
                "JOIN configurations cfg ON cfg.configuration_id=e.configuration_id "
                "JOIN splits s ON s.split_id=cfg.split_id WHERE c.candidate_id=?",
                (candidate_id,),
            ).fetchone()
            if cand_row is None:
                raise NotFoundError("Candidate not found")
            if cand_row["cfg_scope_id"] != scope_id:
                raise ConflictError("Candidate does not belong to this evaluation scope")
            if cand_row["status"] != "completed":
                raise ConflictError("Candidate is not eligible for finalization")

            bundle = self.read_published_results(cand_row["job_id"])
            cand_data = next((c for c in bundle.get("candidates", []) if c.get("family") == cand_row["family"]), None)
            if cand_data is None or not cand_data.get("eligible"):
                raise ConflictError("Candidate is not eligible for finalization")

            from apps.api.services.finalization import compute_candidate_digest
            summary = cand_data.get("summary", [])
            score = next((s["mean"] for s in summary if s.get("name") == cand_row["primary_metric"]), None)
            if score is None or not math.isfinite(score):
                raise ConflictError("Candidate primary metric is undefined or nonfinite")

            computed = compute_candidate_digest(
                candidate_id=candidate_id,
                family=cand_row["family"],
                split_sha256=cand_row["manifest_sha256"],
                config_sha256=cand_row["config_sha256"],
                primary_metric=cand_row["primary_metric"],
                primary_score=score,
                model_seed=cand_data.get("model_seed", 42),
            )
            if computed != digest:
                raise ConflictError("Candidate digest mismatch")

            finalization_id = str(uuid4())
            now = _now()
            run_dict = json.loads(cand_row["run_json"])
            limits = ResourceLimits(**run_dict.get("limits", {}))
            selection = FrozenSelection(
                finalization_id=finalization_id,
                scope_id=scope_id,
                experiment_id=cand_row["experiment_id"],
                candidate_id=candidate_id,
                candidate_digest=digest,
                family=cand_row["family"],
                task=TaskType(scope["task"]),
                primary_metric=cand_row["primary_metric"],
                primary_score=float(score),
                direction=cand_row["direction"],
                split_sha256=cand_row["manifest_sha256"],
                config_sha256=cand_row["config_sha256"],
                model_seed=cand_data.get("model_seed", 42),
                resource_limits=limits,
                created_at=now,
            )

            from ml.tracking.manifest import canonical_bytes, digest_bytes
            from ml.training.refit import candidate_from_dict, freeze_refit
            evidence_sha256 = None
            epoch_budget = None
            plan_json = None
            if cand_data.get("folds"):
                candidate_result = candidate_from_dict(cand_data)
                from apps.api.services.workflow import load_refit_experiment
                prep, _ = load_refit_experiment(self, cand_row["experiment_id"])
                plan = freeze_refit(prep.context, selection, candidate_result)
                evidence_sha256 = plan.evidence_sha256
                epoch_budget = plan.epoch_budget
                plan_json = json.dumps(asdict(plan))
            else:
                evidence_sha256 = digest_bytes(canonical_bytes(cand_data))

            db.execute(
                "INSERT INTO finalizations(finalization_id, scope_id, experiment_id, candidate_id, "
                "candidate_digest, state, evidence_sha256, epoch_budget, plan_json, created_at, updated_at) "
                "VALUES(?,?,?,?,?,'reserved',?,?,?,?,?)",
                (finalization_id, scope_id, cand_row["experiment_id"], candidate_id, digest,
                 evidence_sha256, epoch_budget, plan_json, now, now),
            )
            db.execute(
                "INSERT INTO scope_lock_markers(scope_id, state) VALUES(?, 'reserved') ON CONFLICT(scope_id) DO NOTHING",
                (scope_id,),
            )
            _audit(db, "service", finalization_id, "finalization_reserved")
            return selection

    def enqueue_finalization(self, finalization_id: str, idempotency_key: str) -> JobRecord:
        _text(finalization_id)
        _text(idempotency_key)
        with self._transaction() as db:
            fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (finalization_id,)).fetchone()
            if fin is None:
                raise NotFoundError("Finalization not found")
            self._reject_finalization_deletion(db, fin)
            if fin["refit_job_id"] is not None:
                existing_job = db.execute("SELECT * FROM jobs WHERE job_id=?", (fin["refit_job_id"],)).fetchone()
                if existing_job:
                    return _record(existing_job)

            payload = _digest(_json(["finalization", fin["experiment_id"], fin["candidate_digest"]]))
            existing = db.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing:
                if existing["payload_sha256"] != payload:
                    raise ConflictError("Idempotency key was used with a different payload")
                return _record(existing)

            job_id, now = str(uuid4()), _now()
            db.execute("""INSERT INTO jobs(job_id, experiment_id, kind, idempotency_key, payload_sha256, state, created_at, updated_at)
                          VALUES(?, ?, 'finalization', ?, ?, 'queued', ?, ?)""",
                       (job_id, fin["experiment_id"], idempotency_key, payload, now, now))
            db.execute("UPDATE finalizations SET refit_job_id=?, state='refitting', updated_at=? WHERE finalization_id=?",
                       (job_id, now, finalization_id))
            _audit(db, "service", job_id, "finalization_enqueued")
            result = _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        logger.info("Finalization job queued: %s", result.job_id)
        return result

    def retry_pre_evaluation(self, finalization_id: str, idempotency_key: str) -> JobRecord:
        _text(finalization_id)
        _text(idempotency_key)
        with self._transaction() as db:
            fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (finalization_id,)).fetchone()
            if fin is None:
                raise NotFoundError("Finalization not found")
            self._reject_finalization_deletion(db, fin)

            if fin["evaluation_started_at"] is not None or fin["state"] in ("evaluating", "evaluated", "locked_failed"):
                raise ConflictError("Evaluation has already started or completed; holdout cannot be re-evaluated")

            payload = _digest(_json(["finalization", fin["experiment_id"], fin["candidate_digest"], "retry"]))
            existing = db.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing:
                if existing["payload_sha256"] != payload or existing["job_id"] != fin["refit_job_id"]:
                    raise ConflictError("Idempotency key was used with a different payload")
                return _record(existing)

            prev_job = db.execute("SELECT * FROM jobs WHERE job_id=?", (fin["refit_job_id"],)).fetchone()
            if (fin["state"] != "retryable_pre_evaluation" or prev_job is None
                    or prev_job["kind"] != "finalization" or prev_job["experiment_id"] != fin["experiment_id"]
                    or prev_job["state"] not in ("failed", "interrupted", "cancelled")):
                raise ConflictError("Previous finalization is not a retryable terminal job")
            if prev_job["attempt_id"] is None:
                if prev_job["state"] != "cancelled" or prev_job["owner_token"] is not None:
                    raise ConflictError("Previous child identity is unknown")
            else:
                proc = db.execute("SELECT child_pid,child_created FROM worker_processes WHERE attempt_id=?",
                                  (prev_job["attempt_id"],)).fetchone()
                if not proc or proc["child_pid"] is None or proc["child_created"] is None or \
                        process_liveness(proc["child_pid"], proc["child_created"]) != "dead":
                    raise ConflictError("Previous worker child must be verified dead before retry")

            self._verify_finalization_selection(db, fin)

            job_id, now = str(uuid4()), _now()
            db.execute("""INSERT INTO jobs(job_id, experiment_id, kind, idempotency_key, payload_sha256, state, created_at, updated_at)
                          VALUES(?, ?, 'finalization', ?, ?, 'queued', ?, ?)""",
                       (job_id, fin["experiment_id"], idempotency_key, payload, now, now))
            db.execute("UPDATE finalizations SET refit_job_id=?, state='refitting', updated_at=? WHERE finalization_id=?",
                       (job_id, now, finalization_id))
            _audit(db, "service", finalization_id, "finalization_retried")
            result = _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        logger.info("Finalization job retried: %s", result.job_id)
        return result

    @staticmethod
    def _reject_finalization_deletion(db: sqlite3.Connection, fin: sqlite3.Row) -> None:
        """Serialize job creation with deletion of any canonically equivalent upload."""
        if db.execute(
            "SELECT 1 FROM evaluation_scopes s JOIN datasets d ON d.canonical_sha256=s.canonical_sha256 "
            "JOIN deletion_intents i ON i.dataset_id=d.dataset_id WHERE s.scope_id=? LIMIT 1",
            (fin["scope_id"],),
        ).fetchone():
            raise ConflictError("Dataset deletion is pending")

    def _verify_finalization_selection(self, db: sqlite3.Connection, fin: sqlite3.Row) -> None:
        candidate = db.execute("SELECT c.job_id,c.family,c.config_sha256,c.experiment_id,e.primary_metric,s.manifest_sha256 "
                               "FROM candidates c JOIN experiments e ON e.experiment_id=c.experiment_id "
                               "JOIN configurations cfg ON cfg.configuration_id=e.configuration_id "
                               "JOIN splits s ON s.split_id=cfg.split_id WHERE c.candidate_id=? AND cfg.scope_id=?",
                               (fin["candidate_id"], fin["scope_id"])).fetchone()
        if candidate is None or candidate["experiment_id"] != fin["experiment_id"]:
            raise ConflictError("Frozen candidate identity is unavailable")
        from apps.api.services.finalization import compute_candidate_digest
        bundle = self.read_published_results(candidate["job_id"])
        data = next((c for c in bundle.get("candidates", []) if c.get("family") == candidate["family"]), None)
        score = next((s.get("mean") for s in data.get("summary", [])
                      if s.get("name") == candidate["primary_metric"]), None) if data else None
        if (not data or not data.get("eligible") or score is None or
                compute_candidate_digest(fin["candidate_id"], candidate["family"], candidate["manifest_sha256"],
                                         candidate["config_sha256"], candidate["primary_metric"], score,
                                         data.get("model_seed", 42)) != fin["candidate_digest"]):
            raise ConflictError("Frozen selection no longer matches its evidence")

        if data.get("folds") and (fin["evidence_sha256"] is None or fin["plan_json"] is None):
            raise ConflictError("Frozen refit evidence is unavailable")
        if "evidence_sha256" in fin.keys() and fin["evidence_sha256"] is not None:
            if data.get("folds"):
                from ml.training.refit import candidate_from_dict, freeze_refit
                from apps.api.services.workflow import load_refit_experiment
                prep, _ = load_refit_experiment(self, fin["experiment_id"])
                scope = db.execute("SELECT task FROM evaluation_scopes WHERE scope_id=?", (fin["scope_id"],)).fetchone()
                exp = db.execute("SELECT primary_metric, direction, run_json FROM experiments WHERE experiment_id=?", (fin["experiment_id"],)).fetchone()
                run_dict = json.loads(exp["run_json"])
                limits = ResourceLimits(**run_dict.get("limits", {}))
                sel = FrozenSelection(
                    finalization_id=fin["finalization_id"], scope_id=fin["scope_id"],
                    experiment_id=fin["experiment_id"], candidate_id=fin["candidate_id"],
                    candidate_digest=fin["candidate_digest"], family=candidate["family"],
                    task=TaskType(scope["task"]), primary_metric=exp["primary_metric"],
                    primary_score=float(score), direction=exp["direction"],
                    split_sha256=candidate["manifest_sha256"], config_sha256=candidate["config_sha256"],
                    model_seed=data.get("model_seed", 42), resource_limits=limits,
                    created_at=fin["created_at"],
                )
                plan = freeze_refit(prep.context, sel, candidate_from_dict(data))
                if plan.evidence_sha256 != fin["evidence_sha256"] or plan.epoch_budget != fin["epoch_budget"]:
                    raise ConflictError("Frozen selection no longer matches its evidence or budget")
                try:
                    stored_plan = json.loads(fin["plan_json"])
                except (TypeError, ValueError):
                    raise ConflictError("Frozen refit plan is invalid") from None
                if stored_plan != json.loads(json.dumps(asdict(plan))):
                    raise ConflictError("Frozen refit plan differs")
            else:
                from ml.tracking.manifest import canonical_bytes, digest_bytes
                if digest_bytes(canonical_bytes(data)) != fin["evidence_sha256"]:
                    raise ConflictError("Frozen selection evidence hash changed")

    def start_evaluation(self, finalization_id: str, expected_digest: str, *,
                         owner: WorkerIdentity, job_id: str, attempt_id: str) -> tuple[str, str]:
        _text(finalization_id)
        _text(expected_digest)
        with self._transaction() as db:
            fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (finalization_id,)).fetchone()
            if fin is None:
                raise NotFoundError("Finalization not found")
            if fin["evaluation_started_at"] is not None or fin["state"] in ("evaluating", "evaluated", "locked_failed"):
                raise ConflictError("Evaluation has already started or failed")
            if fin["candidate_digest"] != expected_digest:
                raise ConflictError("Candidate digest mismatch")

            if not isinstance(owner, WorkerIdentity) or fin["state"] != "refitting" or fin["refit_job_id"] != job_id:
                raise ConflictError("Active finalization owner required")
            job = self._owned_running_row(db, owner, job_id)
            if (job["kind"] != "finalization" or job["attempt_id"] != attempt_id
                    or job["experiment_id"] != fin["experiment_id"] or job["cancel_requested"]):
                raise ConflictError("Attempt is not authorized to start evaluation")
            proc = db.execute("SELECT child_pid,child_created FROM worker_processes WHERE attempt_id=?", (attempt_id,)).fetchone()
            if not proc or proc["child_pid"] is None or proc["child_created"] is None or \
                    process_liveness(proc["child_pid"], proc["child_created"]) != "alive":
                raise ConflictError("Active execution child identity required")
            self._verify_finalization_selection(db, fin)

            token = secrets.token_hex(32)
            now = _now()
            db.execute(
                "UPDATE finalizations SET state='evaluating', attempt_id=?, evaluation_started_at=?, updated_at=? WHERE finalization_id=?",
                (attempt_id, now, now, finalization_id),
            )
            db.execute("INSERT INTO evaluation_authorizations(finalization_id,token_sha256) VALUES(?,?)",
                       (finalization_id, _digest(token)))
            db.execute(
                "INSERT INTO scope_lock_markers(scope_id, state) VALUES(?, 'consumed') "
                "ON CONFLICT(scope_id) DO UPDATE SET state='consumed'",
                (fin["scope_id"],),
            )
            _audit(db, "service", finalization_id, "evaluation_started")
            return fin["scope_id"], token

    def get_finalization(self, scope_id: str) -> dict | None:
        _text(scope_id)
        with self._connection() as db:
            row = db.execute("SELECT * FROM finalizations WHERE scope_id=?", (scope_id,)).fetchone()
            return dict(row) if row is not None else None

    def consume_fixture_authorization(self, capability) -> dict:
        """Atomically authorize the fixture evaluator; no holdout file is opened."""
        return self._consume_evaluation_authorization(capability, fixture=True)

    def consume_holdout_authorization(self, capability) -> dict:
        """Consume the active attempt's one read, without publishing a result."""
        return self._consume_evaluation_authorization(capability, fixture=False)

    def _consume_evaluation_authorization(self, capability, *, fixture: bool) -> dict:
        from ml.contracts import HoldoutCapability
        if not isinstance(capability, HoldoutCapability):
            raise PermissionError("Holdout capability required")
        with self._transaction() as db:
            fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (capability.finalization_id,)).fetchone()
            if (fin is None or fin["scope_id"] != capability.scope_id or fin["state"] != "evaluating"
                    or fin["candidate_digest"] != capability.candidate_digest or fin["refit_job_id"] != capability.job_id
                    or fin["attempt_id"] != capability.attempt_id or fin["evaluation_started_at"] is None):
                raise ConflictError("Capability does not match the active selection and attempt")
            owner = WorkerIdentity(capability.owner_token, capability.owner_generation)
            job = self._owned_running_row(db, owner, capability.job_id)
            if job["attempt_id"] != capability.attempt_id or job["cancel_requested"]:
                raise ConflictError("Capability owner is stale or cancelled")
            auth = db.execute("SELECT token_sha256,used FROM evaluation_authorizations WHERE finalization_id=?",
                              (capability.finalization_id,)).fetchone()
            if not auth or auth["used"] or not secrets.compare_digest(auth["token_sha256"], _digest(capability._auth_token)):
                raise ConflictError("Invalid or already consumed capability")
            proc = db.execute("SELECT child_pid,child_created FROM worker_processes WHERE attempt_id=?", (capability.attempt_id,)).fetchone()
            if not proc or proc["child_pid"] is None or proc["child_created"] is None or \
                    process_liveness(proc["child_pid"], proc["child_created"]) != "alive":
                raise ConflictError("Execution child identity is not live")
            now = _now()
            db.execute("UPDATE evaluation_authorizations SET used=1 WHERE finalization_id=?", (capability.finalization_id,))
            if fixture:
                db.execute("UPDATE finalizations SET state='evaluated',updated_at=? WHERE finalization_id=?", (now, capability.finalization_id))
                _audit(db, "fixture", capability.finalization_id, "evaluation_evaluated")
            else:
                _audit(db, "worker", capability.finalization_id, "holdout_read_authorized")
            return dict(fin)

    def publish_finalization(
        self,
        owner: WorkerIdentity,
        job_id: str,
        attempt_id: str,
        finalization_id: str,
        result_path: str,
        result_checksum: str,
        *,
        telemetry: dict[str, object] | None = None,
    ) -> JobRecord:
        _text(job_id)
        _text(attempt_id)
        _text(finalization_id)
        if not isinstance(result_path, str) or not result_path.strip():
            raise ValueError("result_path must be a nonempty string")
        _text(result_checksum)
        if not isinstance(owner, WorkerIdentity):
            raise TypeError("Valid WorkerIdentity required")
        with self._transaction() as db:
            lease = db.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            if (lease["owner_token"] != owner.owner_token or lease["generation"] != owner.generation
                    or lease["active_job_id"] != job_id):
                raise ConflictError("Worker lease is stale or not owned")

            job = db.execute("SELECT * FROM jobs WHERE job_id=? AND attempt_id=?", (job_id, attempt_id)).fetchone()
            if job is None or job["state"] != "running":
                raise ConflictError("Job is not running for this attempt")

            fin = db.execute("SELECT * FROM finalizations WHERE finalization_id=?", (finalization_id,)).fetchone()
            if fin is None or fin["refit_job_id"] != job_id:
                raise ConflictError("Finalization record not found or job mismatch")

            if fin["state"] != "evaluating" or fin["evaluation_started_at"] is None or fin["attempt_id"] != attempt_id:
                raise ConflictError("Finalization was not in evaluating state for this attempt")

            if job["cancel_requested"]:
                now = _now()
                db.execute("UPDATE jobs SET state='cancelled',stage='cancelled',updated_at=? WHERE job_id=?", (now, job_id))
                db.execute("UPDATE finalizations SET state='locked_failed',updated_at=? WHERE finalization_id=?", (now, finalization_id))
                db.execute("UPDATE worker_lease SET owner_token=NULL,active_job_id=NULL WHERE singleton=1")
                _audit(db, "worker", finalization_id, "finalization_cancelled_after_start_locked_failed")
                return _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

            from ml.tracking.manifest import file_digest
            folder = Path(result_path)
            scope_dir = (self.path.parent / "final" / fin["scope_id"]).resolve()
            if folder.is_symlink() or folder.resolve() != folder.absolute() or folder.absolute() != scope_dir / attempt_id:
                raise ValueError("Invalid final bundle location")
            manifest_file = folder / "manifest.json"
            results_file = folder / "results.json"
            if not manifest_file.exists() or not results_file.exists():
                raise ValueError("Incomplete final result bundle")
            if file_digest(manifest_file) != result_checksum:
                raise ValueError("Final bundle manifest checksum mismatch")
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1 or manifest.get("origin") != "modelarena-final-v1":
                raise ValueError("Invalid final bundle manifest metadata")
            if file_digest(results_file) != manifest["artifact_sha256"]["results.json"]:
                raise ValueError("Final bundle results checksum mismatch")
            if (manifest.get("finalization_id") != finalization_id or manifest.get("attempt_id") != attempt_id
                    or manifest.get("scope_id") != fin["scope_id"] or manifest.get("candidate_id") != fin["candidate_id"]):
                raise ValueError("Final bundle metadata does not match finalization attempt")
            try:
                results_payload = json.loads(results_file.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError("Final bundle results payload is invalid JSON") from exc
            from ml.evaluation.finalize import validate_final_result_payload
            if not validate_final_result_payload(results_payload, fin):
                raise ValueError("Final bundle results payload failed schema or identity validation")

            now = _now()
            db.execute("UPDATE finalizations SET state='evaluated',result_checksum=?,result_path=?,updated_at=? WHERE finalization_id=?",
                       (result_checksum, str(folder), now, finalization_id))
            db.execute("UPDATE jobs SET state='completed',stage='completed',updated_at=? WHERE job_id=?", (now, job_id))
            db.execute("UPDATE scope_lock_markers SET state='consumed' WHERE scope_id=?", (fin["scope_id"],))
            db.execute("UPDATE worker_lease SET owner_token=NULL,active_job_id=NULL,heartbeat_at=? WHERE singleton=1", (now,))
            _audit(db, "worker", finalization_id, "finalization_evaluated")
            _audit(db, "worker", job_id, "job_completed")
            return _record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def read_published_final_result(self, scope_id: str) -> dict:
        """Read and verify the published final result for the evaluation scope."""
        _text(scope_id)
        from ml.tracking.manifest import file_digest
        with self._connection() as db:
            fin = db.execute("SELECT * FROM finalizations WHERE scope_id=? AND state='evaluated'", (scope_id,)).fetchone()
        if fin is None or fin["result_path"] is None or fin["result_checksum"] is None:
            raise NotFoundError("No published finalization result for this scope")
        folder = Path(fin["result_path"]).resolve()
        scope_dir = (self.path.parent / "final" / scope_id).resolve()
        if folder.parent != scope_dir and folder != scope_dir / folder.name:
            raise ValueError("Invalid final bundle location")
        if folder.is_symlink():
            raise ValueError("Unsafe bundle path")
        manifest_file = folder / "manifest.json"
        results_file = folder / "results.json"
        if not manifest_file.exists() or not results_file.exists():
            raise ValueError("Incomplete final result bundle")
        if file_digest(manifest_file) != fin["result_checksum"]:
            raise ValueError("Manifest checksum mismatch")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if file_digest(results_file) != manifest["artifact_sha256"]["results.json"]:
            raise ValueError("Results checksum mismatch")
        try:
            results_payload = json.loads(results_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError("Final bundle results payload is invalid JSON") from exc
        from ml.evaluation.finalize import validate_final_result_payload
        if not validate_final_result_payload(results_payload, fin):
            raise ValueError("Final bundle results payload failed schema or identity validation")
        return results_payload

    def _try_recover_staged_final_bundle(self, db: sqlite3.Connection, fin: sqlite3.Row, attempt_id: str, now: str) -> bool:
        from ml.tracking.manifest import file_digest
        scope_dir = (self.path.parent / "final" / fin["scope_id"]).resolve()
        final_dir = scope_dir / attempt_id
        staging_dir = scope_dir / ("staging-" + attempt_id)
        candidate_dir = None
        if final_dir.is_dir() and not final_dir.is_symlink():
            candidate_dir = final_dir
        elif staging_dir.is_dir() and not staging_dir.is_symlink():
            candidate_dir = staging_dir
        if candidate_dir is None:
            return False
        manifest_file = candidate_dir / "manifest.json"
        results_file = candidate_dir / "results.json"
        if not manifest_file.exists() or not results_file.exists():
            return False
        try:
            manifest_sha = file_digest(manifest_file)
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1 or manifest.get("origin") != "modelarena-final-v1":
                return False
            if (manifest.get("finalization_id") != fin["finalization_id"] or manifest.get("attempt_id") != attempt_id
                    or fin["attempt_id"] != attempt_id or manifest.get("scope_id") != fin["scope_id"]
                    or manifest.get("candidate_id") != fin["candidate_id"]):
                return False
            if file_digest(results_file) != manifest.get("artifact_sha256", {}).get("results.json"):
                return False
            try:
                results_payload = json.loads(results_file.read_text(encoding="utf-8"))
            except Exception:
                return False
            from ml.evaluation.finalize import validate_final_result_payload
            if not validate_final_result_payload(results_payload, fin):
                return False
            if candidate_dir == staging_dir:
                staging_dir.rename(final_dir)
                candidate_dir = final_dir
            db.execute("UPDATE finalizations SET state='evaluated',result_checksum=?,result_path=?,updated_at=? WHERE finalization_id=?",
                       (manifest_sha, str(candidate_dir), now, fin["finalization_id"]))
            db.execute("UPDATE jobs SET state='completed',stage='completed',updated_at=? WHERE job_id=?", (now, fin["refit_job_id"]))
            db.execute("UPDATE scope_lock_markers SET state='consumed' WHERE scope_id=?", (fin["scope_id"],))
            _audit(db, "recovery", fin["finalization_id"], "finalization_recovered_and_published")
            return True
        except Exception:
            return False
