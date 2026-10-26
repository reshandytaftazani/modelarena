"""Version-one application schema. No migrations run at module import time."""

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    raw_sha256 TEXT NOT NULL UNIQUE,
    canonical_sha256 TEXT NOT NULL,
    parse_policy_version INTEGER NOT NULL,
    byte_count INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE evaluation_scopes (
    scope_id TEXT PRIMARY KEY,
    canonical_sha256 TEXT NOT NULL,
    target TEXT NOT NULL,
    task TEXT NOT NULL,
    partition_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(canonical_sha256, target, task)
);
CREATE TABLE splits (
    split_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES evaluation_scopes(scope_id),
    manifest_sha256 TEXT NOT NULL UNIQUE,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE configurations (
    configuration_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    scope_id TEXT NOT NULL REFERENCES evaluation_scopes(scope_id),
    split_id TEXT NOT NULL REFERENCES splits(split_id),
    config_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(dataset_id, split_id, config_sha256)
);
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    configuration_id TEXT NOT NULL REFERENCES configurations(configuration_id),
    run_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('maximize', 'minimize')),
    queue_limit INTEGER NOT NULL CHECK(queue_limit BETWEEN 1 AND 10),
    created_at TEXT NOT NULL,
    UNIQUE(configuration_id, payload_sha256)
);
CREATE TABLE jobs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    kind TEXT NOT NULL CHECK(kind='cv'),
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
);
CREATE UNIQUE INDEX one_running_job ON jobs((1)) WHERE state='running';
CREATE INDEX queued_jobs ON jobs(state, sequence);
CREATE TABLE worker_lease (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    owner_token TEXT,
    generation INTEGER NOT NULL CHECK(generation > 0),
    active_job_id TEXT UNIQUE REFERENCES jobs(job_id),
    heartbeat_at TEXT
);
INSERT INTO worker_lease(singleton, generation) VALUES(1, 1);
CREATE TABLE candidates (
    candidate_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    attempt_id TEXT NOT NULL,
    family TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('completed','failed','interrupted','cancelled')),
    config_sha256 TEXT NOT NULL,
    result_json TEXT,
    bundle_path TEXT,
    bundle_sha256 TEXT,
    mlflow_run_id TEXT
);
CREATE TABLE audit_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    actor TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event TEXT NOT NULL
);
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
"""

# Metadata updates must create a new identity, never rewrite an existing run.
for _table in ("datasets", "evaluation_scopes", "splits", "configurations", "experiments"):
    SCHEMA_SQL += f"""
CREATE TRIGGER {_table}_immutable BEFORE UPDATE ON {_table}
BEGIN SELECT RAISE(ABORT, 'metadata is immutable'); END;
"""
