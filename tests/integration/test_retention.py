"""Deletion intents fence active jobs and never claim unverified erasure."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import sys

from fastapi.testclient import TestClient
import pytest

from apps.api.database.repository import ConflictError, JobRepository, WorkerIdentity
from apps.api.main import create_app
from apps.api.services.retention import request_deletion


TOKEN = 'a' * 43


def prepared(client):
    csv = ('x,target\n' + '\n'.join(f'{i},{i * 2}' for i in range(100))).encode()
    dataset_id = client.post('/api/v1/datasets', files={'file': ('a.csv', csv)}).json()['dataset_id']
    config_id = client.post(f'/api/v1/datasets/{dataset_id}/configurations', json={
        'task': 'regression', 'target': 'target', 'features': [{'name': 'x', 'dtype': 'numeric'}]}).json()['configuration_id']
    experiment_id = client.post('/api/v1/experiments', json={'configuration_id': config_id}).json()['experiment_id']
    return dataset_id, experiment_id, csv


def test_active_references_prevent_deletion_then_intent_blocks_new_work(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, experiment_id, csv = prepared(client)
        disclosure = client.get(f'/api/v1/datasets/{dataset_id}/deletion-disclosure')
        assert disclosure.status_code == 200
        assert disclosure.json()['linked_experiments'] >= 1
        assert disclosure.json()['confirmed_deletion_supported'] is False
        assert 'scope_hmac' not in disclosure.text
        job = client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'one'}).json()['job_id']
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).status_code == 409
        assert (tmp_path / 'uploads' / dataset_id / 'source.csv').exists()
        client.post(f'/api/v1/jobs/{job}/cancel')
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': False}).status_code == 422
        deleted = client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True})
        assert deleted.status_code == 202
        assert deleted.json()['state'] == 'deletion_pending'
        assert not (tmp_path / 'uploads' / dataset_id / 'source.csv').exists()
        assert client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'two'}).status_code == 409
        assert client.post('/api/v1/datasets', files={'file': ('a.csv', csv)}).status_code == 409
        assert client.post('/api/v1/datasets', files={'file': ('a.csv', csv.replace(b'\n', b'\r\n'))}).status_code == 409
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).json()['state'] == 'deletion_pending'


def test_locked_file_stays_pending_and_retry_is_safe(tmp_path, monkeypatch):
    from apps.api.services import retention
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, _, _ = prepared(client)
        original = retention._remove_file
        monkeypatch.setattr(retention, '_remove_file', lambda _: (_ for _ in ()).throw(PermissionError()))
        response = client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True})
        assert response.status_code == 202 and response.json()['state'] == 'deletion_pending'
        assert (tmp_path / 'uploads' / dataset_id / 'source.csv').exists()
        monkeypatch.setattr(retention, '_remove_file', original)
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).status_code == 202
        assert not (tmp_path / 'uploads' / dataset_id / 'source.csv').exists()


def test_running_job_rejects_deletion_without_recording_intent(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, experiment_id, _ = prepared(client)
        job_id = client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'running'}).json()['job_id']
        claimed = repo.claim_next(WorkerIdentity('test-worker', repo.lease_generation()))
        assert claimed.job_id == job_id and claimed.state == 'running'
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).status_code == 409
        with repo._connection() as db:
            assert db.execute('SELECT count(*) FROM deletion_intents').fetchone()[0] == 0
        assert (tmp_path / 'uploads' / dataset_id / 'source.csv').exists()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows exclusive file handle acceptance')
def test_real_windows_locked_upload_remains_pending_until_handle_closed(tmp_path):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                   wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, _, _ = prepared(client)
        path = tmp_path / 'uploads' / dataset_id / 'source.csv'
        handle = kernel.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)
        assert handle != wintypes.HANDLE(-1).value
        try:
            response = client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True})
            assert response.status_code == 202
            assert response.json()['state'] == 'deletion_pending'
            assert response.json()['file_cleanup_failed'] is True
            assert path.exists()
        finally:
            kernel.CloseHandle(handle)
        response = client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True})
        assert response.json()['file_cleanup_failed'] is False
        assert not path.exists()


def test_schema_v2_migrates_with_existing_datasets(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with repo._connection() as db:
        db.execute('DROP TABLE deletion_intents')
        db.execute('PRAGMA user_version=2')
    migrated = JobRepository(repo.path)
    with migrated._connection() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 4
        assert db.execute("SELECT name FROM sqlite_master WHERE name='deletion_intents'").fetchone()


def test_schema_v3_migrates_without_losing_existing_dataset(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, _, _ = prepared(client)
    with repo._connection() as db:
        db.execute('DROP TABLE worker_presence')
        db.execute('DROP TABLE scope_lock_markers')
        db.execute('DROP TABLE scope_tombstones')
        db.execute('PRAGMA user_version=3')
    migrated = JobRepository(repo.path)
    with migrated._connection() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 4
        assert db.execute('SELECT 1 FROM datasets WHERE dataset_id=?', (dataset_id,)).fetchone()
        assert db.execute("SELECT count(*) FROM sqlite_master WHERE name IN ('worker_presence','scope_lock_markers','scope_tombstones')").fetchone()[0] == 3


def test_enqueue_and_deletion_are_arbitrated_in_one_database_transaction(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, experiment_id, _ = prepared(client)
    barrier = Barrier(2)

    def enqueue():
        barrier.wait()
        try:
            return repo.enqueue_cv(experiment_id, 'race').state
        except ConflictError:
            return 'conflict'

    def delete():
        barrier.wait()
        try:
            return request_deletion(repo, dataset_id)['state']
        except ConflictError:
            return 'conflict'

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(enqueue), pool.submit(delete)
        assert (a.result(), b.result()) in {('queued', 'conflict'), ('conflict', 'deletion_pending')}


def test_consumed_scope_creates_minimal_keyed_tombstone_before_file_purge(tmp_path):
    from apps.api.services.retention import find_tombstone
    from ml.data.csv_io import validate_csv
    from ml.contracts import ResourceLimits
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, _, csv = prepared(client)
        with repo._connection() as db:
            scope = db.execute('SELECT scope_id,canonical_sha256,target,task FROM evaluation_scopes').fetchone()
            db.execute("INSERT INTO scope_lock_markers(scope_id,state) VALUES(?, 'consumed')", (scope['scope_id'],))
        response = client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True})
        assert response.status_code == 202 and response.json()['state'] == 'deletion_pending'
        alternate = tmp_path / 'equivalent.csv'
        alternate.write_bytes(csv.replace(b'\n', b'\r\n'))
        identity = validate_csv(alternate, ResourceLimits()).canonical_sha256
        assert identity == scope['canonical_sha256']
        assert find_tombstone(repo, identity, scope['target'], scope['task']) == 'consumed'
        with repo._connection() as db:
            row = dict(db.execute('SELECT * FROM scope_tombstones').fetchone())
        assert set(row) == {'scope_hmac', 'version', 'state'}
        assert row['state'] == 'consumed' and scope['target'] not in row['scope_hmac']
        assert row['scope_hmac'] != identity


def test_tombstone_is_detected_after_fixture_metadata_removal_and_equivalent_reupload(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, _, csv = prepared(client)
        with repo._connection() as db:
            scope_id = db.execute('SELECT scope_id FROM evaluation_scopes').fetchone()[0]
            db.execute("INSERT INTO scope_lock_markers VALUES(?, 'consumed')", (scope_id,))
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).status_code == 202

        # Simulate the later metadata-cleanup phase; it is not implemented by
        # the Day 13 deletion endpoint, which continues to return pending.
        with repo._transaction() as db:
            db.execute('DROP TRIGGER audit_no_delete')
            db.execute('DELETE FROM audit_events')
            db.execute('DELETE FROM scope_lock_markers')
            db.execute('DELETE FROM experiments')
            db.execute('DELETE FROM configurations')
            db.execute('DELETE FROM splits')
            db.execute('DELETE FROM evaluation_scopes')
            db.execute('DELETE FROM deletion_intents')
            db.execute('DELETE FROM datasets')
            db.execute("CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END")
        uploaded = client.post('/api/v1/datasets', files={'file': ('equivalent.csv', csv.replace(b'\n', b'\r\n'))})
        assert uploaded.status_code == 201
        assert uploaded.json()['dataset_id'] != dataset_id
        config = client.post(f"/api/v1/datasets/{uploaded.json()['dataset_id']}/configurations", json={
            'task': 'regression', 'target': 'target', 'features': [{'name': 'x', 'dtype': 'numeric'}]})
        assert config.status_code == 201
        assert config.json()['holdout_lock'] == 'consumed'


def test_unreserved_deletion_creates_no_tombstone_or_key(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, _, _ = prepared(client)
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).status_code == 202
    with repo._connection() as db:
        assert db.execute('SELECT count(*) FROM scope_tombstones').fetchone()[0] == 0
    assert not (tmp_path / 'scope-tombstone.key').exists()


def test_lost_tombstone_key_blocks_lookup_without_replacing_key(tmp_path):
    from apps.api.services.retention import find_tombstone
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, _, _ = prepared(client)
        with repo._connection() as db:
            scope = db.execute('SELECT scope_id,canonical_sha256,target,task FROM evaluation_scopes').fetchone()
            db.execute("INSERT INTO scope_lock_markers VALUES(?, 'reserved')", (scope['scope_id'],))
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).status_code == 202
    key = tmp_path / 'scope-tombstone.key'
    key.unlink()
    with pytest.raises(RuntimeError, match='unavailable'):
        find_tombstone(repo, scope['canonical_sha256'], scope['target'], scope['task'])
    assert not key.exists()


def test_consumed_tombstone_never_downgrades_on_retry(tmp_path):
    from apps.api.services.retention import find_tombstone
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, _, _ = prepared(client)
        with repo._connection() as db:
            scope = db.execute('SELECT scope_id,canonical_sha256,target,task FROM evaluation_scopes').fetchone()
            db.execute("INSERT INTO scope_lock_markers VALUES(?, 'consumed')", (scope['scope_id'],))
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).status_code == 202
        with repo._connection() as db:
            db.execute("UPDATE scope_lock_markers SET state='reserved' WHERE scope_id=?", (scope['scope_id'],))
        assert client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True}).status_code == 202
        assert find_tombstone(repo, scope['canonical_sha256'], scope['target'], scope['task']) == 'consumed'


def test_known_private_partition_split_final_and_report_copies_are_removed(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id, experiment_id, _ = prepared(client)
        job = repo.enqueue_cv(experiment_id, 'terminal-attempt')
        owner = WorkerIdentity('fixture-worker', repo.lease_generation())
        claimed = repo.claim_next(owner)
        assert claimed.job_id == job.job_id
        repo.finish_job(owner, job.job_id, claimed.attempt_id, state='failed', stage='failed')
        attempt_id = claimed.attempt_id
        with repo._connection() as db:
            scope_id, split_id = db.execute('SELECT scope_id,split_id FROM configurations WHERE dataset_id=?', (dataset_id,)).fetchone()
        paths = [tmp_path / 'datasets' / scope_id / 'train.csv',
                 tmp_path / 'datasets' / scope_id / 'holdout.csv',
                 tmp_path / 'splits' / f'{split_id}.json',
                 tmp_path / 'final' / scope_id / 'model.bin',
                 tmp_path / 'reports' / f'{experiment_id}.zip',
                 tmp_path / 'results' / attempt_id / 'result.json',
                 tmp_path / 'results' / f'staging-{attempt_id}' / 'result.json',
                 tmp_path / 'jobs' / attempt_id / 'staging' / 'temporary.csv']
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'private fixture')
        response = client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True})
        assert response.status_code == 202 and not response.json()['file_cleanup_failed']
        assert all(not path.exists() for path in paths)
