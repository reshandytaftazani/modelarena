"""Deletion intents fence active jobs and never claim unverified erasure."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient

from apps.api.database.repository import ConflictError, JobRepository
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


def test_schema_v2_migrates_with_existing_datasets(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with repo._connection() as db:
        db.execute('DROP TABLE deletion_intents')
        db.execute('PRAGMA user_version=2')
    migrated = JobRepository(repo.path)
    with migrated._connection() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
        assert db.execute("SELECT name FROM sqlite_master WHERE name='deletion_intents'").fetchone()


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
