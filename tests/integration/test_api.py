"""Security and read-only CV/job transport for the partial Day 13 API."""

from fastapi.testclient import TestClient

from apps.api.main import create_app
from ml.contracts import TaskType
from ml.contracts import ResourceLimits
from apps.api.database.repository import JobRepository
from apps.api.workers.supervisor import WorkerSupervisor
from uuid import UUID
from tests.integration.test_classical_suite import prepared_fixture


TOKEN = "a" * 43


def test_workflow_routes_publish_explicit_response_models(tmp_path):
    app = create_app(JobRepository(tmp_path / 'metadata.sqlite3'), session_token=TOKEN)
    routes = {(route.path, method): route for route in app.routes if hasattr(route, 'methods')
              for method in route.methods}
    expected = [('/api/v1/datasets', 'POST'),
                ('/api/v1/datasets/{dataset_id}/configurations', 'POST'),
                ('/api/v1/configurations/{configuration_id}/profile', 'GET'),
                ('/api/v1/experiments', 'POST'),
                ('/api/v1/experiments/{experiment_id}/jobs', 'POST'),
                ('/api/v1/jobs/{job_id}', 'GET'),
                ('/api/v1/experiments/{experiment_id}/leaderboard', 'GET'),
                ('/api/v1/candidates/{candidate_id}', 'GET')]
    assert all(routes[item].response_model is not None for item in expected)


def test_job_and_empty_leaderboard_with_local_token(tmp_path):
    repo, prepared, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    app = create_app(repo, session_token=TOKEN, allowed_origins=("http://localhost:5173",))
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        response = client.get(f"/api/v1/jobs/{job.job_id}")
        assert response.status_code == 200
        assert response.json()["state"] == "queued"
        assert "owner_token" not in response.json()
        leaderboard = client.get(f"/api/v1/experiments/{prepared.experiment_id}/leaderboard").json()
        assert leaderboard["partition"] == "cross_validation_training"
        assert leaderboard["ranked"] == []
        assert client.post(f"/api/v1/jobs/{job.job_id}/cancel").json()["state"] == "cancelled"
        assert client.get("/api/v1/jobs/unknown").status_code == 404


def test_token_host_and_origin_are_checked(tmp_path):
    repo, _, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    app = create_app(repo, session_token=TOKEN, allowed_origins=("http://localhost:5173",))
    with TestClient(app, base_url="http://localhost:8000") as client:
        url = f"/api/v1/jobs/{job.job_id}"
        assert client.get(url).status_code == 403
        assert client.get(url, headers={"X-ModelArena-Token": "wrong"}).status_code == 403
        assert client.get(url, headers={"X-ModelArena-Token": TOKEN, "Origin": "http://evil.test"}).status_code == 403
        assert client.get(url, headers={"X-ModelArena-Token": TOKEN, "Host": "evil.test"}).status_code == 400
        assert client.get(url, headers={"X-ModelArena-Token": TOKEN, "Origin": "http://localhost:5173"}).status_code == 200
        rejected = client.get(url, headers={"X-ModelArena-Token": "wrong", "Origin": "http://localhost:5173"})
        assert rejected.status_code == 403
        assert rejected.headers['Access-Control-Allow-Origin'] == 'http://localhost:5173'
        assert 'X-Request-ID' in rejected.headers['Access-Control-Expose-Headers']


def test_cancel_rejects_bodies_and_unknown_ids_safely(tmp_path):
    repo, _, _, job = prepared_fixture(tmp_path, TaskType.REGRESSION)
    app = create_app(repo, session_token=TOKEN)
    with TestClient(app, base_url="http://127.0.0.1:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        assert client.post(f"/api/v1/jobs/{job.job_id}/cancel", content=b"x" * 1025).status_code == 413
        assert client.post(f"/api/v1/jobs/{job.job_id}/cancel", content=b"x").status_code == 422
        assert client.post("/api/v1/jobs/unknown/cancel").status_code == 404
        assert repo.get_job(job.job_id).state == "queued"


def test_preflight_is_limited_to_configured_origin_and_headers(tmp_path):
    repo, _, _, _ = prepared_fixture(tmp_path, TaskType.REGRESSION)
    app = create_app(repo, session_token=TOKEN, allowed_origins=("http://localhost:5173",))
    with TestClient(app, base_url="http://localhost:8000") as client:
        headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET",
                   "Access-Control-Request-Headers": "X-ModelArena-Token"}
        response = client.options("/api/v1/jobs/id", headers=headers)
        assert response.status_code == 200
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        headers["Access-Control-Request-Headers"] = "Authorization"
        assert client.options("/api/v1/jobs/id", headers=headers).status_code == 403
        headers["Origin"] = "http://evil.test"
        assert client.options("/api/v1/jobs/id", headers=headers).status_code == 403


def test_upload_is_private_deduplicated_and_never_uses_client_filename(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    app = create_app(repo, session_token=TOKEN)
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        files = {"file": ('../../<script>alert(1)</script>.csv', b'x,target\n1,0\n2,1\n', 'text/csv')}
        first = client.post('/api/v1/datasets', files=files)
        assert first.status_code == 201, first.text
        body = first.json()
        assert body['columns'] == ['x', 'target'] and body['row_count'] == 2
        assert '<script>' not in first.text and '..' not in first.text
        assert (tmp_path / 'uploads' / body['dataset_id'] / 'source.csv').read_bytes() == b'x,target\n1,0\n2,1\n'
        second = client.post('/api/v1/datasets', files=files)
        assert second.status_code == 201 and second.json()['dataset_id'] == body['dataset_id']
        assert len(list((tmp_path / 'uploads').iterdir())) == 1


def test_upload_invalid_and_oversized_rejected_without_dataset(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN, limits=ResourceLimits(upload_bytes=64))
    with TestClient(app, base_url='http://localhost:8000', headers={'X-ModelArena-Token': TOKEN}) as client:
        assert client.post('/api/v1/datasets', files={'file': ('a.csv', b'a,a\n1,2\n')}).status_code == 422
        assert client.post('/api/v1/datasets', files={'file': ('a.csv', b'a\n' + b'x' * 65)}).status_code == 413
        assert client.post('/api/v1/datasets', files=[('file', ('a.csv', b'a\n1')), ('file', ('b.csv', b'b\n2'))]).status_code == 422
    with repo._connection() as db:
        assert db.execute('SELECT count(*) FROM datasets').fetchone()[0] == 0
    assert not list((tmp_path / 'uploads').iterdir())


def test_streamed_multipart_limit_and_truncation_leave_no_staging_files(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN, limits=ResourceLimits(upload_bytes=64))
    prefix = b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="x.csv"\r\n\r\n'
    with TestClient(app, base_url='http://localhost:8000', headers={'X-ModelArena-Token': TOKEN}) as client:
        response = client.post('/api/v1/datasets', headers={'Content-Type': 'multipart/form-data; boundary=boundary'},
                               content=iter([prefix, b'x\n', b'a' * 65, b'\r\n--boundary--\r\n']))
        assert response.status_code == 413
        response = client.post('/api/v1/datasets', headers={'Content-Type': 'multipart/form-data; boundary=boundary'},
                               content=iter([prefix, b'x\n1\n']))
        assert response.status_code == 422
    assert not list((tmp_path / 'uploads').iterdir())


def test_upload_configure_queue_and_cancel_without_training_in_api(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN)
    csv = ('x,target\n' + '\n'.join(f'{i},{i * 2}' for i in range(100))).encode()
    with TestClient(app, base_url='http://localhost:8000', headers={'X-ModelArena-Token': TOKEN}) as client:
        upload = client.post('/api/v1/datasets', files={'file': ('data.csv', csv)}).json()
        dataset_id = upload['dataset_id']
        config = {'task': 'regression', 'target': 'target', 'features': [{'name': 'x', 'dtype': 'numeric'}]}
        configured = client.post(f'/api/v1/datasets/{dataset_id}/configurations', json=config)
        assert configured.status_code == 201, configured.text
        configuration_id = configured.json()['configuration_id']
        profile = client.get(f'/api/v1/configurations/{configuration_id}/profile')
        assert profile.status_code == 200 and profile.json()['partition'] == 'training'
        assert profile.json()['row_count'] == 80
        experiment = client.post('/api/v1/experiments', json={'configuration_id': configuration_id, 'primary_metric': 'rmse'})
        assert experiment.status_code == 201, experiment.text
        experiment_id = experiment.json()['experiment_id']
        queued = client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'once'})
        assert queued.status_code == 202, queued.text
        job_id = queued.json()['job_id']
        assert queued.json()['state'] == 'queued'
        assert client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'once'}).json()['job_id'] == job_id
        other = client.post('/api/v1/experiments', json={'configuration_id': configuration_id, 'primary_metric': 'mae'})
        assert other.status_code == 201
        assert client.post(f"/api/v1/experiments/{other.json()['experiment_id']}/jobs", json={'idempotency_key': 'once'}).status_code == 409
        assert client.post(f'/api/v1/experiments/{experiment_id}/jobs', content=b' ' * 65537).status_code == 413
        assert client.get(f'/api/v1/jobs/{job_id}').json()['state'] == 'queued'
        assert client.post(f'/api/v1/jobs/{job_id}/cancel').json()['state'] == 'cancelled'
        retried = client.post(f'/api/v1/jobs/{job_id}/retry', json={'idempotency_key': 'retry-once'})
        assert retried.status_code == 202
        assert retried.json()['job_id'] != job_id and retried.json()['state'] == 'queued'
        assert client.post(f'/api/v1/jobs/{job_id}/retry', json={'idempotency_key': 'retry-once'}).json()['job_id'] == retried.json()['job_id']


def test_configuration_rejects_bad_targets_and_unknown_datasets(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN)
    with TestClient(app, base_url='http://localhost:8000', headers={'X-ModelArena-Token': TOKEN}) as client:
        bad = {'task': 'regression', 'target': 'unknown', 'features': [{'name': 'x', 'dtype': 'numeric'}]}
        assert client.post('/api/v1/datasets/00000000-0000-0000-0000-000000000000/configurations', json=bad).status_code == 404
        csv = ('x,target\n' + '\n'.join(f'{i},{i * 2}' for i in range(100))).encode()
        dataset_id = client.post('/api/v1/datasets', files={'file': ('a.csv', csv)}).json()['dataset_id']
        assert client.post(f'/api/v1/datasets/{dataset_id}/configurations', json=bad).status_code == 422


def test_queued_upload_can_be_loaded_by_separate_worker_after_reopening_store(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN)
    csv = ('x,target\n' + '\n'.join(f'{i},{i * 2}' for i in range(100))).encode()
    with TestClient(app, base_url='http://localhost:8000', headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id = client.post('/api/v1/datasets', files={'file': ('data.csv', csv)}).json()['dataset_id']
        config_id = client.post(f'/api/v1/datasets/{dataset_id}/configurations', json={
            'task': 'regression', 'target': 'target', 'features': [{'name': 'x', 'dtype': 'numeric'}]}).json()['configuration_id']
        experiment_id = client.post('/api/v1/experiments', json={'configuration_id': config_id}).json()['experiment_id']
        job_id = client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'worker'}).json()['job_id']
        worker = WorkerSupervisor(JobRepository(repo.path), workload='classical', load_from_storage=True,
                                  heartbeat_seconds=0.2, sample_seconds=0.1, poll_seconds=0.05)
        assert worker.run_once().state == 'completed'
        job_response = client.get(f'/api/v1/jobs/{job_id}').json()
        assert job_response['state'] == 'completed'
        assert job_response['progress']['finished_units'] == 25
        assert job_response['progress']['total_units'] == 25
        assert job_response['progress']['family'] == 'mlp'
        assert job_response['progress']['fold_index'] == 4
        ranking = client.get(f'/api/v1/experiments/{experiment_id}/leaderboard').json()
        assert ranking['job_id'] == job_id and len(ranking['ranked']) == 5
        candidate = client.get(f"/api/v1/candidates/{ranking['ranked'][0]['candidate_id']}")
        assert candidate.status_code == 200
        assert candidate.json()['diagnostics']['partition'] == 'out_of_fold_training'
        assert 'bundle_path' not in candidate.text
        with repo._connection() as db:
            attempt_id = db.execute('SELECT attempt_id FROM jobs WHERE job_id=?', (job_id,)).fetchone()[0]
        assert (tmp_path / 'results' / attempt_id).exists()
        deletion = client.request('DELETE', f'/api/v1/datasets/{dataset_id}', json={'acknowledge': True})
        assert deletion.status_code == 202 and deletion.json()['state'] == 'deletion_pending'
        assert not (tmp_path / 'results' / attempt_id).exists()
        assert (tmp_path / 'mlflow.sqlite3').exists()  # still not confirmed deleted
        assert client.get(f'/api/v1/configurations/{config_id}/profile').status_code == 409
        assert client.get(f'/api/v1/experiments/{experiment_id}/leaderboard').status_code == 409
        assert client.get(f"/api/v1/candidates/{ranking['ranked'][0]['candidate_id']}").status_code == 409


def test_worker_rejects_source_replaced_after_queue(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN)
    csv = ('x,target\n' + '\n'.join(f'{i},{i * 2}' for i in range(100))).encode()
    with TestClient(app, base_url='http://localhost:8000', headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id = client.post('/api/v1/datasets', files={'file': ('data.csv', csv)}).json()['dataset_id']
        config_id = client.post(f'/api/v1/datasets/{dataset_id}/configurations', json={
            'task': 'regression', 'target': 'target', 'features': [{'name': 'x', 'dtype': 'numeric'}]}).json()['configuration_id']
        experiment_id = client.post('/api/v1/experiments', json={'configuration_id': config_id}).json()['experiment_id']
        job_id = client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'tampered'}).json()['job_id']
        (tmp_path / 'uploads' / dataset_id / 'source.csv').write_bytes(csv.replace(b'1,2', b'1,3', 1))
        worker = WorkerSupervisor(JobRepository(repo.path), workload='classical', load_from_storage=True)
        assert worker.run_once().state == 'failed'
        assert worker.last_child_pid is None
        assert client.get(f'/api/v1/jobs/{job_id}').json()['state'] == 'failed'
        assert client.get(f'/api/v1/experiments/{experiment_id}/leaderboard').json()['ranked'] == []


def test_all_error_responses_have_safe_request_ids_and_envelopes(tmp_path, monkeypatch):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN)
    with TestClient(app, base_url='http://localhost:8000') as client:
        requests = [
            client.get('/health'),
            client.get('/health', headers={'Host': 'evil.test', 'X-ModelArena-Token': TOKEN}),
            client.get('/api/v1/missing', headers={'X-ModelArena-Token': TOKEN}),
            client.post('/api/v1/experiments', headers={'X-ModelArena-Token': TOKEN}, json={'configuration_id': 42, 'secret-value': 'hidden'}),
        ]
        for response in requests:
            assert response.status_code >= 400
            payload = response.json()['error']
            assert set(payload) == {'code', 'message', 'details', 'request_id'}
            assert str(UUID(payload['request_id'])) == payload['request_id']
            assert response.headers['X-Request-ID'] == payload['request_id']
            assert 'hidden' not in response.text and 'secret-value' not in response.text
        assert len({response.headers['X-Request-ID'] for response in requests}) == len(requests)
        assert requests[-1].json()['error']['details'] == [{'field': 'configuration_id', 'code': 'invalid'}]

        def fail(_):
            raise RuntimeError('secret-path-and-private-value')
        monkeypatch.setattr(repo, 'get_job', fail)
        response = client.get('/api/v1/jobs/00000000-0000-0000-0000-000000000001', headers={'X-ModelArena-Token': TOKEN})
        assert response.status_code == 500
        assert response.json()['error']['code'] == 'internal_error'
        assert 'secret-path-and-private-value' not in response.text


def test_invalid_host_header_and_encoded_traversal_are_rejected(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    with TestClient(create_app(repo, session_token=TOKEN), base_url='http://localhost:8000',
                    headers={'X-ModelArena-Token': TOKEN}) as client:
        for host in ('localhost.evil.test', '127.0.0.1.evil.test', 'evil.test:80',
                     'localhost:notaport', 'localhost:70000', 'localhost:', 'localhost:1:2'):
            assert client.get('/health', headers={'Host': host}).status_code == 400
        for path in ('/api/v1/jobs/%2e%2e%2fmetadata.sqlite3', '/api/v1/datasets/%2e%2e%2fmetadata.sqlite3/deletion-disclosure'):
            response = client.get(path)
            assert response.status_code == 404
            assert 'metadata.sqlite3' not in response.text
        duplicate_host = client.get('/health', headers=[('Host', 'localhost'), ('Host', 'evil.test')])
        assert duplicate_host.status_code == 400


def test_get_experiment_report_endpoint(tmp_path):
    from apps.api.services.reports import verify_report_zip
    from tests.unit.test_reports import _create_cv_attempt
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")
    app = create_app(repo, session_token=TOKEN)
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        res_404 = client.get("/api/v1/experiments/00000000-0000-0000-0000-000000000000/report")
        assert res_404.status_code == 404

        res = client.get(f"/api/v1/experiments/{prepared.experiment_id}/report")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/zip"
        downloaded = tmp_path / "downloaded.zip"
        downloaded.write_bytes(res.content)
        verified = verify_report_zip(downloaded)
        assert verified["manifest"]["status"] == "cv_only"
        assert "CV only — holdout not evaluated" in verified["html"]
