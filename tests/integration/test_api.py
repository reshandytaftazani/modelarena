"""Security and read-only CV/job transport for the partial Day 13 API."""

from fastapi.testclient import TestClient

from apps.api.main import create_app
from ml.contracts import TaskType
from tests.integration.test_classical_suite import prepared_fixture


TOKEN = "a" * 43


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
