"""Local process entrypoints, health, and worker-presence lifecycle."""

from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys
import socket
from time import monotonic, sleep

import httpx
import psutil
import pytest
from fastapi.testclient import TestClient

from apps.api.database.repository import JobRepository
from apps.api.main import create_app
from apps.api.workers.monitor import process_creation_time
from apps.api.workers.monitor import terminate_owned_child


TOKEN = "a" * 43


def test_health_reports_verified_worker_and_stale_unknown(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    app = create_app(repo, session_token=TOKEN)
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        assert client.get("/health").json()["worker"] == "not_detected"
        with repo._connection() as db:
            db.execute("INSERT INTO worker_presence VALUES(?,?,?,?)", (
                "test-owner", os.getpid(), process_creation_time(os.getpid()), datetime.now(timezone.utc).isoformat()))
        assert client.get("/health").json()["worker"] == "available"
        with repo._connection() as db:
            db.execute("UPDATE worker_presence SET heartbeat_at=?", ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),))
        assert client.get("/health").json()["worker"] == "unknown"


def test_worker_once_entrypoint_starts_and_unregisters_on_windows(tmp_path):
    env = dict(os.environ, MODELARENA_HOME=str(tmp_path))
    result = subprocess.run([sys.executable, "-m", "apps.api.workers.runner", "--once"],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    with repo._connection() as db:
        assert db.execute("SELECT count(*) FROM worker_presence").fetchone()[0] == 0


def test_server_uses_random_session_token_and_loopback_only(tmp_path, monkeypatch, capsys):
    from apps.api import server
    monkeypatch.setenv("MODELARENA_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    server.main(["--port", "8765"])
    output = capsys.readouterr().out
    token = output.split("Session token: ", 1)[1].splitlines()[0]
    assert len(token) >= 32
    app, options = calls[0]
    assert options["host"] == "127.0.0.1" and options["port"] == 8765
    assert options["access_log"] is False
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/health").status_code == 403
        assert client.get("/health", headers={"X-ModelArena-Token": token}).status_code == 200


def test_real_api_server_binds_loopback_and_accepts_its_session_token(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(os.environ, MODELARENA_HOME=str(tmp_path))
    process = subprocess.Popen([sys.executable, "-m", "apps.api.server", "--port", str(port)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = process.stdout.readline().strip()
        assert line.startswith("Session token: ")
        token = line.removeprefix("Session token: ")
        url = f"http://127.0.0.1:{port}/health"
        deadline = monotonic() + 20
        with httpx.Client(trust_env=False, timeout=1) as client:
            while True:
                try:
                    response = client.get(url, headers={"X-ModelArena-Token": token})
                    break
                except httpx.ConnectError:
                    if process.poll() is not None or monotonic() > deadline:
                        raise
                    sleep(0.1)
            assert response.status_code == 200
            assert response.json()["worker"] == "not_detected"
            assert client.get(url).status_code == 403
    finally:
        process.terminate()
        process.communicate(timeout=15)


def test_real_worker_entrypoint_completes_queued_experiment(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    app = create_app(repo, session_token=TOKEN)
    csv = ('x,target\n' + '\n'.join(f'{i},{i * 2}' for i in range(100))).encode()
    with TestClient(app, base_url="http://localhost:8000", headers={"X-ModelArena-Token": TOKEN}) as client:
        dataset_id = client.post('/api/v1/datasets', files={'file': ('data.csv', csv)}).json()['dataset_id']
        config_id = client.post(f'/api/v1/datasets/{dataset_id}/configurations', json={
            'task': 'regression', 'target': 'target', 'features': [{'name': 'x', 'dtype': 'numeric'}]}).json()['configuration_id']
        experiment_id = client.post('/api/v1/experiments', json={'configuration_id': config_id}).json()['experiment_id']
        job_id = client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'runtime'}).json()['job_id']
        env = dict(os.environ, MODELARENA_HOME=str(tmp_path))
        process = subprocess.Popen([sys.executable, '-m', 'apps.api.workers.runner', '--once'],
                                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        seen_online = False
        try:
            deadline = monotonic() + 90
            while process.poll() is None and monotonic() < deadline:
                seen_online |= client.get('/health').json()['worker'] == 'available'
                sleep(0.2)
            stdout, stderr = process.communicate(timeout=5)
            assert process.returncode == 0, (stdout, stderr)
            assert seen_online
            assert client.get(f'/api/v1/jobs/{job_id}').json()['state'] == 'completed'
            assert len(client.get(f'/api/v1/experiments/{experiment_id}/leaderboard').json()['ranked']) == 5
            assert client.get('/health').json()['worker'] == 'not_detected'
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=15)


def test_worker_once_exit_status_reports_failed_job(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN)
    csv = ('x,target\n' + '\n'.join(f'{i},{i * 2}' for i in range(100))).encode()
    with TestClient(app, base_url='http://localhost:8000', headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id = client.post('/api/v1/datasets', files={'file': ('data.csv', csv)}).json()['dataset_id']
        config_id = client.post(f'/api/v1/datasets/{dataset_id}/configurations', json={
            'task': 'regression', 'target': 'target', 'features': [{'name': 'x', 'dtype': 'numeric'}]}).json()['configuration_id']
        experiment_id = client.post('/api/v1/experiments', json={'configuration_id': config_id}).json()['experiment_id']
        job_id = client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'failure'}).json()['job_id']
        (tmp_path / 'uploads' / dataset_id / 'source.csv').write_bytes(b'tampered\nvalue\n')
        result = subprocess.run([sys.executable, '-m', 'apps.api.workers.runner', '--once'],
                                env=dict(os.environ, MODELARENA_HOME=str(tmp_path)), capture_output=True, text=True, timeout=40)
        assert result.returncode != 0
        assert client.get(f'/api/v1/jobs/{job_id}').json()['state'] == 'failed'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows console control acceptance')
def test_windows_worker_ctrl_c_cleans_up_presence(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    process = subprocess.Popen([sys.executable, '-m', 'apps.api.workers.runner'],
                               env=dict(os.environ, MODELARENA_HOME=str(tmp_path)),
                               creationflags=subprocess.CREATE_NEW_CONSOLE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worker = None
    try:
        deadline = monotonic() + 30
        while monotonic() < deadline:
            with repo._connection() as db:
                worker = db.execute('SELECT pid,process_created FROM worker_presence').fetchone()
            if worker:
                break
            assert process.poll() is None
            sleep(0.1)
        assert worker is not None
        helper = ("import ctypes,sys,time; k=ctypes.WinDLL('kernel32',use_last_error=True); "
                  "k.FreeConsole(); assert k.AttachConsole(int(sys.argv[1])); "
                  "assert k.SetConsoleCtrlHandler(None,True); "
                  "assert k.GenerateConsoleCtrlEvent(0,0); time.sleep(0.5); k.FreeConsole()")
        sent = subprocess.run([sys.executable, '-c', helper, str(worker['pid'])],
                              capture_output=True, text=True, timeout=10)
        assert sent.returncode == 0, sent.stderr
        assert process.wait(timeout=20) == 130
        with repo._connection() as db:
            assert db.execute('SELECT count(*) FROM worker_presence WHERE pid=?', (worker['pid'],)).fetchone()[0] == 0
    finally:
        if worker:
            terminate_owned_child(worker['pid'], creation_time=worker['process_created'], timeout=2)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows console control acceptance')
def test_windows_api_server_ctrl_c_releases_listener(tmp_path):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([sys.executable, '-m', 'apps.api.server', '--port', str(port)],
                               env=dict(os.environ, MODELARENA_HOME=str(tmp_path)),
                               creationflags=subprocess.CREATE_NEW_CONSOLE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    child = None
    try:
        token_line = process.stdout.readline().strip()
        assert token_line.startswith('Session token: ')
        token = token_line.removeprefix('Session token: ')
        deadline = monotonic() + 20
        with httpx.Client(trust_env=False, timeout=1) as client:
            while True:
                try:
                    assert client.get(f'http://127.0.0.1:{port}/health', headers={'X-ModelArena-Token': token}).status_code == 200
                    break
                except httpx.ConnectError:
                    assert monotonic() < deadline and process.poll() is None
                    sleep(0.1)
        children = psutil.Process(process.pid).children(recursive=True)
        child = children[-1] if children else psutil.Process(process.pid)
        child_identity = (child.pid, child.create_time())
        helper = ("import ctypes,sys,time; k=ctypes.WinDLL('kernel32',use_last_error=True); "
                  "k.FreeConsole(); assert k.AttachConsole(int(sys.argv[1])); "
                  "assert k.SetConsoleCtrlHandler(None,True); "
                  "assert k.GenerateConsoleCtrlEvent(0,0); time.sleep(0.5); k.FreeConsole()")
        sent = subprocess.run([sys.executable, '-c', helper, str(child.pid)], capture_output=True, text=True, timeout=10)
        assert sent.returncode == 0, sent.stderr
        assert process.wait(timeout=20) == 0
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', port))
    finally:
        if child:
            terminate_owned_child(child_identity[0], creation_time=child_identity[1], timeout=2)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows console control acceptance')
def test_windows_ctrl_c_interrupts_real_spawned_worker_job(tmp_path):
    repo = JobRepository(tmp_path / 'metadata.sqlite3')
    app = create_app(repo, session_token=TOKEN)
    csv = ('x,target\n' + '\n'.join(f'{i},{i * 2}' for i in range(100))).encode()
    with TestClient(app, base_url='http://localhost:8000', headers={'X-ModelArena-Token': TOKEN}) as client:
        dataset_id = client.post('/api/v1/datasets', files={'file': ('data.csv', csv)}).json()['dataset_id']
        config_id = client.post(f'/api/v1/datasets/{dataset_id}/configurations', json={
            'task': 'regression', 'target': 'target', 'features': [{'name': 'x', 'dtype': 'numeric'}]}).json()['configuration_id']
        experiment_id = client.post('/api/v1/experiments', json={'configuration_id': config_id}).json()['experiment_id']
        job_id = client.post(f'/api/v1/experiments/{experiment_id}/jobs', json={'idempotency_key': 'ctrl-c'}).json()['job_id']
        process = subprocess.Popen([sys.executable, '-m', 'apps.api.workers.runner'],
                                   env=dict(os.environ, MODELARENA_HOME=str(tmp_path)),
                                   creationflags=subprocess.CREATE_NEW_CONSOLE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        worker = child = None
        try:
            deadline = monotonic() + 30
            while monotonic() < deadline:
                with repo._connection() as db:
                    worker = db.execute('SELECT pid,process_created FROM worker_presence').fetchone()
                    child = db.execute('SELECT child_pid,child_created FROM worker_processes WHERE child_pid IS NOT NULL').fetchone()
                if worker and child:
                    break
                assert process.poll() is None
                sleep(0.1)
            assert worker is not None and child is not None
            helper = ("import ctypes,sys,time; k=ctypes.WinDLL('kernel32',use_last_error=True); "
                      "k.FreeConsole(); assert k.AttachConsole(int(sys.argv[1])); "
                      "assert k.SetConsoleCtrlHandler(None,True); "
                      "assert k.GenerateConsoleCtrlEvent(0,0); time.sleep(0.5); k.FreeConsole()")
            sent = subprocess.run([sys.executable, '-c', helper, str(worker['pid'])], capture_output=True, text=True, timeout=10)
            assert sent.returncode == 0, sent.stderr
            assert process.wait(timeout=25) == 130
            assert repo.get_job(job_id).state == 'interrupted'
            with repo._connection() as db:
                assert db.execute('SELECT active_job_id FROM worker_lease').fetchone()[0] is None
                assert db.execute('SELECT count(*) FROM worker_presence WHERE pid=?', (worker['pid'],)).fetchone()[0] == 0
        finally:
            if child:
                terminate_owned_child(child['child_pid'], creation_time=child['child_created'], timeout=2)
            if worker:
                terminate_owned_child(worker['pid'], creation_time=worker['process_created'], timeout=2)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
