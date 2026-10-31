"""Watchdog control messages are not interchangeable with UI progress."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from queue import Queue
from threading import Event
from time import sleep

import pytest

from apps.api.workers.supervisor import WorkerSupervisor
from ml.contracts import ResourceLimits, RunConfig, TaskType
from tests.integration.test_classical_suite import prepared_fixture


class LiveProcess:
    exitcode = None

    def join(self, timeout=None):
        pass


@pytest.fixture
def monitor_setup(tmp_path):
    run = RunConfig(limits=ResourceLimits(mlp_fold_seconds=1, cv_job_seconds=4))
    repo, prepared, _, _ = prepared_fixture(tmp_path, TaskType.REGRESSION, run)
    worker = WorkerSupervisor(repo, workload="classical", classical_input=(prepared, run),
                              limits=run.limits, poll_seconds=0.02, heartbeat_seconds=0.2)
    job = repo.claim_next(worker.owner)
    worker._cancel_event = Event()
    parent, child = multiprocessing.get_context("spawn").Pipe()
    worker._fold_pipe = parent
    yield worker, job, parent, child
    parent.close()
    child.close()


def test_start_and_finish_queued_together_cannot_forge_ack(monitor_setup):
    worker, job, _, child = monitor_setup
    child.send((0, False, None))
    child.send((0, True, None))  # no opportunity to receive the supervisor's challenge
    outcome = worker._monitor(LiveProcess(), Queue(), job, None)
    assert (outcome.state, outcome.code) == ("failed", "timeout")


@pytest.mark.parametrize("send_finish", [False, True])
def test_lost_or_delayed_finish_times_out_even_without_progress(monitor_setup, send_finish):
    worker, job, _, child = monitor_setup
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker._monitor, LiveProcess(), Queue(), job, None)
        child.send((0, False, None))
        _, _, proof = child.recv()
        sleep(1.2)
        if send_finish:
            child.send((0, True, proof))
        outcome = future.result(timeout=3)
    assert (outcome.state, outcome.code) == ("failed", "timeout")
    assert worker.last_result is None


def test_missing_start_cannot_publish_result(monitor_setup):
    worker, job, _, _ = monitor_setup
    messages = Queue()
    messages.put({"kind": "result", "payload": {"candidates": []}})
    outcome = worker._monitor(LiveProcess(), messages, job, None)
    assert (outcome.state, outcome.code) == ("failed", "timeout")
    assert worker.last_result is None


def test_channel_close_after_cancel_is_not_a_timeout(monitor_setup):
    worker, job, _, child = monitor_setup
    worker.repository.request_cancel(job.job_id)
    child.close()
    outcome = worker._monitor(LiveProcess(), Queue(), job, None)
    assert (outcome.state, outcome.code) == ("cancelled", "cancelled")
