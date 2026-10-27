"""Single-slot supervisor: claim, spawn one child, heartbeat, monitor, publish.

Boundaries that matter for correctness:

* Claiming is one short SQLite transaction; the singleton lease and the job row
  are updated together, so a second supervisor observes a held slot instead of
  starting a second fit.
* The child is a real spawned process (``multiprocessing`` ``spawn``). Nothing
  numerical is imported in the parent, so the API process never trains and a
  Windows child cannot inherit an unexpected thread pool.
* Heartbeats and resource samples come from the supervisor, not from the child,
  so a silent native fit still looks alive and a hung child is still measured.
* Terminal publication requires the verified attempt owner; a stale worker
  cannot overwrite a result.
* Every persisted string is allowlisted. Exception text, paths and row values
  are never stored or logged.

Cancellation uses a process Event with a bounded grace period. Restart
reconciliation requires verified process death before fencing and releasing the
slot; an unobservable child is never killed by a stored PID alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing
import os
from queue import Empty
from time import perf_counter, sleep
from typing import Any, Mapping

from apps.api.database.repository import (
    ConflictError, JobRecord, JobRepository, WorkerIdentity, render_telemetry_event,
)
from apps.api.workers import control, execute, monitor
from ml.contracts import ResourceLimits

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_SECONDS = 2.0
DEFAULT_SAMPLE_SECONDS = 0.5
DEFAULT_POLL_SECONDS = 0.1
DEFAULT_CANCEL_SECONDS = 0.2
#: Grace before a breached or unresponsive child is terminated, then killed.
TERMINATE_TIMEOUT_SECONDS = 5.0
KILL_TIMEOUT_SECONDS = 2.0

#: Telemetry keys copied from the sampler into the persisted audit event.
_TELEMETRY_KEYS = (
    "wall_seconds", "cpu_seconds", "normalized_cpu_percent", "logical_cores",
    "peak_rss_bytes", "minimum_available_bytes", "sample_count", "sample_interval_seconds",
)

#: Stages the child may report. Terminal stages are supervisor-owned.
_CHILD_STAGES = control.WORKER_STAGES - control.TERMINAL_STATES


@dataclass(frozen=True, slots=True)
class _Outcome:
    state: str
    stage: str
    code: str | None


class WorkerSupervisor:
    """Claims at most one job per :meth:`run_once` call and never overlaps."""

    def __init__(
        self,
        repository: JobRepository,
        *,
        limits: ResourceLimits | None = None,
        workload: str = "probe",
        workload_options: Mapping[str, Any] | None = None,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        owner_token: str = "supervisor",
        generation: int | None = None,
    ) -> None:
        if not isinstance(repository, JobRepository):
            raise TypeError("JobRepository required")
        limits = limits if limits is not None else ResourceLimits()
        if not isinstance(limits, ResourceLimits):
            raise ValueError("limits must be ResourceLimits")
        self.repository = repository
        self.limits = limits
        self.workload = control.check_workload(workload)
        self.options = control.validate_options(self.workload, dict(workload_options or {}))
        self.heartbeat_seconds = _interval(heartbeat_seconds, "heartbeat", 0.05, 60)
        self.sample_seconds = monitor.check_interval(sample_seconds)
        self.poll_seconds = _interval(poll_seconds, "poll", 0.01, 5)
        # Adopt the persisted generation so fencing matches the stored lease.
        self.owner = WorkerIdentity(
            owner_token,
            repository.lease_generation() if generation is None else generation,
        )
        self.last_child_pid: int | None = None
        self.child_exit_code: int | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_resources: dict[str, Any] | None = None
        self._failure_seen = False
        self._child_created: float | None = None

    # -- public API ---------------------------------------------------------
    def run_once(self) -> JobRecord | None:
        """Claim the oldest queued job, execute it in a child and publish a result."""
        self.last_child_pid = None
        self.child_exit_code = None
        self.last_result = None
        self.last_resources = None
        self._failure_seen = False
        self._child_created = None
        if not self.repository.reconcile_worker():
            return None
        # Recovery may have advanced the fencing generation after construction.
        self.owner = WorkerIdentity(self.owner.owner_token, self.repository.lease_generation())
        claimed = self.repository.claim_next(
            self.owner, supervisor_process=(os.getpid(), monitor.process_creation_time(os.getpid())),
        )
        if claimed is None:
            return None
        if not claimed.attempt_id:  # pragma: no cover - claim always stamps one
            raise RuntimeError("Claimed job has no attempt identity")
        logger.info("Supervisor claimed job %s", claimed.job_id)
        return self._execute(claimed)

    # -- execution ----------------------------------------------------------
    def _execute(self, job: JobRecord) -> JobRecord:
        assert job.attempt_id is not None
        try:
            self.repository.record_progress(self.owner, job.job_id, "starting")
            outcome = self._spawn_and_wait(job, job.attempt_id)
        except ConflictError:
            # The slot moved underneath us; publish nothing.
            logger.warning("Ownership lost for job %s; no result published", job.job_id)
            return self.repository.get_job(job.job_id)
        except KeyboardInterrupt:
            outcome = _Outcome("interrupted", "interrupted", "supervisor_stopped")
        except Exception:
            # Only a closed code crosses this boundary; no exception text.
            logger.warning("Supervisor failure while executing job %s", job.job_id)
            outcome = _Outcome("failed", "failed", "supervisor_error")
        return self._publish(job, job.attempt_id, outcome)

    def _publish(self, job: JobRecord, attempt_id: str, outcome: _Outcome) -> JobRecord:
        # Telemetry is diagnostics, not the result, so it is validated before the
        # terminal transaction. A value the audit trail cannot render must not
        # abort publication and strand the job in ``running`` holding the slot.
        telemetry = self._telemetry()
        try:
            render_telemetry_event(telemetry)  # Validate only; the repository renders again.
        except ValueError:
            logger.warning("Dropping unrenderable telemetry for job %s", job.job_id)
            telemetry = None
        try:
            final = self.repository.finish_job(
                self.owner, job.job_id, attempt_id, state=outcome.state,
                stage=outcome.stage, safe_error=outcome.code, telemetry=telemetry,
            )
            if final.state != "completed":
                self.last_result = None
            return final
        except ConflictError:
            logger.warning("Stale supervisor refused publication for job %s", job.job_id)
            return self.repository.get_job(job.job_id)

    def _telemetry(self) -> dict[str, object]:
        # ``get`` rather than ``[]``: a sampler summary missing a key must not
        # abort publication and leave the job stuck in ``running``.
        telemetry: dict[str, object] = {}
        if self.last_resources:
            for key in _TELEMETRY_KEYS:
                if key in self.last_resources:
                    telemetry[key] = self.last_resources[key]
        if self.last_child_pid is not None:
            telemetry["child_pid"] = self.last_child_pid
        if self.child_exit_code is not None:
            telemetry["child_exit_code"] = self.child_exit_code
        return telemetry

    def _spawn_and_wait(self, job: JobRecord, attempt_id: str) -> _Outcome:
        context = multiprocessing.get_context("spawn")
        messages = context.Queue()
        self._cancel_event = context.Event()
        start_event = context.Event()
        process = context.Process(
            # The spawn target must live in ``execute`` so the fresh interpreter
            # imports only the light control module. A supervisor helper would
            # make the child import the repository, sklearn and OpenMP before
            # ``apply_thread_environment`` could bound any pool.
            target=execute.child_main,
            args=(job.job_id, attempt_id, self._child_limits(), self.workload,
                  dict(self.options), messages, self._cancel_event, start_event),
            name=f"modelarena-exec-{job.job_id[:8]}",
            daemon=False,
        )
        sampler: monitor.ChildSampler | None = None
        try:
            process.start()
            self.last_child_pid = process.pid
            self._child_created = monitor.process_creation_time(process.pid)
            self.repository.record_child(self.owner, job.job_id, attempt_id,
                                         process.pid, self._child_created)
            start_event.set()
            logger.info("Spawned execution child pid=%s", process.pid)
            # A child that dies before psutil can read its creation time stays
            # unsampled instead of aborting the run: monitoring then continues
            # without telemetry and the exit code alone decides the outcome.
            try:
                creation_time = monitor.process_creation_time(process.pid)
                sampler = monitor.ChildSampler(
                    process.pid, creation_time=creation_time, interval_seconds=self.sample_seconds,
                ).start()
            except RuntimeError:
                sampler = None
            outcome = self._monitor(process, messages, job, sampler)
        except KeyboardInterrupt:
            self._cancel_event.set()
            if process.pid is not None:
                process.join(5.0)
            raise
        finally:
            self._shutdown(process, messages, sampler)
        return outcome

    def _shutdown(self, process, messages, sampler) -> None:
        """Join the owned child, terminating only a verified identity."""
        # ``pid`` stays None when ``start`` itself failed, and joining an
        # unstarted process asserts, so that case must not reach ``join``.
        if process.pid is not None:
            if process.is_alive():
                creation_time = self._child_created
                if creation_time is not None:
                    monitor.terminate_owned_child(
                        process.pid, creation_time=creation_time,
                        timeout=KILL_TIMEOUT_SECONDS,
                    )
                process.join(0)
                if (process.is_alive() and creation_time is not None
                        and monitor.process_liveness(process.pid, creation_time) == "alive"):
                    process.kill()
                    process.join(KILL_TIMEOUT_SECONDS)
            else:
                process.join(KILL_TIMEOUT_SECONDS)
            self.child_exit_code = process.exitcode
        if sampler is not None:
            sampler.stop()
            try:
                self.last_resources = sampler.summary()
            except RuntimeError:  # pragma: no cover - child vanished instantly
                self.last_resources = None
        try:
            messages.close()
            messages.join_thread()
        except (OSError, ValueError):  # pragma: no cover - pipe already torn down
            pass
        if process.pid is not None and process.is_alive():
            # Never free the slot merely because cleanup failed. Reconciliation
            # requires proof of death, including when identity was inaccessible.
            raise ConflictError("Execution child is still alive or unverified")

    def _child_limits(self) -> dict[str, object]:
        """Only settings the child actually applies; it never enforces its own cap.

        The supervisor owns RSS/system-memory enforcement, so sending those
        ceilings into the child would imply a guarantee the child does not make.
        """
        return {"compute_threads": self.limits.compute_threads}

    def _monitor(self, process, messages, job: JobRecord, sampler) -> _Outcome:
        started = perf_counter()
        last_heartbeat = last_sample = started
        # Cancellation has its own bounded cadence: even a slow heartbeat or
        # configured monitoring poll must not postpone signalling user intent.
        last_cancel_check = started - DEFAULT_CANCEL_SECONDS
        pressure = monitor.MemoryPressureTracker(
            self.limits.system_reserve_bytes, self.limits.memory_pressure_seconds,
        )
        stage = "starting"
        cancel_started = None
        while True:
            now = perf_counter()
            # Drain all pending child messages before deciding anything.
            while True:
                try:
                    message = messages.get_nowait()
                except Empty:
                    break
                except (OSError, EOFError):  # pragma: no cover - pipe teardown race
                    break
                reported = message.get("stage")
                if message.get("kind") == "progress" and reported in _CHILD_STAGES:
                    if reported != stage:
                        self.repository.record_progress(self.owner, job.job_id, reported)
                        stage = reported
                elif message.get("kind") == "result":
                    payload = message.get("payload")
                    if isinstance(payload, dict):
                        self.last_result = payload
                elif message.get("kind") == "failure":
                    self._failure_seen = True

            # Cancel before considering a received result. The final database
            # transaction arbitrates requests that arrive after this check.
            if now - last_cancel_check >= DEFAULT_CANCEL_SECONDS:
                last_cancel_check = now
                if self.repository.cancel_requested(job.job_id) and cancel_started is None:
                    cancel_started = now
                    self._cancel_event.set()
                    self.last_result = None

            if cancel_started is not None:
                self.last_result = None
                if process.exitcode is not None or now - cancel_started >= 5.0:
                    return _Outcome("cancelled", "cancelled", "cancelled")
                sleep(min(self.poll_seconds, DEFAULT_CANCEL_SECONDS))
                continue

            if self.last_result is not None:
                process.join(TERMINATE_TIMEOUT_SECONDS)
                return _Outcome("completed", "completed", None)

            if process.exitcode is not None:
                code = "workload_failed" if self._failure_seen else "child_exited"
                state = "failed" if self._failure_seen else "interrupted"
                return _Outcome(state, state, code)

            if now - started > self.limits.cv_job_seconds:
                return _Outcome("failed", "failed", "timeout")

            if sampler is not None and now - last_sample >= self.sample_seconds:
                try:
                    sampler.sample()
                except RuntimeError:  # pragma: no cover - child vanished
                    sampler = None
                last_sample = now
            if sampler is not None:
                latest = sampler.latest
                if latest is not None:
                    if int(latest["rss_bytes"]) > self.limits.child_rss_bytes:
                        return _Outcome("failed", "failed", "memory_limit_exceeded")
                    if pressure.update(int(latest["available_bytes"]), now=now):
                        return _Outcome("failed", "failed", "resource_pressure")

            if now - last_heartbeat >= self.heartbeat_seconds:
                self.repository.heartbeat(self.owner, job.job_id)
                last_heartbeat = now

            sleep(min(self.poll_seconds, DEFAULT_CANCEL_SECONDS))


def _interval(value: object, name: str, minimum: float, maximum: float) -> float:
    if (type(value) not in (int, float) or isinstance(value, bool)
            or not minimum <= value <= maximum):
        raise ValueError(f"Invalid {name} interval")
    return float(value)
