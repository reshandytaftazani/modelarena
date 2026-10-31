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
import secrets
from queue import Empty
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any, Mapping

from apps.api.database.repository import (
    ConflictError, JobRecord, JobRepository, WorkerIdentity, render_telemetry_event,
)
from apps.api.workers import control, execute, monitor
from ml.contracts import ResourceLimits

if TYPE_CHECKING:
    from apps.api.services.jobs import PreparedExperiment
    from ml.contracts import RunConfig

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
        options: Mapping[str, Any] | None = None,
        classical_input: tuple[PreparedExperiment, RunConfig] | None = None,
        load_from_storage: bool = False,
        claim_all: bool = False,
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
        effective_options = workload_options if workload_options is not None else options
        self.options = control.validate_options(self.workload, dict(effective_options or {}))
        if type(load_from_storage) is not bool or (load_from_storage and (self.workload not in ("classical", "finalization") or classical_input is not None)):
            raise ValueError("Stored input is available only for the classical workload")
        if type(claim_all) is not bool or (claim_all and (self.workload != "classical" or not load_from_storage)):
            raise ValueError("Mixed job dispatch requires the stored-input worker")
        if self.workload == "classical" and classical_input is None and not load_from_storage:
            raise ValueError("Classical workload requires trusted prepared input")
        if self.workload not in ("classical", "finalization") and classical_input is not None:
            raise ValueError("Classical workload requires trusted prepared input")
        self.classical_input = classical_input
        self.load_from_storage = load_from_storage
        self.claim_all = claim_all
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
        self._fold_pipe = None
        self._evaluation_pipe = None
        self._staged_finalization = None
        self.finalization_input = None

    # -- public API ---------------------------------------------------------
    def run_once(self) -> JobRecord | None:
        """Claim the oldest queued job, execute it in a child and publish a result."""
        self.last_child_pid = None
        self.child_exit_code = None
        self.last_result = None
        self.last_resources = None
        self._failure_seen = False
        self._child_created = None
        self._evaluation_pipe = None
        self._staged_finalization = None
        self.finalization_input = None
        if not self.repository.reconcile_worker():
            return None
        from ml.tracking.reconcile import reconcile_orphans
        try:
            reconcile_orphans(self.repository)
        except Exception:
            # Do not start another attempt while cross-store recovery is unknown.
            logger.warning("Tracking reconciliation unavailable; worker remains idle")
            return None
        # Recovery may have advanced the fencing generation after construction.
        self.owner = WorkerIdentity(self.owner.owner_token, self.repository.lease_generation())
        claimed = self.repository.claim_next(
            self.owner, supervisor_process=(os.getpid(), monitor.process_creation_time(os.getpid())),
            kind=(None if self.claim_all
                  else "finalization" if self.workload == "finalization" else "cv"),
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
            if job.kind == "finalization":
                import json
                from apps.api.services.workflow import load_refit_experiment
                from ml.contracts import FrozenSelection, ResourceLimits, TaskType
                from ml.training.refit import candidate_from_dict, freeze_refit
                with self.repository._connection() as db:
                    fin = db.execute("SELECT * FROM finalizations WHERE refit_job_id=?", (job.job_id,)).fetchone()
                    if fin is None:
                        raise ValueError("Finalization not found")
                    fin = dict(fin)
                    cand_row = db.execute("SELECT * FROM candidates WHERE candidate_id=?", (fin["candidate_id"],)).fetchone()
                    scope = db.execute("SELECT task FROM evaluation_scopes WHERE scope_id=?", (fin["scope_id"],)).fetchone()
                    exp = db.execute("SELECT primary_metric, direction, run_json FROM experiments WHERE experiment_id=?", (fin["experiment_id"],)).fetchone()
                bundle = self.repository.read_published_results(cand_row["job_id"])
                cand_data = next(c for c in bundle.get("candidates", []) if c.get("family") == cand_row["family"])
                candidate_result = candidate_from_dict(cand_data)
                prepared, run = load_refit_experiment(self.repository, job.experiment_id)
                if run.limits != self.limits:
                    raise ValueError("Finalization resource limits differ from worker limits")
                run_dict = json.loads(exp["run_json"])
                limits = ResourceLimits(**run_dict.get("limits", {}))
                score = next(s["mean"] for s in cand_data["summary"] if s["name"] == exp["primary_metric"])
                selection = FrozenSelection(
                    finalization_id=fin["finalization_id"], scope_id=fin["scope_id"],
                    experiment_id=fin["experiment_id"], candidate_id=fin["candidate_id"],
                    candidate_digest=fin["candidate_digest"], family=cand_row["family"],
                    task=TaskType(scope["task"]), primary_metric=exp["primary_metric"],
                    primary_score=float(score), direction=exp["direction"],
                    split_sha256=prepared.context.split.manifest_sha256, config_sha256=cand_row["config_sha256"],
                    model_seed=cand_data.get("model_seed", 42), resource_limits=limits,
                    created_at=fin["created_at"],
                )
                plan = freeze_refit(prepared.context, selection, candidate_result)
                if fin.get("evidence_sha256") and plan.evidence_sha256 != fin["evidence_sha256"]:
                    raise ValueError("Finalization evidence mismatch")
                self.finalization_input = (prepared.context, plan, candidate_result, fin, str(self.repository.path))
            elif self.workload == "classical":
                from apps.api.services.jobs import PreparedExperiment
                from ml.contracts import RunConfig
                try:
                    if self.load_from_storage:
                        from apps.api.services.workflow import load_experiment
                        self.classical_input = load_experiment(self.repository, job.experiment_id)
                    prepared, run = self.classical_input
                    if (not isinstance(prepared, PreparedExperiment) or not isinstance(run, RunConfig)
                            or prepared.experiment_id != job.experiment_id or run.limits != self.limits):
                        raise ValueError("Classical input does not match job")
                    self.repository.verify_classical_input(job.experiment_id, prepared.context, run)
                except (ConflictError, TypeError, ValueError) as exc:
                    # Input mismatch is a failed job, not loss of worker ownership.
                    raise ValueError("Invalid classical input") from exc
            outcome = self._spawn_and_wait(job, job.attempt_id)
        except ConflictError:
            # The slot moved underneath us; publish nothing.
            self.last_result = None
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
        publication = None
        if job.kind == "finalization":
            if outcome.state == "completed" and (self._staged_finalization or (self.last_result and self.last_result.get("result_path"))):
                if not self._staged_finalization and self.last_result:
                    self._staged_finalization = (self.last_result["result_path"], self.last_result["result_checksum"], self.last_result["finalization_id"])
                result_path, result_checksum, finalization_id = self._staged_finalization
                return self.repository.publish_finalization(
                    self.owner, job.job_id, attempt_id, finalization_id, result_path, result_checksum,
                    telemetry=telemetry,
                )
            else:
                from apps.api.database.repository import _now
                with self.repository._transaction() as db:
                    current = self.repository._owned_running_row(db, self.owner, job.job_id)
                    if current["attempt_id"] != attempt_id:
                        raise ConflictError("Attempt does not own this job")
                    fin = db.execute("SELECT * FROM finalizations WHERE refit_job_id=?", (job.job_id,)).fetchone()
                    if fin and fin["evaluation_started_at"] is not None:
                        if current["cancel_requested"] or not self.repository._try_recover_staged_final_bundle(db, fin, attempt_id, _now()):
                            db.execute("UPDATE finalizations SET state='locked_failed',updated_at=? WHERE finalization_id=?", (_now(), fin["finalization_id"]))
                            db.execute("UPDATE jobs SET state=?,stage=?,safe_error=?,updated_at=? WHERE job_id=?", (outcome.state, outcome.stage, outcome.code, _now(), job.job_id))
                            db.execute("UPDATE worker_lease SET owner_token=NULL,active_job_id=NULL WHERE singleton=1")
                        else:
                            db.execute("UPDATE worker_lease SET owner_token=NULL,active_job_id=NULL WHERE singleton=1")
                    elif fin:
                        db.execute("UPDATE finalizations SET state='retryable_pre_evaluation',updated_at=? WHERE finalization_id=?", (_now(), fin["finalization_id"]))
                        db.execute("UPDATE jobs SET state=?,stage=?,safe_error=?,updated_at=? WHERE job_id=?", (outcome.state, outcome.stage, outcome.code, _now(), job.job_id))
                        db.execute("UPDATE worker_lease SET owner_token=NULL,active_job_id=NULL WHERE singleton=1")
                return self.repository.get_job(job.job_id)
        if outcome.state == "completed" and self.workload == "classical":
            try:
                from ml.tracking.publication import stage_attempt
                prepared, run = self.classical_input
                publication = stage_attempt(self.repository.path.parent, job, prepared.context, run,
                                            self.repository.tracking_dataset(job.experiment_id),
                                            self.last_result, telemetry)
            except Exception:
                logger.warning("Tracking or bundle staging failed for job %s", job.job_id)
                self.last_result = None
                outcome = _Outcome("failed", "failed", "supervisor_error")
        try:
            final = self.repository.finish_job(
                self.owner, job.job_id, attempt_id, state=outcome.state,
                stage=outcome.stage, safe_error=outcome.code, telemetry=telemetry,
                publication=publication,
            )
            if final.state != "completed":
                self.last_result = None
            return final
        except ConflictError:
            self.last_result = None
            logger.warning("Stale supervisor refused publication for job %s", job.job_id)
            return self.repository.get_job(job.job_id)
        except Exception:
            # A failed DB transaction cannot publish partial candidate rows. Try
            # to close the attempt as failed; if the store remains unavailable,
            # preserve the running lease for verified-death reconciliation.
            self.last_result = None
            logger.warning("Result publication failed for job %s", job.job_id)
            try:
                return self.repository.finish_job(self.owner, job.job_id, attempt_id,
                                                  state="failed", stage="failed",
                                                  safe_error="supervisor_error", telemetry=telemetry)
            except Exception:
                logger.warning("Could not persist publication failure for job %s", job.job_id)
                raise

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
        if job.kind == "finalization":
            parent_fold, child_fold = None, None
            parent_eval, child_eval = context.Pipe()
            self._fold_pipe = None
            self._evaluation_pipe = parent_eval
            target_workload = "finalization"
            child_kwargs = {
                "finalization_input": self.finalization_input,
                "evaluation_channel": child_eval,
            }
        else:
            parent_eval, child_eval = None, None
            self._evaluation_pipe = None
            parent_fold, child_fold = context.Pipe() if self.workload == "classical" else (None, None)
            self._fold_pipe = parent_fold
            target_workload = self.workload
            child_kwargs = ({"classical_input": (self.classical_input[0].context, self.classical_input[1]), "fold_channel": child_fold}
                            if self.classical_input is not None else {})
        self._cancel_event = context.Event()
        start_event = context.Event()
        process = context.Process(
            target=execute.child_main,
            args=(job.job_id, attempt_id, self._child_limits(), target_workload,
                  dict(self.options), messages, self._cancel_event, start_event),
            kwargs=child_kwargs,
            name=f"modelarena-exec-{job.job_id[:8]}",
            daemon=False,
        )
        sampler: monitor.ChildSampler | None = None
        try:
            process.start()
            if child_fold is not None:
                child_fold.close()
            if child_eval is not None:
                child_eval.close()
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
            if parent_fold is not None:
                parent_fold.close()
            if parent_eval is not None:
                parent_eval.close()
            self._fold_pipe = None
            self._evaluation_pipe = None
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
        mlp_active_fold: int | None = None
        mlp_fold_started: float | None = None
        mlp_fold_deadline_exceeded = False
        mlp_completed = 0
        fold_proof = None
        fold_closed = False
        while True:
            now = perf_counter()
            # This is a control channel, not best-effort progress. The child
            # blocks at each boundary until the supervisor acknowledges it.
            if self._evaluation_pipe is not None:
                try:
                    while self._evaluation_pipe.poll():
                        eval_msg = self._evaluation_pipe.recv()
                        if eval_msg[0] == "ready_for_evaluation":
                            if self._cancel_event.is_set() or self.repository.cancel_requested(job.job_id):
                                self._evaluation_pipe.send(("cancelled", None))
                                return _Outcome("cancelled", "cancelled", "cancelled")
                            from apps.api.services.finalization import start_evaluation
                            fin_dict = self.finalization_input[3]
                            try:
                                capability = start_evaluation(
                                    self.repository, fin_dict["finalization_id"], eval_msg[1],
                                    owner=self.owner, job_id=job.job_id, attempt_id=job.attempt_id,
                                )
                                self._evaluation_pipe.send(("capability", capability))
                            except Exception as e:
                                logger.warning("start_evaluation failed: %s", e)
                                self._evaluation_pipe.send(("error", None))
                                return _Outcome("failed", "failed", "evaluation_start_failed")
                        elif eval_msg[0] == "staged":
                            self._staged_finalization = (eval_msg[1], eval_msg[2], self.finalization_input[3]["finalization_id"])
                except EOFError:
                    pass
                except (OSError, TypeError, ValueError):
                    pass
            if self._fold_pipe is not None and not fold_closed:
                try:
                    while self._fold_pipe.poll():
                        fold, finished, proof = self._fold_pipe.recv()
                        now = perf_counter()
                        if type(fold) is not int or type(finished) is not bool:
                            return _Outcome("failed", "failed", "timeout")
                        if not finished:
                            if mlp_active_fold is not None or fold != mlp_completed or proof is not None:
                                return _Outcome("failed", "failed", "timeout")
                            mlp_active_fold, mlp_fold_started = fold, now
                            fold_proof = secrets.token_hex(16)
                        else:
                            if (mlp_active_fold != fold or mlp_fold_started is None
                                    or proof != fold_proof or now - mlp_fold_started >= self.limits.mlp_fold_seconds):
                                return _Outcome("failed", "failed", "timeout")
                            mlp_active_fold, mlp_fold_started = None, None
                            mlp_completed += 1
                        self._fold_pipe.send((fold, finished, fold_proof))
                        if finished:
                            fold_proof = None
                except EOFError:
                    fold_closed = True
                    if mlp_active_fold is not None or mlp_completed != self.classical_input[0].context.split.fold_count:
                        self.last_result = None
                        if (self.repository.cancel_requested(job.job_id) and
                                (mlp_fold_started is None or perf_counter() - mlp_fold_started < self.limits.mlp_fold_seconds)):
                            return _Outcome("cancelled", "cancelled", "cancelled")
                        return _Outcome("failed", "failed", "timeout")
                except (OSError, TypeError, ValueError):
                    self.last_result = None
                    if (self.repository.cancel_requested(job.job_id) and
                            (mlp_fold_started is None or perf_counter() - mlp_fold_started < self.limits.mlp_fold_seconds)):
                        return _Outcome("cancelled", "cancelled", "cancelled")
                    return _Outcome("failed", "failed", "timeout")
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
                    if "family" in message:
                        family = message.get("family")
                        fold_index = message.get("fold_index")
                        finished_units = message.get("finished_units")
                        total_units = message.get("total_units")
                        try:
                            self.repository.record_work_progress(self.owner, job.job_id, family,
                                                                 fold_index, finished_units,
                                                                 total_units)
                        except ValueError:
                            logger.warning("Ignored invalid worker progress for job %s", job.job_id)

                elif message.get("kind") == "result":
                    payload = message.get("payload")
                    if isinstance(payload, dict):
                        self.last_result = payload
                elif message.get("kind") == "failure":
                    self._failure_seen = True

            # Watchdog check for active MLP fold deadline
            now = perf_counter()
            if mlp_active_fold is not None and mlp_fold_started is not None:
                if now - mlp_fold_started >= self.limits.mlp_fold_seconds:
                    mlp_fold_deadline_exceeded = True

            if mlp_fold_deadline_exceeded:
                self.last_result = None
                return _Outcome("failed", "failed", "timeout")

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
                if mlp_fold_deadline_exceeded:
                    self.last_result = None
                    return _Outcome("failed", "failed", "timeout")
                if mlp_active_fold is not None and mlp_fold_started is not None:
                    if now - mlp_fold_started > self.limits.mlp_fold_seconds:
                        self.last_result = None
                        return _Outcome("failed", "failed", "timeout")
                if self.workload == "classical" and job.kind != "finalization":
                    if (self._fold_pipe is None or mlp_active_fold is not None or
                            mlp_completed != self.classical_input[0].context.split.fold_count):
                        self.last_result = None
                        return _Outcome("failed", "failed", "timeout")
                    candidates = self.last_result.get("candidates", [])
                    for candidate in candidates:
                        if candidate.get("family") == "mlp":
                            for fold in candidate.get("folds", []):
                                fit_sec = fold.get("fit_seconds", 0.0)
                                if fit_sec >= self.limits.mlp_fold_seconds or fold.get("status") == "timeout":
                                    self.last_result = None
                                    return _Outcome("failed", "failed", "timeout")
                process.join(TERMINATE_TIMEOUT_SECONDS)
                return _Outcome("completed", "completed", None)

            if process.exitcode is not None:
                if mlp_fold_deadline_exceeded:
                    self.last_result = None
                    return _Outcome("failed", "failed", "timeout")
                code = "workload_failed" if self._failure_seen else "child_exited"
                state = "failed" if self._failure_seen else "interrupted"
                return _Outcome(state, state, code)

            timeout = self.limits.finalization_seconds if job.kind == "finalization" else self.limits.cv_job_seconds
            if now - started > timeout:
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
