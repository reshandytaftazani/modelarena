"""Child entrypoint for one execution. Import-safe: no numeric library at scope.

A freshly spawned Windows child imports this module before any work starts, so
module scope stays cheap and side-effect free. Thread environment variables are
applied inside :func:`child_main` before the first numerical import, and each
stage enables exactly one T-wide compute layer:

* BLAS stage: BLAS up to T, OpenMP pinned to one.
* Torch stage: torch intra-op T, inter-op one, BLAS and OpenMP pinned to one.

Reporting the pools per stage is how the tests prove there is no nested T-wide
pair. The ``probe`` workload exercises the process without user data. The
classical workload receives an internally prepared training-only context; it
never opens a dataset, a holdout partition or a user-supplied path.
"""

import os
import time
from typing import Any, Mapping

from apps.api.workers.control import validate_options

PROGRESS = "progress"
RESULT = "result"
FAILURE = "failure"

_BLAS_ENV = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def apply_thread_environment(compute_threads: int) -> None:
    """Set pool ceilings before the first numerical import of this process."""
    if type(compute_threads) is not int or compute_threads not in (2, 4, 6):
        raise ValueError("Compute threads must be 2, 4 or 6")
    # OpenMP stays at one so BLAS can own the only T-wide layer by default.
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in _BLAS_ENV:
        os.environ[name] = str(compute_threads)


def _emit(queue, kind: str, **payload: Any) -> None:
    """Best-effort progress; a vanished parent must never crash the child."""
    try:
        queue.put({"kind": kind, **payload}, timeout=5)
    except Exception:
        pass


def _pool_report(pools) -> list[dict[str, Any]]:
    return [
        {"user_api": str(pool["user_api"]), "internal_api": str(pool["internal_api"]),
         "num_threads": int(pool["num_threads"])}
        for pool in pools
    ]


def _probe(options: Mapping[str, Any], queue, compute_threads: int, cancel_event=None) -> dict[str, Any]:
    """Bounded synthetic work; never touches user data."""
    _emit(queue, PROGRESS, stage="preparing")
    allocation = None
    if options["allocate_mb"]:
        allocation = bytearray(options["allocate_mb"] * 1024 * 1024)
    _emit(queue, PROGRESS, stage="fitting")

    steps = options["steps"]
    per_step = float(options["work_seconds"]) / steps
    for _ in range(steps):
        if cancel_event is not None and cancel_event.wait(per_step):
            return {}
        if cancel_event is None and per_step > 0:
            time.sleep(per_step)
        _emit(queue, PROGRESS, stage="fitting")

    if options["silent_seconds"] > 0:
        time.sleep(float(options["silent_seconds"]))

    if options["raise_message"]:
        # Deliberately unhandled below: the message must never reach the
        # supervisor, a log record or a persisted column.
        raise RuntimeError(options["raise_message"])

    result: dict[str, Any] = {
        "pid": os.getpid(), "parent_pid": os.getppid(), "steps": steps,
        "compute_threads": compute_threads, "workload": "probe",
        "torch_intra_threads": None, "torch_interop_threads": None,
        "threadpools": {}, "wide_layers": {},
    }
    if options["report_pools"]:
        result.update(_measure_pools(compute_threads))
    del allocation
    return result


def _measure_pools(compute_threads: int) -> dict[str, Any]:
    """Observe each stage separately so nesting cannot hide inside one snapshot.

    torch's intra-op pool defaults to the physical core count and ignores
    ``OMP_NUM_THREADS``, so it is pinned to one before the BLAS stage. Otherwise
    torch's OpenMP pool and BLAS would be T-wide at the same time.
    """
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stages: dict[str, list[dict[str, Any]]] = {}
    # BLAS owns the only T-wide layer here; OpenMP and torch intra-op are one.
    with threadpool_limits(limits=compute_threads, user_api="blas"), \
            threadpool_limits(limits=1, user_api="openmp"):
        matrix = torch.arange(400, dtype=torch.float64).reshape(20, 20)
        (matrix @ matrix).sum()
        stages["blas"] = _pool_report(threadpool_info())
    # Torch intra-op owns the T-wide layer; BLAS and OpenMP are pinned to one.
    with threadpool_limits(limits=1, user_api="blas"), \
            threadpool_limits(limits=1, user_api="openmp"):
        torch.set_num_threads(compute_threads)
        torch.arange(64, dtype=torch.float32).sum()
        stages["torch"] = _pool_report(threadpool_info())
        # Read torch's own counters inside the stage: leaving the context
        # restores the OpenMP ceiling and would misreport the effective value.
        intra, interop = torch.get_num_threads(), torch.get_num_interop_threads()
    wide = {name: sum(1 for pool in pools if pool["num_threads"] > 1)
            for name, pools in stages.items()}
    return {
        "threadpools": stages, "wide_layers": wide,
        "torch_intra_threads": intra, "torch_interop_threads": interop,
    }


def child_main(
    job_id: str,
    attempt_id: str,
    limits: Mapping[str, Any],
    workload: str,
    options: Mapping[str, Any],
    queue,
    cancel_event=None,
    start_event=None,
    classical_input=None,
    fold_channel=None,
    finalization_input=None,
    evaluation_channel=None,
) -> None:
    """Run one workload and report a bounded result. Never returns data rows."""
    del job_id, attempt_id  # Identifiers are for supervisor logs only.
    if start_event is not None and not start_event.wait(10):
        return  # Parent died before durably registering our identity; never fit.
    compute_threads = int(limits["compute_threads"])
    checked = validate_options(workload, options)
    apply_thread_environment(compute_threads)
    try:
        if workload == "classical":
            # A spawned process has not performed any torch work yet. Inter-op
            # can only be configured once and before parallel work starts.
            import torch
            torch.set_num_interop_threads(1)
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=compute_threads, user_api="blas"), \
                threadpool_limits(limits=1, user_api="openmp"):
            if workload == "probe":
                result = _probe(checked, queue, compute_threads, cancel_event)
            elif workload == "classical":
                from dataclasses import asdict
                from ml.training.classical_suite import SuiteCancelled, evaluate_classical_suite
                from ml.training.cv import compare_candidates

                context, run = classical_input
                _emit(queue, PROGRESS, stage="fitting")
                mlp_interop_threads = None
                fold_proof = None

                def progress(family, fold, finished, total, fold_finished=False):
                    nonlocal mlp_interop_threads, fold_proof
                    if family == "mlp":
                        if fold_channel is None:
                            raise RuntimeError("MLP watchdog channel is required")
                        # The child cannot enter a fold before the supervisor
                        # arms the timer, or proceed to another without its ack.
                        fold_channel.send((fold, fold_finished, fold_proof if fold_finished else None))
                        ack_fold, ack_finished, proof = fold_channel.recv()
                        if (ack_fold, ack_finished) != (fold, fold_finished) or not isinstance(proof, str):
                            raise RuntimeError("Invalid watchdog acknowledgement")
                        fold_proof = None if fold_finished else proof
                        if not fold_finished:
                            mlp_interop_threads = torch.get_num_interop_threads()
                    _emit(queue, PROGRESS, stage="fitting", family=family, fold_index=fold,
                          finished_units=finished, total_units=total, fold_finished=fold_finished)
                try:
                    candidates = evaluate_classical_suite(
                        context, run, cancelled=cancel_event.is_set if cancel_event is not None else lambda: False,
                        include_mlp=True,
                        on_progress=progress,
                    )
                except SuiteCancelled:
                    return
                comparison = compare_candidates(context, candidates)
                from threadpoolctl import threadpool_info
                result = {"pid": os.getpid(), "workload": "classical", "candidates": [
                    {**asdict(candidate), "eligible": candidate.eligible} for candidate in candidates
                ], "comparison": {
                    "ranked": [{"family": entry.candidate.family, "mean": entry.mean, "std": entry.std,
                                "mean_fit_seconds": entry.mean_fit_seconds, "baseline_delta": entry.baseline_delta,
                                "tied": entry.tied} for entry in comparison.ranked],
                    "unranked": [candidate.family for candidate in comparison.unranked],
                }, "observed_threadpools": _pool_report(threadpool_info()),
                    "mlp_interop_threads": mlp_interop_threads}
            elif workload == "finalization":
                if evaluation_channel is None or finalization_input is None:
                    raise RuntimeError("Finalization channel and input are required")
                import torch
                torch.set_num_interop_threads(1)
                from dataclasses import asdict
                from time import perf_counter
                from apps.api.database.repository import JobRepository, _now
                from ml.evaluation.finalize import read_holdout, score_holdout, stage_final_bundle
                from ml.training.refit import refit_selected

                context, plan, candidate_result, fin, db_path = finalization_input
                cancel_check = cancel_event.is_set if cancel_event is not None else (lambda: False)
                _emit(queue, PROGRESS, stage="refitting")
                if checked.get("crash_before_evaluation"):
                    os._exit(int(checked["crash_before_evaluation"]))
                if checked.get("raise_before_evaluation"):
                    raise RuntimeError(str(checked["raise_before_evaluation"]))
                model = refit_selected(context, plan, candidate_result, control=cancel_check)
                if cancel_event is not None and cancel_event.is_set():
                    return
                # Refit complete; request HoldoutCapability from supervisor
                evaluation_channel.send(("ready_for_evaluation", plan.selection.candidate_digest))
                msg = evaluation_channel.recv()
                if not isinstance(msg, tuple) or msg[0] != "capability" or msg[1] is None:
                    raise RuntimeError("Failed to receive holdout capability")
                capability = msg[1]
                _emit(queue, PROGRESS, stage="evaluating")
                if checked.get("crash_after_evaluation_start"):
                    os._exit(int(checked["crash_after_evaluation_start"]))
                if checked.get("raise_after_evaluation_start"):
                    raise RuntimeError(str(checked["raise_after_evaluation_start"]))
                child_repo = JobRepository(db_path)
                holdout_view = read_holdout(child_repo, capability)
                if cancel_event is not None and cancel_event.is_set():
                    return
                eval_start = perf_counter()
                task = plan.selection.task
                bundle = score_holdout(model, holdout_view, task)
                eval_seconds = perf_counter() - eval_start
                now = _now()
                result_payload = {
                    "finalization_id": fin["finalization_id"],
                    "scope_id": fin["scope_id"],
                    "experiment_id": fin["experiment_id"],
                    "candidate_id": fin["candidate_id"],
                    "candidate_digest": fin["candidate_digest"],
                    "task": task.value,
                    "primary_metric": plan.selection.primary_metric,
                    "primary_score": plan.selection.primary_score,
                    "evaluation_seconds": eval_seconds,
                    "evaluated_at": now,
                    "metrics": asdict(bundle),
                    "epoch_budget": plan.epoch_budget,
                    "evidence_sha256": plan.evidence_sha256,
                }
                final_path, manifest_sha = stage_final_bundle(
                    child_repo.path.parent, fin["finalization_id"], fin["scope_id"],
                    capability.attempt_id, fin["candidate_id"], result_payload,
                )
                evaluation_channel.send(("staged", str(final_path), manifest_sha))
                from threadpoolctl import threadpool_info
                result = {
                    "pid": os.getpid(),
                    "workload": "finalization",
                    "finalization_id": fin["finalization_id"],
                    "result_path": str(final_path),
                    "result_checksum": manifest_sha,
                    "observed_threadpools": _pool_report(threadpool_info()),
                }
            else:  # pragma: no cover - validate_options already rejected this
                raise ValueError("Unknown workload")
    except Exception:
        # Only a closed code crosses the process boundary; no exception text.
        _emit(queue, FAILURE, code="workload_failed")
        raise SystemExit(1) from None

    if cancel_event is not None and cancel_event.is_set():
        return
    if workload == "probe" and checked["crash_exit_code"]:
        os._exit(checked["crash_exit_code"])
    _emit(queue, RESULT, payload=result)
