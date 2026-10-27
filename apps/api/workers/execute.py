"""Child entrypoint for one execution. Import-safe: no numeric library at scope.

A freshly spawned Windows child imports this module before any work starts, so
module scope stays cheap and side-effect free. Thread environment variables are
applied inside :func:`child_main` before the first numerical import, and each
stage enables exactly one T-wide compute layer:

* BLAS stage: BLAS up to T, OpenMP pinned to one.
* Torch stage: torch intra-op T, inter-op one, BLAS and OpenMP pinned to one.

Reporting the pools per stage is how the tests prove there is no nested T-wide
pair. The ``probe`` workload exists so process identity, heartbeat, progress and
resource limits can be exercised for real without training on user data. Dataset
CV execution is wired in with the classical registry (Days 10-12); nothing here
reads a dataset, a holdout partition or a user path.
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


def _probe(options: Mapping[str, Any], queue, compute_threads: int) -> dict[str, Any]:
    """Bounded synthetic work; never touches user data."""
    _emit(queue, PROGRESS, stage="preparing")
    allocation = None
    if options["allocate_mb"]:
        allocation = bytearray(options["allocate_mb"] * 1024 * 1024)
    _emit(queue, PROGRESS, stage="fitting")

    steps = options["steps"]
    per_step = float(options["work_seconds"]) / steps
    for _ in range(steps):
        if per_step > 0:
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
) -> None:
    """Run one workload and report a bounded result. Never returns data rows."""
    del job_id, attempt_id  # Identifiers are for supervisor logs only.
    compute_threads = int(limits["compute_threads"])
    checked = validate_options(workload, options)
    apply_thread_environment(compute_threads)
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=compute_threads, user_api="blas"), \
                threadpool_limits(limits=1, user_api="openmp"):
            if workload == "probe":
                result = _probe(checked, queue, compute_threads)
            else:  # pragma: no cover - validate_options already rejected this
                raise ValueError("Unknown workload")
    except Exception:
        # Only a closed code crosses the process boundary; no exception text.
        _emit(queue, FAILURE, code="workload_failed")
        raise SystemExit(1) from None

    if checked["crash_exit_code"]:
        os._exit(checked["crash_exit_code"])
    _emit(queue, RESULT, payload=result)
