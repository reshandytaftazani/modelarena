"""Bounded vocabulary shared by the supervisor, the child and the repository.

Nothing here imports a numeric library, psutil or the database layer, so this
module is safe to import inside a freshly spawned Windows child. Stages and
error codes are closed allowlists: they are persisted and rendered, so an
unvalidated string could leak a path, a row value or an exception message.
"""

from dataclasses import dataclass
from typing import Any, Mapping

#: Every stage the supervisor may persist. Ordered, lowercase, no free text.
WORKER_STAGES = frozenset({
    "starting", "preparing", "fitting", "scoring", "finalizing",
    "refitting", "evaluating",
    "completed", "failed", "cancelled", "interrupted",
})

#: Terminal job states the worker may publish. ``running``/``queued`` are not
#: terminal and are owned by the claim path.
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "interrupted"})

#: Fixed failure vocabulary. No exception text, path or row value may enter it.
SAFE_ERROR_CODES = frozenset({
    "workload_failed", "child_exited", "timeout", "memory_limit_exceeded",
    "resource_pressure", "supervisor_error", "cancelled", "supervisor_stopped", "owner_lost",
})


def check_stage(stage: object) -> str:
    """Return a persisted stage name, rejecting anything outside the allowlist."""
    if not isinstance(stage, str) or stage not in WORKER_STAGES:
        raise ValueError("Unsupported worker stage")
    return stage


def check_state(state: object) -> str:
    if not isinstance(state, str) or state not in TERMINAL_STATES:
        raise ValueError("Unsupported terminal job state")
    return state


def check_safe_error(code: object) -> str | None:
    if code is None:
        return None
    if not isinstance(code, str) or code not in SAFE_ERROR_CODES:
        raise ValueError("Unsupported safe error code")
    return code


@dataclass(frozen=True, slots=True)
class WorkloadOption:
    """One typed, bounded probe option; used to reject free-form client input."""

    name: str
    kind: type
    minimum: float | int | None = None
    maximum: float | int | None = None


_PROBE_OPTIONS = (
    WorkloadOption("work_seconds", (int, float), 0, 600),
    WorkloadOption("steps", int, 1, 1000),
    WorkloadOption("silent_seconds", (int, float), 0, 600),
    WorkloadOption("allocate_mb", int, 0, 2048),
    WorkloadOption("crash_exit_code", int, 0, 127),
    WorkloadOption("raise_message", str, None, None),
    WorkloadOption("report_pools", bool, None, None),
)

_FINALIZATION_OPTIONS = (
    WorkloadOption("crash_before_evaluation", int, 0, 127),
    WorkloadOption("raise_before_evaluation", str, None, None),
    WorkloadOption("crash_after_evaluation_start", int, 0, 127),
    WorkloadOption("raise_after_evaluation_start", str, None, None),
)

#: Registry of executable workloads. A new entry needs its own validation and
#: tests; nothing is resolved from a client-supplied path or import string.
WORKLOADS: Mapping[str, tuple[WorkloadOption, ...]] = {
    "probe": _PROBE_OPTIONS,
    "classical": _FINALIZATION_OPTIONS,
    "finalization": _FINALIZATION_OPTIONS,
}

PROBE_DEFAULTS: Mapping[str, Any] = {
    "work_seconds": 0.2, "steps": 2, "silent_seconds": 0.0, "allocate_mb": 0,
    "crash_exit_code": 0, "raise_message": "", "report_pools": False,
}

FINALIZATION_DEFAULTS: Mapping[str, Any] = {
    "crash_before_evaluation": 0,
    "raise_before_evaluation": "",
    "crash_after_evaluation_start": 0,
    "raise_after_evaluation_start": "",
}


def check_workload(name: object) -> str:
    if not isinstance(name, str) or name not in WORKLOADS:
        raise ValueError("Unknown worker workload")
    return name


def validate_options(name: str, options: Mapping[str, Any]) -> dict[str, Any]:
    """Merge defaults and validate types/ranges; never accept extra keys."""
    check_workload(name)
    if not isinstance(options, Mapping):
        raise ValueError("Workload options must be a mapping")
    declared = {option.name: option for option in WORKLOADS[name]}
    unknown = set(options) - set(declared)
    if unknown:
        raise ValueError("Unknown workload option")
    merged = dict(PROBE_DEFAULTS) if name == "probe" else dict(FINALIZATION_DEFAULTS) if name in ("classical", "finalization") else {}
    merged.update(options)
    for key, option in declared.items():
        value = merged[key]
        if option.kind is bool:
            if type(value) is not bool:
                raise ValueError(f"Option {key} must be boolean")
            continue
        if option.kind is str:
            if not isinstance(value, str):
                raise ValueError(f"Option {key} must be a string")
            continue
        if option.kind is int:
            if type(value) is not int:
                raise ValueError(f"Option {key} must be an integer")
        elif not isinstance(value, option.kind) or isinstance(value, bool):
            raise ValueError(f"Option {key} must be numeric")
        if option.minimum is not None and value < option.minimum:
            raise ValueError(f"Option {key} is below its bound")
        if option.maximum is not None and value > option.maximum:
            raise ValueError(f"Option {key} is above its bound")
        if isinstance(value, float) and not value == value:  # NaN
            raise ValueError(f"Option {key} must be finite")
    return merged
