"""Child resource observation with an explicit, verifiable lifecycle.

The supervisor samples the *owned child* process, never the API process, and
never terminates a process by a stored PID alone. ``psutil`` creation time is
compared before any terminate call so a recycled PID cannot be mistaken for our
child. Sampled RSS can understate instantaneous peaks; no OS quota is claimed.

Only numbers are retained. No path, exception text, row value or environment
variable enters a summary, so summaries are safe to persist and log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import psutil

MIN_INTERVAL_SECONDS = 0.05
MAX_INTERVAL_SECONDS = 5.0
#: Windows reuses PIDs; a generous but finite identity tolerance.
CREATION_TIME_TOLERANCE_SECONDS = 2.0


def normalized_cpu_percent(cpu_seconds: float, wall_seconds: float, logical_cores: int) -> float:
    """100% means every logical processor is occupied, not one of them."""
    if (not math.isfinite(cpu_seconds) or not math.isfinite(wall_seconds)
            or cpu_seconds < 0 or wall_seconds <= 0 or logical_cores < 1):
        raise ValueError("Invalid CPU observation")
    return 100 * cpu_seconds / wall_seconds / logical_cores


def logical_cores() -> int:
    return psutil.cpu_count() or 1


def process_creation_time(pid: int) -> float:
    """Read the creation time used to prove a PID still identifies our child."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("A positive process id is required")
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error as error:
        raise RuntimeError("Owned child process is not observable") from error


def check_interval(seconds: object) -> float:
    if (type(seconds) not in (int, float) or isinstance(seconds, bool)
            or not MIN_INTERVAL_SECONDS <= seconds <= MAX_INTERVAL_SECONDS):
        raise ValueError("Sample interval must be between 0.05 and 5 seconds")
    return float(seconds)


@dataclass(slots=True)
class MemoryPressureTracker:
    """True once available memory stays below the reserve for a full window.

    A single low sample is not an emergency, and recovery resets the window, so
    a transient dip cannot interrupt a job.
    """

    reserve_bytes: int
    seconds: float
    _since: float | None = field(default=None, repr=False)
    _breaching: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.reserve_bytes) is not int or self.reserve_bytes < 0:
            raise ValueError("Reserve must be a nonnegative integer")
        if type(self.seconds) not in (int, float) or isinstance(self.seconds, bool) or self.seconds <= 0:
            raise ValueError("Pressure window must be positive")

    def update(self, available_bytes: int, *, now: float) -> bool:
        if type(available_bytes) is not int or available_bytes < 0:
            raise ValueError("Available memory must be a nonnegative integer")
        if available_bytes >= self.reserve_bytes:
            self._since, self._breaching = None, False
            return False
        if self._since is None:
            self._since = now
        self._breaching = (now - self._since) >= self.seconds
        return self._breaching


class ChildSampler:
    """Explicit start/sample/stop observation of one owned child process."""

    def __init__(self, pid: int, *, creation_time: float, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = check_interval(interval_seconds)
        if type(creation_time) not in (int, float) or isinstance(creation_time, bool):
            raise ValueError("A numeric creation time is required")
        try:
            self._process = psutil.Process(pid)
            actual = self._process.create_time()
        except psutil.Error as error:
            raise RuntimeError("Owned child process is not observable") from error
        if abs(actual - float(creation_time)) > CREATION_TIME_TOLERANCE_SECONDS:
            raise RuntimeError("Process identity does not match the owned child")
        self._pid = pid
        self._samples: list[dict[str, float]] = []
        self._started_at: float | None = None
        self._stopped = False
        self._ended = False
        self._cpu_baseline: float | None = None

    @property
    def latest(self) -> dict[str, float] | None:
        """Most recent observation, or None before the first successful sample."""
        return self._samples[-1] if self._samples else None

    def start(self) -> "ChildSampler":
        if self._started_at is not None:
            raise RuntimeError("A sampler cannot be restarted")
        self._started_at = time.perf_counter()
        self.sample()
        return self

    def sample(self) -> None:
        """Take one observation; a dead child ends sampling but keeps history."""
        if self._started_at is None:
            raise RuntimeError("Sampler has not started")
        if self._stopped:
            raise RuntimeError("Sampler has stopped")
        try:
            memory = self._process.memory_info().rss
            cpu = self._process.cpu_times()
            available = psutil.virtual_memory().available
        except psutil.NoSuchProcess:
            self._ended = True
            return
        except psutil.Error as error:
            raise RuntimeError("Child observation failed") from error
        total_cpu = cpu.user + cpu.system
        if self._cpu_baseline is None:
            self._cpu_baseline = total_cpu
        self._samples.append({
            "elapsed_seconds": time.perf_counter() - self._started_at,
            "rss_bytes": float(memory),
            "process_cpu_seconds": total_cpu,
            "available_bytes": float(available),
        })

    def stop(self) -> None:
        self._stopped = True
        self._ended_before_stop = self._ended

    def summary(self) -> dict[str, Any]:
        if self._started_at is None or not self._stopped or not self._samples:
            raise RuntimeError("Sampler has not finished")
        first, last = self._samples[0], self._samples[-1]
        wall = last["elapsed_seconds"] - first["elapsed_seconds"]
        cpu = max(0.0, last["process_cpu_seconds"] - (self._cpu_baseline or 0.0))
        cores = logical_cores()
        return {
            "pid": self._pid,
            "sample_count": len(self._samples),
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "normalized_cpu_percent": (normalized_cpu_percent(cpu, wall, cores)
                                       if wall > 0 else 0.0),
            "logical_cores": cores,
            "peak_rss_bytes": int(max(item["rss_bytes"] for item in self._samples)),
            "minimum_available_bytes": int(min(item["available_bytes"] for item in self._samples)),
            "sample_interval_seconds": self.interval_seconds,
            "ended_before_stop": bool(getattr(self, "_ended_before_stop", False)),
            "peak_note": "Sampled owned-child RSS; transient peaks may be missed.",
        }


def terminate_owned_child(pid: int, *, creation_time: float, timeout: float = 5.0) -> bool:
    """Terminate only a verified owned child; an unverified PID is never killed."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("A positive process id is required")
    if type(timeout) not in (int, float) or isinstance(timeout, bool) or not 0 < timeout <= 30:
        raise ValueError("Termination timeout must be between 0 and 30 seconds")
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - float(creation_time)) > CREATION_TIME_TOLERANCE_SECONDS:
            return False
        process.terminate()
        process.wait(timeout=timeout)
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False
    return True
