"""Record inventory, free memory and lock identity; never train or benchmark.

Run from the checkout with ``uv run python -m scripts.profile_hardware``.
The report is stored privately under application home and also printed as JSON.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil

from apps.api.settings import load_settings


def _probe(command: list[str], cwd: Path) -> dict[str, str]:
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable"}
    if result.returncode:
        return {"status": "unavailable"}
    return {"status": "observed", "value": result.stdout.strip()}


def build_inventory(repository: Path) -> dict[str, object]:
    """Observe this machine without creating files or importing ML frameworks."""
    memory = psutil.virtual_memory()
    cpu_name = platform.processor()
    os_caption = platform.platform()
    build = platform.version().split(".")[-1]
    if sys.platform == "win32":
        result = _probe([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
            "$cpu=Get-CimInstance Win32_Processor; $os=Get-CimInstance Win32_OperatingSystem; "
            "@{cpu=($cpu | Select-Object -First 1).Name; "
            "caption=$os.Caption; build=$os.BuildNumber} | ConvertTo-Json -Compress",
        ], repository)
        if result["status"] == "observed":
            try:
                observed = json.loads(result["value"])
                cpu_name, os_caption, build = observed["cpu"], observed["caption"], observed["build"]
            except (ValueError, KeyError, TypeError):
                pass
    reference = {
        "cpu": "AMD Ryzen 5 5500U", "physical_cores": 6, "logical_cores": 12,
        "installed_gib": 15.33, "os": "Windows 11 Home Single Language", "build": "22621",
    }
    comparisons = {
        "cpu": "Ryzen 5 5500U" in cpu_name,
        "physical_cores": psutil.cpu_count(logical=False) == 6,
        "logical_cores": psutil.cpu_count() == 12,
        "ram": abs(memory.total / 1024**3 - 15.33) <= 0.1,
        "os": "Windows 11 Home Single Language" in os_caption,
        "build": build == "22621",
    }
    locks = {}
    for filename in ("uv.lock", "pyproject.toml", "package-lock.json", "package.json",
                     ".python-version", ".node-version"):
        path = repository / filename
        locks[filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else None
    packages = {}
    for name in ("fastapi", "pydantic", "numpy", "pandas", "scikit-learn",
                 "torch", "mlflow", "psutil", "pytest", "httpx"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv.exe")
    battery = psutil.sensors_battery()
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": {"name": cpu_name, "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count()},
        "os": {"caption": os_caption, "build": build, "machine": platform.machine()},
        "memory": {"total_bytes": memory.total, "available_bytes": memory.available,
                   "total_gib": round(memory.total / 1024**3, 3),
                   "available_gib": round(memory.available / 1024**3, 3)},
        "reference": reference,
        "reference_comparison": {
            "status": "match" if all(comparisons.values()) else "mismatch",
            "checks": comparisons,
            "note": "OS updates may explain differences; this is inventory, not performance evidence.",
        },
        "power": {
            "active_scheme": _probe(["powercfg.exe", "/getactivescheme"], repository)
            if sys.platform == "win32" else {"status": "unavailable"},
            "plugged_in": battery.power_plugged if battery else None,
            "battery_percent": battery.percent if battery else None,
            "windows_power_mode": "not observed; active scheme may differ from Settings power mode",
        },
        "tools": {"python": platform.python_version(), "uv": _probe([uv, "--version"], repository),
                  "node": _probe(["node", "--version"], repository),
                  "npm": _probe(["npm.cmd" if os.name == "nt" else "npm", "--version"], repository)},
        "packages": packages,
        "locks": locks,
        "code_revision": _probe(["git", "rev-parse", "HEAD"], repository),
        "resource_settings_status": "provisional; not benchmarked or enforced",
        "focused_hours_per_day": "2–3 (user confirmed 2026-10-22)",
    }


def main() -> None:
    settings = load_settings()
    inventory = build_inventory(Path(__file__).resolve().parents[1])
    inventory["resource_limits"] = asdict(settings.limits)
    output = settings.app_home / "inventory"
    output.mkdir(parents=True, exist_ok=True)
    # A pre-existing junction must not redirect generated evidence elsewhere.
    if not output.resolve().is_relative_to(settings.app_home):
        raise ValueError("Inventory directory escapes application home")
    report = output / "hardware.json"
    if report.is_symlink():
        raise ValueError("Inventory report cannot be a symlink")
    payload = json.dumps(inventory, indent=2, ensure_ascii=False)
    report.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
