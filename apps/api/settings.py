"""Explicit environment loading without import-time I/O or directory creation."""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ml.contracts import ResourceLimits


@dataclass(frozen=True, slots=True)
class Settings:
    app_home: Path
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        home = self.app_home
        if not isinstance(home, Path) or not home.is_absolute():
            raise ValueError("Application home must be an absolute local path")
        if str(home).startswith(("\\\\", "//")):
            raise ValueError("Network shares cannot host active SQLite storage")
        resolved = home.resolve()
        if any(part.casefold().startswith("onedrive") or part.casefold() in
               {"dropbox", "google drive", "googledrive", "iclouddrive"}
               for part in resolved.parts):
            raise ValueError("Application home cannot be in a synchronized folder")
        if resolved.exists() and not resolved.is_dir():
            raise ValueError("Application home must be a directory")
        if sys.platform == "win32":
            import ctypes
            if ctypes.windll.kernel32.GetDriveTypeW(resolved.anchor) == 4:
                raise ValueError("Network drives cannot host active SQLite storage")
        if not isinstance(self.limits, ResourceLimits):
            raise ValueError("limits must be ResourceLimits")
        object.__setattr__(self, "app_home", resolved)

    @property
    def metadata_path(self) -> Path:
        return self.app_home / "metadata.sqlite3"

    @property
    def mlflow_database_path(self) -> Path:
        return self.app_home / "mlflow.sqlite3"

    @property
    def mlflow_artifacts_path(self) -> Path:
        return self.app_home / "mlflow-artifacts"

    @property
    def mlflow_tracking_uri(self) -> str:
        return "sqlite:///" + self.mlflow_database_path.as_posix()


def load_settings(
    environ: Mapping[str, str] | None = None,
    *,
    platform: str | None = None,
    user_home: Path | None = None,
) -> Settings:
    env = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if "MODELARENA_HOME" in env:
        raw_home = env["MODELARENA_HOME"]
    elif platform == "win32":
        if not env.get("LOCALAPPDATA"):
            raise ValueError("LOCALAPPDATA or absolute MODELARENA_HOME is required")
        raw_home = str(Path(env["LOCALAPPDATA"]) / "ModelArena")
    else:
        base = Path(env.get("XDG_DATA_HOME", str((user_home or Path.home()) / ".local/share")))
        raw_home = str(base / "modelarena")
    if not raw_home or raw_home.startswith(("\\\\", "//")):
        raise ValueError("Application home must be an absolute local path")
    try:
        threads = int(env.get("MODELARENA_COMPUTE_THREADS", "4"))
    except ValueError as error:
        raise ValueError("MODELARENA_COMPUTE_THREADS must be 2, 4 or 6") from error
    return Settings(app_home=Path(raw_home), limits=ResourceLimits(compute_threads=threads))
