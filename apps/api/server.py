"""Loopback-only API server with a fresh, terminal-displayed session token."""

import argparse
import secrets

import uvicorn

from apps.api.database.repository import JobRepository
from apps.api.main import create_app
from apps.api.settings import load_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start the loopback-only ModelArena API")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ui-origin", action="append", default=[])
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    settings = load_settings()
    token = secrets.token_urlsafe(32)
    app = create_app(JobRepository(settings.metadata_path), session_token=token,
                     allowed_origins=tuple(args.ui_origin), limits=settings.limits)
    print("Session token: " + token, flush=True)
    print("Keep this token private. It changes on API restart.", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
