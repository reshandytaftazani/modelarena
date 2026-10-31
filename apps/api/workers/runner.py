"""Explicit local worker process; training always runs in a spawned child."""

import argparse
import logging
from time import sleep
from uuid import uuid4

from apps.api.database.repository import JobRepository
from apps.api.console import enable_console_interrupt
from apps.api.settings import load_settings
from apps.api.workers.presence import WorkerPresence
from apps.api.workers.supervisor import WorkerSupervisor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local ModelArena worker")
    parser.add_argument("--once", action="store_true", help="claim at most one job and exit")
    args = parser.parse_args(argv)
    settings = load_settings()
    repository = JobRepository(settings.metadata_path)
    worker = WorkerSupervisor(repository, limits=settings.limits, workload="classical",
                              load_from_storage=True, claim_all=True, owner_token=str(uuid4()))
    try:
        with WorkerPresence(repository):
            while True:
                result = worker.run_once()
                if result is not None and result.safe_error == "supervisor_stopped":
                    return 130
                if args.once:
                    return 0 if result is None or result.state == "completed" else 1
                if result is None:
                    sleep(2)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Worker stopped")
        return 130


if __name__ == "__main__":
    enable_console_interrupt()
    raise SystemExit(main())
