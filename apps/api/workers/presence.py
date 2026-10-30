"""Best-effort, observable worker presence; never used to steal a lease."""

from datetime import datetime, timezone
import logging
import os
from threading import Event, Thread
from uuid import uuid4

from apps.api.database.repository import JobRepository, _now
from apps.api.workers.monitor import process_creation_time, process_liveness

logger = logging.getLogger(__name__)
PRESENCE_INTERVAL = 2.0
PRESENCE_TIMEOUT = 10.0


class WorkerPresence:
    def __init__(self, repository: JobRepository):
        self.repository = repository
        self.owner = str(uuid4())
        self.pid = os.getpid()
        self.created = process_creation_time(self.pid)
        self.stop = Event()
        self.thread = Thread(target=self._heartbeat, name="worker-presence", daemon=True)

    def __enter__(self):
        with self.repository._transaction() as db:
            db.execute("INSERT INTO worker_presence VALUES(?,?,?,?)", (self.owner, self.pid, self.created, _now()))
        self.thread.start()
        return self

    def _heartbeat(self):
        while not self.stop.wait(PRESENCE_INTERVAL):
            try:
                with self.repository._transaction() as db:
                    db.execute("UPDATE worker_presence SET heartbeat_at=? WHERE owner_token=? AND pid=? AND process_created=?",
                               (_now(), self.owner, self.pid, self.created))
            except Exception:
                logger.warning("Worker presence heartbeat unavailable")

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            logger.warning("Worker presence shutdown is incomplete")
            return
        try:
            with self.repository._transaction() as db:
                db.execute("DELETE FROM worker_presence WHERE owner_token=?", (self.owner,))
        except Exception:
            logger.warning("Worker presence cleanup unavailable")


def worker_status(repository: JobRepository) -> str:
    with repository._connection() as db:
        records = list(db.execute("SELECT pid,process_created,heartbeat_at FROM worker_presence"))
    now = datetime.now(timezone.utc)
    uncertain = False
    for row in records:
        try:
            heartbeat = datetime.fromisoformat(row["heartbeat_at"])
            age = (now - heartbeat).total_seconds()
            liveness = process_liveness(row["pid"], row["process_created"])
        except (ValueError, TypeError, OverflowError):
            uncertain = True
            continue
        if liveness == "alive" and 0 <= age <= PRESENCE_TIMEOUT:
            return "available"
        if liveness != "dead":
            uncertain = True
    return "unknown" if uncertain else "not_detected"
