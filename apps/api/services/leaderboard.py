"""Read the most recent published CV comparison, never final evaluation data."""

from dataclasses import dataclass
import math

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError


def _ensure_available(repository: JobRepository, experiment_id: str) -> None:
    with repository._connection() as db:
        pending = db.execute("SELECT 1 FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
                             "JOIN datasets source ON source.dataset_id=c.dataset_id "
                             "JOIN datasets d ON d.canonical_sha256=source.canonical_sha256 "
                             "JOIN deletion_intents i ON i.dataset_id=d.dataset_id WHERE e.experiment_id=?",
                             (experiment_id,)).fetchone()
    if pending:
        raise ConflictError("Dataset deletion is pending")


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    candidate_id: str
    family: str
    mean: float | None = None
    std: float | None = None
    mean_fit_seconds: float | None = None
    baseline_delta: float | None = None
    tied: bool = False


@dataclass(frozen=True, slots=True)
class Leaderboard:
    experiment_id: str
    job_id: str | None
    primary_metric: str
    direction: str
    ranked: tuple[LeaderboardEntry, ...]
    unranked: tuple[LeaderboardEntry, ...]
    partition: str = "cross_validation_training"


def get_leaderboard(repository: JobRepository, experiment_id: str) -> Leaderboard:
    """Use one completed CV attempt, preserving the worker's verified comparison.

    Do not combine attempts: their training diagnostics and timings belong to
    separate runs. An unfinished retry does not replace a published result.
    """
    _ensure_available(repository, experiment_id)
    with repository._connection() as db:
        experiment = db.execute("SELECT primary_metric,direction FROM experiments WHERE experiment_id=?",
                                (experiment_id,)).fetchone()
        if experiment is None:
            raise NotFoundError("Experiment not found")
        job = db.execute("SELECT job_id FROM jobs WHERE experiment_id=? AND kind='cv' AND state='completed' "
                         "AND EXISTS (SELECT 1 FROM candidates WHERE candidates.job_id=jobs.job_id) "
                         "ORDER BY sequence DESC LIMIT 1", (experiment_id,)).fetchone()
        rows = [] if job is None else db.execute(
            "SELECT candidate_id,family,status FROM candidates WHERE job_id=? AND experiment_id=?",
            (job["job_id"], experiment_id)).fetchall()
    if job is None:
        return Leaderboard(experiment_id, None, experiment["primary_metric"], experiment["direction"], (), ())

    try:
        payload = repository.read_published_results(job["job_id"])
    except (OSError, NotFoundError):
        _ensure_available(repository, experiment_id)
        raise
    identities = {row["family"]: row for row in rows}
    comparison = payload["comparison"]
    ranked = []
    seen = set()
    for entry in comparison["ranked"]:
        family = entry["family"]
        if family in seen or family not in identities or identities[family]["status"] != "completed":
            raise ValueError("Published comparison does not match candidate metadata")
        numbers = (entry["mean"], entry["std"], entry["mean_fit_seconds"])
        delta = entry["baseline_delta"]
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers)
                or entry["std"] < 0 or entry["mean_fit_seconds"] < 0
                or (delta is not None and (type(delta) not in (int, float) or not math.isfinite(delta)))
                or type(entry["tied"]) is not bool):
            raise ValueError("Invalid published comparison")
        ranked.append(LeaderboardEntry(identities[family]["candidate_id"], family, *numbers, delta, entry["tied"]))
        seen.add(family)
    unranked = []
    for family in comparison["unranked"]:
        if family in seen or family not in identities or identities[family]["status"] != "failed":
            raise ValueError("Published comparison does not match candidate metadata")
        unranked.append(LeaderboardEntry(identities[family]["candidate_id"], family))
        seen.add(family)
    if len(identities) != len(rows) or seen != identities.keys():
        raise ValueError("Published comparison is incomplete")
    _ensure_available(repository, experiment_id)
    return Leaderboard(experiment_id, job["job_id"], experiment["primary_metric"], experiment["direction"],
                       tuple(ranked), tuple(unranked))


def get_candidate(repository: JobRepository, candidate_id: str) -> dict:
    """Return only the named candidate's CV data from a verified bundle."""
    with repository._connection() as db:
        row = db.execute("SELECT c.job_id,c.family,c.experiment_id FROM candidates c JOIN jobs j ON j.job_id=c.job_id "
                         "WHERE c.candidate_id=? AND j.state='completed'", (candidate_id,)).fetchone()
    if row is None:
        raise NotFoundError("Candidate not found")
    _ensure_available(repository, row["experiment_id"])
    try:
        payload = repository.read_published_results(row["job_id"])
    except (OSError, NotFoundError):
        _ensure_available(repository, row["experiment_id"])
        raise
    matches = [candidate for candidate in payload["candidates"] if candidate["family"] == row["family"]]
    if len(matches) != 1:
        raise ValueError("Published candidate does not match metadata")
    _ensure_available(repository, row["experiment_id"])
    return {"candidate_id": candidate_id, **matches[0]}
