"""Read the most recent published CV comparison, never final evaluation data."""

from dataclasses import dataclass
import math

from apps.api.database.repository import JobRepository, NotFoundError


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

    payload = repository.read_published_results(job["job_id"])
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
    return Leaderboard(experiment_id, job["job_id"], experiment["primary_metric"], experiment["direction"],
                       tuple(ranked), tuple(unranked))
