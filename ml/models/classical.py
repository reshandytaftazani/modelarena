"""The Day 6 Dummy reference registry; other classical families arrive Day 10."""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor

from ml.contracts import TaskType


def make_classical(task: TaskType, family: str, seed: int) -> BaseEstimator:
    if not isinstance(task, TaskType) or family != "dummy":
        raise ValueError("Unsupported task or classical family")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Model seed must be an unsigned 32-bit integer")
    if task == TaskType.BINARY_CLASSIFICATION:
        return DummyClassifier(strategy="prior", random_state=seed)
    return DummyRegressor(strategy="mean")
