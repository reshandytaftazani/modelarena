"""Closed, reproducible classical model registry; no user-supplied estimators."""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge

from ml.contracts import TaskType


def classical_families(task: TaskType) -> tuple[str, ...]:
    if task == TaskType.BINARY_CLASSIFICATION and isinstance(task, TaskType):
        return ("dummy", "logistic", "random_forest", "hist_gradient_boosting")
    if task == TaskType.REGRESSION and isinstance(task, TaskType):
        return ("dummy", "ridge", "random_forest", "hist_gradient_boosting")
    raise ValueError("Unsupported task")


def model_families(task: TaskType) -> tuple[str, ...]:
    """Unified registry of all supported families per task (classical and MLP)."""
    if task == TaskType.BINARY_CLASSIFICATION and isinstance(task, TaskType):
        return ("dummy", "logistic", "random_forest", "hist_gradient_boosting", "mlp")
    if task == TaskType.REGRESSION and isinstance(task, TaskType):
        return ("dummy", "ridge", "random_forest", "hist_gradient_boosting", "mlp")
    raise ValueError("Unsupported task")


def make_classical(task: TaskType, family: str, seed: int) -> BaseEstimator:
    if not isinstance(family, str) or family not in classical_families(task):
        raise ValueError("Unsupported task or classical family")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Model seed must be an unsigned 32-bit integer")
    classification = task == TaskType.BINARY_CLASSIFICATION
    if family == "dummy":
        return DummyClassifier(strategy="prior", random_state=seed) if classification else DummyRegressor(strategy="mean")
    if family == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if family == "ridge":
        return Ridge(alpha=1.0, solver="svd")
    if family == "random_forest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(n_estimators=100, max_depth=16, min_samples_leaf=2, n_jobs=1, random_state=seed)
    cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
    return cls(max_iter=100, early_stopping=False, random_state=seed)
