"""The closed, task-specific classical estimator registry."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge

from ml.contracts import TaskType
from ml.models.classical import classical_families, make_classical


@pytest.mark.parametrize("task,expected", [
    (TaskType.BINARY_CLASSIFICATION, (DummyClassifier, LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier)),
    (TaskType.REGRESSION, (DummyRegressor, Ridge, RandomForestRegressor, HistGradientBoostingRegressor)),
])
def test_task_registry_and_architecture_defaults(task, expected):
    families = classical_families(task)
    assert len(families) == len(set(families)) == 4
    models = [make_classical(task, family, 42) for family in families]
    assert tuple(type(model) for model in models) == expected
    linear, forest, hgb = models[1:]
    if task == TaskType.BINARY_CLASSIFICATION:
        assert linear.max_iter == 1000 and linear.random_state == 42
    else:
        assert linear.alpha == 1 and linear.solver == "svd"
    assert (forest.n_estimators, forest.max_depth, forest.min_samples_leaf, forest.n_jobs, forest.random_state) == (100, 16, 2, 1, 42)
    assert (hgb.max_iter, hgb.early_stopping, hgb.random_state) == (100, False, 42)


@pytest.mark.parametrize("task", list(TaskType))
def test_models_produce_finite_predictions_and_binary_positive_scores(task):
    x = np.arange(80, dtype=float).reshape(40, 2)
    y = np.arange(40) % 2 if task == TaskType.BINARY_CLASSIFICATION else np.arange(40, dtype=float)
    for family in classical_families(task):
        model = make_classical(task, family, 77).fit(x, y)
        assert np.isfinite(model.predict(x)).all()
        if task == TaskType.BINARY_CLASSIFICATION:
            scores = model.predict_proba(x)[:, list(model.classes_).index(1)]
            assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))


def test_registry_rejects_untrusted_inputs():
    for task, family, seed in [("regression", "dummy", 42), (TaskType.REGRESSION, "logistic", 42),
                               (TaskType.BINARY_CLASSIFICATION, "ridge", 42),
                               (TaskType.REGRESSION, "__import__", 42), (TaskType.REGRESSION, "dummy", True)]:
        with pytest.raises(ValueError):
            make_classical(task, family, seed)
