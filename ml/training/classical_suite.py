"""Serial classical CV on one frozen training-only context."""

from dataclasses import replace
from typing import Callable

from ml.contracts import CandidateResult, RunConfig, TrainingContext
from ml.models.classical import classical_families, model_families
from ml.training.cv import evaluate_candidate


class SuiteCancelled(Exception):
    """No partial suite result may be published after cancellation."""


def evaluate_classical_suite(
    context: TrainingContext, run: RunConfig, *,
    cancelled: Callable[[], bool] = lambda: False,
    on_candidate: Callable[[CandidateResult], None] = lambda result: None,
    on_progress: Callable[[str, int, int, int], None] = lambda _family, _fold, _finished, _total: None,
    include_mlp: bool = False,
) -> tuple[CandidateResult, ...]:
    results = []
    families = (model_families if include_mlp else classical_families)(context.training.config.task)
    folds = context.split.fold_count
    for family_index, family in enumerate(families):
        if cancelled():
            raise SuiteCancelled()
        def on_fold(fold_index: int, finished: bool) -> None:
            try:
                on_progress(family, fold_index, family_index * folds + fold_index + int(finished), len(families) * folds, fold_finished=finished)
            except TypeError:
                on_progress(family, fold_index, family_index * folds + fold_index + int(finished), len(families) * folds)
        result = evaluate_candidate(context, replace(run, family=family), cancelled=cancelled, on_fold=on_fold)
        results.append(result)
        on_candidate(result)
    if cancelled():
        raise SuiteCancelled()
    return tuple(results)
