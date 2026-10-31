"""Training-only full refit and write-only, private local artifacts.

No holdout reader, artifact deserializer, or client-facing path is provided here.
The refit plan is an in-process contract; durable reservation integration belongs
to the finalization worker, not this module.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import pickle
import secrets
from statistics import median
from time import perf_counter
from typing import Callable
from uuid import uuid4

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ml.contracts import CandidateResult, EpochLoss, FrozenSelection, TrainingContext
from ml.models.classical import make_classical, model_families
from ml.models.mlp import create_dataloader, make_mlp
from ml.preprocessing.factory import FoldPreprocessor
from ml.tracking.manifest import canonical_bytes, digest_bytes, execution_provenance, file_digest
from ml.training.cv import compare_candidates
from ml.training.mlp_loop import RunControl, train_mlp

_HANDLE_KEY = secrets.token_bytes(32)


@dataclass(frozen=True, slots=True)
class RefitPlan:
    selection: FrozenSelection
    evidence_sha256: str
    dataset_sha256: str
    training_view_sha256: str
    epoch_budget: int | None
    best_epochs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrustedModel:
    """Live objects produced by this process, not a deserialized model."""
    model: object = field(repr=False)
    preprocessor: FoldPreprocessor = field(repr=False)
    plan: RefitPlan
    train_row_ids_sha256: str
    training_loss: tuple[EpochLoss, ...]
    fit_seconds: float
    _tag: bytes = field(repr=False)


def _model_tag(model: object, preprocessor: FoldPreprocessor, plan: RefitPlan, rows: str,
               history: tuple[EpochLoss, ...], fit_seconds: float) -> bytes:
    payload = [id(model), id(preprocessor), asdict(plan), rows, [asdict(p) for p in history], fit_seconds]
    return hmac.new(_HANDLE_KEY, canonical_bytes(payload), hashlib.sha256).digest()


def freeze_refit(context: TrainingContext, selection: FrozenSelection, candidate: CandidateResult) -> RefitPlan:
    """Validate complete CV evidence and freeze its digest and epoch policy."""
    if not isinstance(selection, FrozenSelection) or not isinstance(candidate, CandidateResult):
        raise TypeError("Frozen selection and candidate evidence required")
    ranked = compare_candidates(context, (candidate,)).ranked
    run = candidate.run_config
    if (not ranked or run is None or candidate.family not in model_families(candidate.task)
            or candidate.family != selection.family or candidate.task != selection.task
            or candidate.model_seed != selection.model_seed or run.limits != selection.resource_limits
            or candidate.split_sha256 != selection.split_sha256
            or candidate.primary_metric != selection.primary_metric or candidate.direction != selection.direction
            or digest_bytes(canonical_bytes(asdict(run))) != selection.config_sha256
            or not math.isclose(ranked[0].mean, selection.primary_score, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Selection does not match complete frozen CV evidence")
    # Same v1 selection digest as the reservation service, without importing API code.
    digest = digest_bytes(canonical_bytes([
        "modelarena-candidate-digest-v1", selection.candidate_id, selection.family,
        selection.split_sha256, selection.config_sha256, selection.primary_metric,
        round(float(selection.primary_score), 6), selection.model_seed,
    ]))
    if not hmac.compare_digest(digest, selection.candidate_digest):
        raise ValueError("Selection digest mismatch")
    best = []
    for fold, frozen in zip(candidate.folds, context.split.folds, strict=True):
        if fold.error is not None:
            raise ValueError("Completed fold carries an error")
        if candidate.family == "mlp":
            if fold.inner_fit_row_count != len(frozen.inner.fit_row_ids):
                raise ValueError("Inner fit evidence does not match frozen split")
            # Recompute the stopping decision, rather than trusting a detached number.
            best_loss = float("inf")
            actual_best = 0
            non_improvements = 0
            for point in fold.loss_curve:
                if actual_best == 0 or point.val_loss < best_loss - 1e-6:
                    best_loss, actual_best = point.val_loss, point.epoch
                    non_improvements = 0
                else:
                    non_improvements += 1
                if non_improvements >= 10 and point.epoch != fold.epochs_trained:
                    raise ValueError("Validation curve continues after early stopping")
            if actual_best != fold.best_epoch:
                raise ValueError("Best epoch does not match the validation curve")
            if ((fold.stop_reason == "max_epochs" and
                 (fold.epochs_trained != run.limits.max_epochs or non_improvements >= 10)) or
                (fold.stop_reason == "early_stopping_patience" and non_improvements != 10)):
                raise ValueError("Fold stopping evidence does not match the frozen policy")
            best.append(actual_best)
    budget = min(100, max(1, math.ceil(median(best)))) if best else None
    return RefitPlan(selection, digest_bytes(canonical_bytes(asdict(candidate))), context.split.dataset_sha256,
                     digest_bytes(canonical_bytes(asdict(context.training))), budget, tuple(best))


def refit_selected(context: TrainingContext, plan: RefitPlan, candidate: CandidateResult, *,
                   control: RunControl | Callable[[], bool] | None = None) -> TrustedModel:
    """Fit fresh objects on exactly the frozen training rows, never validation/holdout."""
    if not isinstance(plan, RefitPlan) or freeze_refit(context, plan.selection, candidate) != plan:
        raise ValueError("Refit evidence or frozen plan changed")
    run = candidate.run_config
    check = control if isinstance(control, RunControl) else RunControl(cancelled=control or (lambda: False))
    check.check_cancelled()
    started = perf_counter()
    preprocessor = FoldPreprocessor(run.limits)
    with threadpool_limits(limits=run.limits.compute_threads, user_api="blas"), \
            threadpool_limits(limits=1, user_api="openmp"):
        encoded = preprocessor.fit_transform(context.training, dtype=np.float32 if plan.epoch_budget else np.float64)
        check.check_cancelled()
        if plan.epoch_budget is None:
            model = make_classical(candidate.task, candidate.family, candidate.model_seed)
            model.fit(encoded, np.asarray(context.training.targets))
            history = ()
        else:
            model = make_mlp(in_features=encoded.shape[1], seed=candidate.model_seed, task=candidate.task)
            loader = create_dataloader(encoded, np.asarray(context.training.targets, dtype=np.float32),
                                       batch_size=32, shuffle=True, seed=candidate.model_seed)
            with threadpool_limits(limits=1, user_api="blas"):
                result = train_mlp(model, loader, None, max_epochs=plan.epoch_budget,
                                   compute_threads=run.limits.compute_threads, run_control=check,
                                   restore_best_weights=False)
            if result.epochs_trained != plan.epoch_budget or len(result.history) != plan.epoch_budget:
                raise ValueError("Final training did not complete the frozen epoch budget")
            history = result.history
        check.check_cancelled()
    rows = digest_bytes(canonical_bytes(context.training.row_ids))
    fit_seconds = perf_counter() - started
    return TrustedModel(model, preprocessor, plan, rows, history, fit_seconds,
                        _model_tag(model, preprocessor, plan, rows, history, fit_seconds))


# A process-local authentication key makes a handle issued by the writer, not a
# caller-supplied path/manifest, the authority for verification. This does not
# protect against the local OS user or hostile code in the same Python process.
@dataclass(frozen=True, slots=True)
class InternalArtifact:
    directory: Path
    manifest_sha256: str
    _tag: bytes = field(repr=False)


def _tag(directory: Path, manifest_sha: str) -> bytes:
    return hmac.new(_HANDLE_KEY, canonical_bytes([str(directory), manifest_sha]), hashlib.sha256).digest()


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_pickle(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())


def _artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_internal_artifact(private_root: Path, model: TrustedModel) -> InternalArtifact:
    """Write locally generated objects only; no external artifact is accepted."""
    if not isinstance(model, TrustedModel):
        raise TypeError("Live locally refitted model required")
    if not hmac.compare_digest(model._tag, _model_tag(model.model, model.preprocessor, model.plan,
                                                     model.train_row_ids_sha256, model.training_loss, model.fit_seconds)):
        raise PermissionError("Unknown live model origin")
    root = Path(private_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    staging = root / ("staging-" + identifier)
    final = root / identifier
    staging.mkdir()
    _write_pickle(staging / "preprocessor.pkl", model.preprocessor)
    files = {"preprocessor.pkl": _artifact_digest(staging / "preprocessor.pkl")}
    if model.plan.selection.family == "mlp":
        parameters = {"input_features": model.model.net[0].in_features, "hidden_dims": [64, 32],
                      "batch_size": 32, "optimizer": "Adam", "learning_rate": 0.001,
                      "loss": "BCEWithLogitsLoss" if model.plan.selection.task.value == "binary_classification" else "MSELoss"}
        with (staging / "weights.pt").open("xb") as stream:
            torch.save(model.model.state_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        name = "weights.pt"
    else:
        parameters = model.model.get_params(deep=False)
        name = "model.pkl"
        _write_pickle(staging / name, model.model)
    files[name] = _artifact_digest(staging / name)
    selection = model.plan.selection
    metadata = {"schema_version": 1, "origin": "modelarena-local-refit-v1",
                "selection_digest": selection.candidate_digest, "candidate_id": selection.candidate_id,
                "finalization_id": selection.finalization_id, "split_sha256": selection.split_sha256,
                "config_sha256": selection.config_sha256, "family": selection.family, "task": selection.task,
                "model_seed": selection.model_seed, "resource_limits": asdict(selection.resource_limits),
                "dataset_sha256": model.plan.dataset_sha256, "dataset_config": asdict(model.preprocessor._config),
                "training_view_sha256": model.plan.training_view_sha256,
                "model_parameters": parameters,
                "evidence_sha256": model.plan.evidence_sha256, "best_epochs": model.plan.best_epochs,
                "epoch_budget": model.plan.epoch_budget, "train_row_ids_sha256": model.train_row_ids_sha256,
                "fit_seconds": model.fit_seconds,
                "training_loss": [asdict(point) for point in model.training_loss],
                "artifact_sha256": files, "provenance": execution_provenance()}
    _write_bytes(staging / "manifest.json", canonical_bytes(metadata))
    manifest_sha = file_digest(staging / "manifest.json")
    staging.rename(final)
    return InternalArtifact(final, manifest_sha, _tag(final, manifest_sha))


def verify_internal_artifact(handle: InternalArtifact) -> dict:
    """Verify a process-issued handle and checksums, without deserializing models."""
    if not isinstance(handle, InternalArtifact):
        raise TypeError("Process-issued artifact handle required")
    if not hmac.compare_digest(handle._tag, _tag(handle.directory, handle.manifest_sha256)):
        raise PermissionError("Unknown artifact origin")
    if file_digest(handle.directory / "manifest.json") != handle.manifest_sha256:
        raise ValueError("Artifact manifest checksum mismatch")
    metadata = json.loads((handle.directory / "manifest.json").read_text(encoding="utf-8"))
    expected = {"preprocessor.pkl", "weights.pt" if metadata.get("family") == "mlp" else "model.pkl"}
    if (metadata.get("schema_version") != 1 or metadata.get("origin") != "modelarena-local-refit-v1"
            or set(metadata.get("artifact_sha256", {})) != expected):
        raise ValueError("Unknown artifact origin or layout")
    for name, digest in metadata["artifact_sha256"].items():
        if _artifact_digest(handle.directory / name) != digest:
            raise ValueError("Artifact checksum mismatch")
    return metadata
