"""Unit tests for offline report generation, checksums, escaping, and leakage safety."""

from dataclasses import asdict, replace
import hashlib
import html
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import pytest

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError, WorkerIdentity
from apps.api.services.finalization import compute_candidate_digest, reserve
from apps.api.services.jobs import prepare_experiment
from apps.api.services.reports import build_report, verify_report_zip
from apps.api.services.retention import request_deletion
from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, MetricBundle, MetricValue, ResourceLimits, RunConfig, TaskType
from ml.data.csv_io import validate_csv
from ml.evaluation import finalize
from ml.evaluation.finalize import stage_final_bundle
from ml.tracking.publication import stage_attempt
from tests.integration.test_classical_suite import prepared_fixture


def _create_cv_attempt(repo: JobRepository, prepared, run, candidate_family="dummy", loss_curve=None):
    """Create and publish a verified CV bundle in repository storage."""
    with repo._connection() as db:
        job_row = db.execute(
            "SELECT * FROM jobs WHERE experiment_id=? AND kind='cv' ORDER BY sequence DESC LIMIT 1",
            (prepared.experiment_id,),
        ).fetchone()
        exp_row = db.execute(
            "SELECT e.primary_metric, e.direction, c.dataset_id, d.raw_sha256, d.canonical_sha256 "
            "FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN datasets d ON d.dataset_id=c.dataset_id WHERE e.experiment_id=?",
            (prepared.experiment_id,),
        ).fetchone()

    primary_metric = exp_row["primary_metric"]
    direction = exp_row["direction"]

    # Claim job
    owner = WorkerIdentity("test-worker", 1)
    claimed = repo.claim_next(owner, kind="cv")
    assert claimed is not None and claimed.job_id == job_row["job_id"]

    home = repo.path.parent
    folds = [
        {
            "fold_index": i,
            "validation_row_ids": list(prepared.context.split.folds[i].validation_row_ids),
            "status": "completed",
            "fit_seconds": 0.05,
            "score_seconds": 0.01,
            "metrics": {
                "task": "binary_classification",
                "sample_count": len(prepared.context.split.folds[i].validation_row_ids),
                "metrics": [{"name": primary_metric, "value": 0.85}],
            },
            "best_epoch": 10 if loss_curve else None,
            "epochs_trained": 15 if loss_curve else None,
            "stop_reason": "early_stopping_patience" if loss_curve else None,
            "loss_curve": loss_curve,
        }
        for i in range(len(prepared.context.split.folds))
    ]

    payload = {
        "observed_threadpools": [1],
        "comparison": {
            "ranked": [
                {
                    "family": candidate_family,
                    "mean": 0.85,
                    "std": 0.02,
                    "mean_fit_seconds": 0.05,
                    "baseline_delta": 0.15 if candidate_family != "dummy" else 0.0,
                    "tied": False,
                }
            ],
            "unranked": [],
        },
        "candidates": [
            {
                "family": candidate_family,
                "task": prepared.context.training.config.task.value,
                "split_sha256": prepared.context.split.manifest_sha256,
                "primary_metric": primary_metric,
                "direction": direction,
                "eligible": True,
                "model_seed": 42,
                "total_seconds": 0.3,
                "run_config": asdict(replace(run, family=candidate_family, primary_metric=primary_metric)),
                "summary": [{"name": primary_metric, "mean": 0.85, "std": 0.02, "valid_folds": 5}],
                "folds": folds,
            }
        ],
    }

    staged = stage_attempt(
        home,
        claimed,
        prepared.context,
        run,
        {"dataset_id": exp_row["dataset_id"], "raw_sha256": exp_row["raw_sha256"], "canonical_sha256": exp_row["canonical_sha256"]},
        payload,
        telemetry={"wall_seconds": 0.5},
    )

    repo.finish_job(owner, claimed.job_id, claimed.attempt_id, state="completed", stage="completed", publication=staged)
    return staged


def _create_final_bundle(repo: JobRepository, prepared, candidate_family="dummy"):
    """Create and publish a verified finalization bundle."""
    with repo._connection() as db:
        config_row = db.execute(
            "SELECT c.scope_id, s.manifest_sha256 FROM experiments e "
            "JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN splits s ON s.split_id=c.split_id WHERE e.experiment_id=?",
            (prepared.experiment_id,),
        ).fetchone()
        cand_row = db.execute(
            "SELECT candidate_id, config_sha256 FROM candidates WHERE experiment_id=? AND family=?",
            (prepared.experiment_id, candidate_family),
        ).fetchone()

    scope_id = config_row["scope_id"]
    candidate_id = cand_row["candidate_id"]
    digest = compute_candidate_digest(
        candidate_id, candidate_family, config_row["manifest_sha256"], cand_row["config_sha256"], "f1", 0.85, 42
    )

    selection, job = reserve(repo, prepared.experiment_id, candidate_id, digest, True)
    assert job is not None

    with repo._connection() as db:
        fin_row = db.execute(
            "SELECT evidence_sha256 FROM finalizations WHERE finalization_id=?",
            (selection.finalization_id,),
        ).fetchone()

    attempt_id = "final-attempt-1"
    home = repo.path.parent
    final_result_payload = {
        "finalization_id": selection.finalization_id,
        "scope_id": scope_id,
        "candidate_id": candidate_id,
        "candidate_digest": digest,
        "evidence_sha256": fin_row["evidence_sha256"],
        "task": TaskType.BINARY_CLASSIFICATION.value,
        "primary_metric": "f1",
        "metrics": {
            "task": TaskType.BINARY_CLASSIFICATION.value,
            "sample_count": 20,
            "metrics": [
                {"name": "f1", "value": 0.91, "reason": None},
                {"name": "precision", "value": 0.89, "reason": None},
                {"name": "recall", "value": 0.93, "reason": None},
            ],
            "confusion_matrix": [[9, 1], [1, 9]],
        },
        "evaluation_seconds": 0.04,
        "evaluated_at": "2026-11-01T03:00:00Z",
    }

    final_dir, manifest_sha = stage_final_bundle(
        home, selection.finalization_id, scope_id, attempt_id, candidate_id, final_result_payload
    )

    with repo._transaction() as db:
        db.execute(
            "UPDATE finalizations SET state='evaluated', result_checksum=?, result_path=?, attempt_id=?, evaluation_started_at=?, updated_at=? WHERE finalization_id=?",
            (manifest_sha, str(final_dir), attempt_id, "2026-11-01T03:00:00Z", "2026-11-01T03:00:00Z", selection.finalization_id),
        )
        db.execute(
            "UPDATE jobs SET state='completed', stage='completed', attempt_id=?, updated_at=? WHERE job_id=?",
            (attempt_id, "2026-11-01T03:00:00Z", job.job_id),
        )

    return selection, final_dir


def test_build_report_cv_only(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")

    report_path = build_report(repo, prepared.experiment_id)
    assert report_path.exists()
    assert report_path.suffix == ".zip"

    verified = verify_report_zip(report_path)
    manifest = verified["manifest"]
    metrics = verified["metrics"]
    curves = verified["curves"]
    html_text = verified["html"]

    # Assert status is CV only
    assert manifest["status"] == "cv_only"
    assert metrics["status"] == "cv_only"
    assert manifest["finalization"] is None
    assert metrics["final_evaluation"] is None

    # Assert distinct CV-only banner
    assert "CV only — holdout not evaluated" in html_text
    assert "Final Evaluation (Holdout Partition — Evaluated Once)" not in html_text

    # Assert CV comparison is present
    assert "Cross-Validation Leaderboard" in html_text
    assert "dummy" in html_text


def test_build_report_finalized(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")
    selection, _ = _create_final_bundle(repo, prepared, candidate_family="dummy")

    report_path = build_report(repo, prepared.experiment_id)
    assert report_path.exists()

    verified = verify_report_zip(report_path)
    manifest = verified["manifest"]
    metrics = verified["metrics"]
    html_text = verified["html"]

    # Assert status is evaluated
    assert manifest["status"] == "evaluated"
    assert metrics["status"] == "evaluated"
    assert manifest["finalization"] is not None
    assert manifest["finalization"]["candidate_id"] == selection.candidate_id
    assert metrics["final_evaluation"] is not None

    # Assert distinct Final Evaluation banner
    assert "Final Evaluation (Holdout Partition — Evaluated Once)" in html_text
    assert "CV only — holdout not evaluated" not in html_text

    # Assert final holdout metrics & confusion matrix
    assert "Holdout Confusion Matrix" in html_text
    assert "TN (45.0%)" in html_text
    assert "TP (45.0%)" in html_text
    assert "0.9100" in html_text  # F1 score


def test_report_html_escaping_for_malicious_inputs(tmp_path):
    # Create CSV with malicious XSS content in feature names, column data, and target
    csv_path = tmp_path / "xss.csv"
    csv_path.write_text(
        'x,"<script>alert(\'feat\')</script>","target"\n'
        + "\n".join(f"{i},{i % 2},'<img src=x onerror=alert(1)>'" if i % 2 else f"{i},{i % 2},'safe'" for i in range(100)),
        encoding="utf-8",
    )
    config = DatasetConfig(
        TaskType.BINARY_CLASSIFICATION,
        "target",
        FeatureSchema(
            (
                FeatureColumn("x", FeatureType.NUMERIC),
                FeatureColumn("<script>alert('feat')</script>", FeatureType.NUMERIC),
            )
        ),
        positive_label="'<img src=x onerror=alert(1)>'",
    )
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    run = RunConfig()
    prepared = prepare_experiment(repo, validate_csv(csv_path, run.limits), config, run)
    repo.enqueue_cv(prepared.experiment_id, "suite")
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")

    report_path = build_report(repo, prepared.experiment_id)
    verified = verify_report_zip(report_path)
    html_text = verified["html"]

    # Verify no raw unescaped script or img tag exists
    assert "<script>alert('feat')</script>" not in html_text
    assert "<img src=x onerror=alert(1)>" not in html_text

    # Verify escaped entities exist
    assert "&lt;script&gt;alert(&#x27;feat&#x27;)&lt;/script&gt;" in html_text or "&lt;script&gt;alert(&quot;feat&quot;)&lt;/script&gt;" in html_text


def test_no_raw_rows_no_model_binaries_no_external_assets(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")

    report_path = build_report(repo, prepared.experiment_id)

    with zipfile.ZipFile(report_path, "r") as zf:
        namelist = zf.namelist()
        for name in namelist:
            assert not any(name.endswith(ext) for ext in (".pkl", ".pt", ".pth", ".bin", ".joblib", ".exe", ".py"))

        html_content = zf.read("report.html").decode("utf-8")
        # No external script or link
        assert "<script" not in html_content.lower()
        assert "http://" not in html_content
        assert "https://" not in html_content
        assert "fonts.googleapis.com" not in html_content

        # No raw rows from input CSV (e.g. "group-0", "group-1")
        # In prepared_fixture, categories are "group-0", "group-1", "group-2"
        # Features table has counts and types, not the raw data rows
        metrics_content = zf.read("metrics.json").decode("utf-8")
        assert "group-0" not in metrics_content
        assert "group-1" not in metrics_content


def test_no_report_triggered_refit_or_holdout_read(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")

    # Poison holdout reader and refit
    def poison_read_holdout(*args, **kwargs):
        raise AssertionError("POISON: read_holdout was called during report generation!")

    def poison_refit(*args, **kwargs):
        raise AssertionError("POISON: refit was called during report generation!")

    with patch("ml.evaluation.finalize.read_holdout", side_effect=poison_read_holdout):
        with patch("ml.training.refit.refit_selected", side_effect=poison_refit):
            report_path = build_report(repo, prepared.experiment_id)
            assert report_path.exists()

    # Also test deleting or corrupting holdout partition on disk
    holdout_file = repo.path.parent / "partitions" / prepared.experiment_id / "holdout.json"
    if holdout_file.exists():
        holdout_file.write_text("CORRUPTED_POISON_DATA", encoding="utf-8")

    # build_report must succeed because it never reads holdout.json
    report_path2 = build_report(repo, prepared.experiment_id)
    assert report_path2.exists()


def test_mlp_loss_curves_rendering(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    curve = [
        {"epoch": 1, "train_loss": 0.69, "val_loss": 0.65},
        {"epoch": 2, "train_loss": 0.55, "val_loss": 0.52},
        {"epoch": 3, "train_loss": 0.45, "val_loss": 0.46},
        {"epoch": 4, "train_loss": 0.38, "val_loss": 0.42},
        {"epoch": 5, "train_loss": 0.32, "val_loss": 0.40},
    ]
    _create_cv_attempt(repo, prepared, run, candidate_family="mlp", loss_curve=curve)

    report_path = build_report(repo, prepared.experiment_id)
    verified = verify_report_zip(report_path)

    curves = verified["curves"]
    assert "mlp" in curves["curves"]
    folds = list(curves["curves"]["mlp"].values())[0]["folds"]
    assert len(folds) == 5
    assert folds[0]["loss_curve"] == curve

    html_text = verified["html"]
    assert "MLP Learning Trajectory" in html_text
    assert "<polyline" in html_text
    assert "Best: 10" in html_text or "Best Epoch" in html_text


def test_corrupted_cv_bundle_rejected(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    staged = _create_cv_attempt(repo, prepared, run, candidate_family="dummy")

    # Corrupt results.json in bundle
    bundle_path = Path(staged["path"])
    results_file = bundle_path / "results.json"
    results_file.write_text('{"corrupted": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="Results checksum mismatch"):
        build_report(repo, prepared.experiment_id)


def test_corrupted_final_bundle_rejected(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")
    selection, final_dir = _create_final_bundle(repo, prepared, candidate_family="dummy")

    # Corrupt results.json in finalization bundle
    results_file = final_dir / "results.json"
    results_file.write_text('{"corrupted": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="Results checksum mismatch"):
        build_report(repo, prepared.experiment_id)


def test_deletion_pending_conflict(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")

    with repo._connection() as db:
        ds_id = db.execute(
            "SELECT c.dataset_id FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id WHERE e.experiment_id=?",
            (prepared.experiment_id,),
        ).fetchone()["dataset_id"]
    request_deletion(repo, ds_id)

    with pytest.raises(ConflictError, match="Dataset deletion is pending"):
        build_report(repo, prepared.experiment_id)


def test_experiment_not_found(tmp_path):
    repo = JobRepository(tmp_path / "metadata.sqlite3")
    with pytest.raises(NotFoundError, match="Experiment 'non-existent' not found"):
        build_report(repo, "non-existent")


def test_symlink_output_path_rejected(tmp_path):
    repo, prepared, run, _ = prepared_fixture(tmp_path, TaskType.BINARY_CLASSIFICATION)
    _create_cv_attempt(repo, prepared, run, candidate_family="dummy")

    target_file = tmp_path / "target.zip"
    target_file.touch()
    symlink_path = tmp_path / "symlink.zip"
    try:
        symlink_path.symlink_to(target_file)
    except OSError:
        pytest.skip("Symlinks not supported on this platform/privilege level")

    with pytest.raises(ValueError, match="Unsafe output path"):
        build_report(repo, prepared.experiment_id, output_path=symlink_path)
