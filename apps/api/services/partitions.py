"""Private immutable partition storage for execution inputs.

This module writes partitions during preparation but deliberately provides no
holdout reader. Finalization's gated reader verifies the held-out checksum.
"""

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from uuid import UUID, uuid4

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError
from ml.contracts import DatasetConfig, ResourceLimits, SplitManifest, TrainingView
from ml.data.csv_io import ValidatedDataset
from ml.data.splits import verify_manifest
from ml.data.validation import configure_dataset


def _folder(repository: JobRepository, experiment_id: str, *, db=None) -> Path:
    with repository._connection() if db is None else nullcontext(db) as db:
        row = db.execute(
            "SELECT c.scope_id,c.split_id,EXISTS(SELECT 1 FROM deletion_intents i "
            "JOIN datasets d ON d.dataset_id=i.dataset_id JOIN datasets source "
            "ON source.canonical_sha256=d.canonical_sha256 WHERE source.dataset_id=c.dataset_id) AS pending "
            "FROM experiments e JOIN configurations c ON c.configuration_id=e.configuration_id "
            "WHERE e.experiment_id=?", (experiment_id,),
        ).fetchone()
    if row is None:
        raise NotFoundError("Experiment not found")
    if row["pending"]:
        raise ConflictError("Dataset deletion is pending")
    for value in (row["scope_id"], row["split_id"]):
        if str(UUID(value)) != value:
            raise ValueError("Invalid partition identity")
    folder = repository.path.parent / "datasets" / row["scope_id"] / row["split_id"]
    if folder.resolve() != folder.absolute():
        raise ValueError("Unsafe partition path")
    return folder


def _payload(data: ValidatedDataset, split: SplitManifest, partition: str) -> bytes:
    ids = split.train_row_ids if partition == "train" else split.holdout_row_ids
    return json.dumps({"columns": data.columns, "partition": partition,
                       "records": [(i, data.rows[i]) for i in ids]},
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


def save_training_partition(repository: JobRepository, experiment_id: str,
                            data: ValidatedDataset, split: SplitManifest) -> None:
    """Publish no-clobber partitions whose hashes are frozen in the split."""
    if not verify_manifest(split) or data.canonical_sha256 != split.dataset_sha256:
        raise ValueError("Invalid partition source")
    contents = []
    for name, expected in (("train", split.train_sha256), ("holdout", split.holdout_sha256)):
        content = _payload(data, split, name)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Partition source does not match frozen split")
        contents.append((name, content))
    # The deletion-intent transaction uses the same write lock. Keep the
    # intent check and file publication indivisible relative to cleanup: a
    # deletion either sees these files or prevents their creation entirely.
    # Payload generation stays outside the transaction; no fitting occurs here.
    with repository._transaction() as db:
        folder = _folder(repository, experiment_id, db=db)
        folder.mkdir(parents=True, exist_ok=True)
        if folder.resolve() != folder.absolute():
            raise ValueError("Unsafe partition path")
        for name, content in contents:
            _publish(folder, name, content)


def _publish(folder: Path, name: str, content: bytes) -> None:
    destination = folder / (name + ".json")
    temporary = folder / ("staging-" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.resolve() != destination.absolute():
                raise ValueError("Unsafe partition file")
            # Never reopen the holdout on repeated preparation. Its checksum is
            # checked by the gated reader after durable evaluation-start.
            if name == "train" and destination.read_bytes() != content:
                raise ConflictError("Frozen training partition differs")
    finally:
        temporary.unlink(missing_ok=True)


def load_training_partition(repository: JobRepository, experiment_id: str, config: DatasetConfig,
                            split: SplitManifest, limits: ResourceLimits) -> TrainingView:
    """Verify and materialize only the saved training rows; never fall back to source."""
    if not verify_manifest(split):
        raise ValueError("Invalid split manifest")
    path = _folder(repository, experiment_id) / "train.json"
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("Unsafe partition file")
    # JSON escaping may expand each input byte sixfold; identities add overhead.
    with path.open("rb") as stream:
        content = stream.read(limits.upload_bytes * 6 + limits.max_rows * 32 + 65537)
    if len(content) > limits.upload_bytes * 6 + limits.max_rows * 32 + 65536:
        raise ValueError("Training partition exceeds bounds")
    if hashlib.sha256(content).hexdigest() != split.train_sha256:
        raise ValueError("Training partition checksum mismatch")
    payload = json.loads(content)
    if set(payload) != {"columns", "partition", "records"} or payload["partition"] != "train":
        raise ValueError("Invalid training partition")
    records = payload["records"]
    if tuple(record[0] for record in records) != split.train_row_ids:
        raise ValueError("Training identities differ from frozen split")
    columns = tuple(payload["columns"])
    rows = tuple(tuple(record[1]) for record in records)
    if (not columns or len(set(columns)) != len(columns)
            or any(not isinstance(name, str) for name in columns)
            or any(len(row) != len(columns) or any(not isinstance(v, str) for v in row) for row in rows)):
        raise ValueError("Invalid training records")
    data = ValidatedDataset(columns, rows, len(content), "", split.dataset_sha256)
    configured = configure_dataset(data, config)
    return TrainingView(config, split.train_row_ids, configured.rows, configured.targets, configured.label_mapping)
