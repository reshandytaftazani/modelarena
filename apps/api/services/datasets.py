"""Private, bounded multipart CSV ingestion; client filenames are discarded."""

import os
from pathlib import Path
from uuid import UUID, uuid4

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError, _now
from ml.contracts import ResourceLimits
from ml.data.csv_io import validate_csv


class UploadError(ValueError):
    def __init__(self, code: str, status: int = 422, *, details: list[dict[str, str]] | None = None):
        self.code, self.status = code, status
        self.details = details or []
        super().__init__(code)


def upload_root(repository: JobRepository) -> Path:
    root = repository.path.parent / "uploads"
    root.mkdir(exist_ok=True)
    if root.is_symlink() or root.resolve() != root.absolute():
        raise UploadError("unsafe_storage", 409)
    return root


class MultipartUpload:
    """Exactly one file part, streamed directly into an exclusive private file."""

    def __init__(self, root: Path, content_type: str, limits: ResourceLimits):
        kind, options = parse_options_header(content_type.encode("latin-1"))
        boundary = options.get(b"boundary", b"")
        if kind.lower() != b"multipart/form-data" or not 1 <= len(boundary) <= 200:
            raise UploadError("invalid_multipart")
        self.path = root / ("staging-" + uuid4().hex + ".csv")
        self.limits = limits
        self.parts = self.size = self.total = 0
        self.complete = False
        self.field = bytearray()
        self.value = bytearray()
        self.headers = {}
        self.parser = MultipartParser(boundary, callbacks={
            "on_part_begin": self.part_begin, "on_header_begin": self.header_begin,
            "on_header_field": self.header_field, "on_header_value": self.header_value,
            "on_header_end": self.header_end, "on_headers_finished": self.headers_finished,
            "on_part_data": self.data, "on_end": self.end,
        })
        self.file = self.path.open("xb")

    def part_begin(self):
        self.parts += 1
        if self.parts != 1:
            raise UploadError("invalid_multipart")

    def header_begin(self):
        self.field.clear()
        self.value.clear()

    def header_field(self, data, start, end):
        self.field.extend(data[start:end])
        if len(self.field) > 1024:
            raise UploadError("invalid_multipart")

    def header_value(self, data, start, end):
        self.value.extend(data[start:end])
        if len(self.value) > 8192:
            raise UploadError("invalid_multipart")

    def header_end(self):
        name = bytes(self.field).lower()
        if name in self.headers or len(self.headers) >= 10:
            raise UploadError("invalid_multipart")
        self.headers[name] = bytes(self.value)

    def headers_finished(self):
        disposition, options = parse_options_header(self.headers.get(b"content-disposition", b""))
        if disposition.lower() != b"form-data" or options.get(b"name") != b"file" or b"filename" not in options:
            raise UploadError("invalid_multipart")

    def data(self, chunk, start, end):
        self.size += end - start
        if self.size > self.limits.upload_bytes:
            raise UploadError("upload_too_large", 413)
        self.file.write(chunk[start:end])

    def end(self):
        self.complete = True

    def write(self, chunk: bytes):
        self.total += len(chunk)
        if self.total > self.limits.upload_bytes + 65536:
            raise UploadError("upload_too_large", 413)
        self.parser.write(chunk)

    def finish(self):
        self.parser.finalize()
        if not self.complete or self.parts != 1:
            raise UploadError("invalid_multipart")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()

    def cleanup(self):
        self.file.close()
        self.path.unlink(missing_ok=True)


def publish_upload(repository: JobRepository, path: Path, limits: ResourceLimits) -> dict:
    root = upload_root(repository)
    data = validate_csv(path, limits, expected_root=root)
    created = None
    folder_created = False
    try:
        with repository._transaction() as db:
            existing = db.execute("SELECT dataset_id FROM datasets WHERE raw_sha256=?", (data.raw_sha256,)).fetchone()
            dataset_id = existing[0] if existing else str(uuid4())
            if db.execute("SELECT 1 FROM deletion_intents i JOIN datasets d ON d.dataset_id=i.dataset_id "
                          "WHERE d.canonical_sha256=?", (data.canonical_sha256,)).fetchone():
                raise ConflictError("Dataset deletion is pending")
            folder = root / dataset_id
            if folder.exists() and (folder.is_symlink() or folder.resolve() != folder.absolute()):
                raise UploadError("unsafe_storage", 409)
            if not folder.exists():
                folder.mkdir()
                folder_created = True
            destination = folder / "source.csv"
            if destination.exists():
                saved = validate_csv(destination, limits, expected_root=root)
                if saved.raw_sha256 != data.raw_sha256:
                    raise ConflictError("Stored dataset identity does not match")
            else:
                path.rename(destination)
                created = destination
            if not existing:
                db.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?)", (
                    dataset_id, data.raw_sha256, data.canonical_sha256, data.parse_policy_version,
                    data.byte_count, data.row_count, len(data.columns), _now()))
    except BaseException:
        if created:
            created.unlink(missing_ok=True)
        if folder_created:
            folder.rmdir()
        raise
    return {"dataset_id": dataset_id, "raw_sha256": data.raw_sha256,
            "canonical_sha256": data.canonical_sha256, "columns": data.columns,
            "row_count": data.row_count, "byte_count": data.byte_count}


def load_dataset(repository: JobRepository, dataset_id: str, limits: ResourceLimits):
    try:
        if str(UUID(dataset_id)) != dataset_id:
            raise ValueError
    except ValueError:
        raise NotFoundError("Dataset not found") from None
    with repository._connection() as db:
        row = db.execute("SELECT raw_sha256,canonical_sha256 FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        pending = db.execute("SELECT 1 FROM deletion_intents WHERE dataset_id=?", (dataset_id,)).fetchone()
    if row is None:
        raise NotFoundError("Dataset not found")
    if pending:
        raise ConflictError("Dataset deletion is pending")
    root = upload_root(repository)
    path = root / dataset_id / "source.csv"
    data = validate_csv(path, limits, expected_root=root)
    if data.raw_sha256 != row["raw_sha256"] or data.canonical_sha256 != row["canonical_sha256"]:
        raise ConflictError("Stored dataset does not match its identity")
    return data
