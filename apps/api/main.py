"""Restricted local transport for persisted job state and CV comparisons.

This factory does not start a server or worker. Deletion remains incomplete.
"""

from dataclasses import asdict
import hmac
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from python_multipart.exceptions import MultipartParseError
from starlette.concurrency import run_in_threadpool

from apps.api.database.repository import ConflictError, JobRecord, JobRepository, NotFoundError
from apps.api.services.leaderboard import get_candidate, get_leaderboard
from apps.api.services.datasets import MultipartUpload, UploadError, publish_upload, upload_root
from apps.api.services.workflow import configuration, configure, create_experiment
from apps.api.services.retention import deletion_disclosure, request_deletion
from apps.api.workers.presence import worker_status
from ml.contracts import DatasetConfig, FeatureColumn, FeatureSchema, FeatureType, ResourceLimits, TaskType
from ml.data.validation import CsvValidationError


class WorkProgress(BaseModel):
    family: str
    fold_index: int
    finished_units: int
    total_units: int


class JobResponse(BaseModel):
    job_id: str
    experiment_id: str
    kind: str
    state: str
    cancel_requested: bool
    stage: str
    created_at: str
    updated_at: str
    heartbeat_at: str | None
    safe_error: str | None
    progress: WorkProgress | None = None


class DatasetResponse(BaseModel):
    dataset_id: str
    raw_sha256: str
    canonical_sha256: str
    columns: tuple[str, ...]
    row_count: int
    byte_count: int


class FeatureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=1024)
    dtype: FeatureType


class ConfigurationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: TaskType
    target: str = Field(min_length=1, max_length=1024)
    features: list[FeatureInput] = Field(min_length=1, max_length=200)
    positive_label: str | None = Field(default=None, max_length=65536)
    acknowledge_duplicates: bool = Field(default=False, strict=True)
    split_seed: int = Field(default=42, strict=True, ge=0, lt=2**32)


class ExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    configuration_id: str = Field(max_length=36)
    primary_metric: str | None = Field(default=None, max_length=32)
    model_seed: int = Field(default=42, strict=True, ge=0, lt=2**32)


class QueueInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: str = Field(min_length=1, max_length=200)


class DeletionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    acknowledge: bool = Field(strict=True)


async def _body(request: Request, schema: type[BaseModel]) -> BaseModel:
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > 65536:
            raise UploadError("body_too_large", 413)
        data.extend(chunk)
    try:
        return schema.model_validate_json(data)
    except ValidationError:
        raise UploadError("invalid_request") from None


def _job(repository: JobRepository, record: JobRecord) -> JobResponse:
    return JobResponse.model_validate({**asdict(record), "progress": repository.get_work_progress(record.job_id)})


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code}})


def _opaque_id(value: str) -> str:
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except ValueError:
        raise NotFoundError("Not found") from None
    return value


def create_app(repository: JobRepository, *, session_token: str,
               allowed_origins: tuple[str, ...] = (), limits: ResourceLimits | None = None) -> FastAPI:
    """Create an app with an externally generated, secret per-session token.

    The caller must bind the server to loopback and share the token only with the
    local client. No endpoint discloses the token.
    """
    if not isinstance(repository, JobRepository):
        raise TypeError("JobRepository required")
    limits = limits or ResourceLimits()
    if not isinstance(limits, ResourceLimits):
        raise ValueError("ResourceLimits required")
    if not isinstance(session_token, str) or len(session_token) < 32 or not session_token.isascii():
        raise ValueError("A secret session token of at least 32 ASCII characters is required")
    for origin in allowed_origins:
        parsed = urlsplit(origin)
        if (parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1")
                or parsed.username is not None or parsed.password is not None or parsed.path
                or parsed.query or parsed.fragment or parsed.netloc.endswith(":")):
            raise ValueError("Origins must be exact loopback origins")
        try:
            parsed.port
        except ValueError:
            raise ValueError("Invalid origin port") from None

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        origins = request.headers.getlist("origin")
        if len(origins) > 1 or (origins and origins[0] not in allowed_origins):
            return _error(403, "origin_rejected")
        origin = origins[0] if origins else None
        # Browser preflight contains no session token; permit only an exact
        # configured origin and narrowly scoped methods and request headers.
        if request.method == "OPTIONS" and origin:
            method = request.headers.get("access-control-request-method", "")
            headers = {h.strip().lower() for h in request.headers.get("access-control-request-headers", "").split(",") if h.strip()}
            if method not in {"GET", "POST", "DELETE"} or not headers <= {"x-modelarena-token", "content-type"}:
                return _error(403, "preflight_rejected")
            return JSONResponse(content={}, headers={"Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, DELETE", "Access-Control-Allow-Headers": "X-ModelArena-Token, Content-Type",
                "Vary": "Origin"})
        tokens = request.headers.getlist("x-modelarena-token")
        if len(tokens) != 1 or not tokens[0].isascii() or not hmac.compare_digest(tokens[0], session_token):
            return _error(403, "token_rejected")
        response = await call_next(request)
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    # Outermost middleware: reject untrusted Host even for preflight requests.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"], www_redirect=False)

    @app.exception_handler(NotFoundError)
    async def not_found(_request: Request, _exc: NotFoundError):
        return _error(404, "not_found")

    @app.exception_handler(ConflictError)
    async def conflict(_request: Request, _exc: ConflictError):
        return _error(409, "conflict")

    @app.exception_handler(UploadError)
    async def invalid_upload(_request: Request, exc: UploadError):
        return _error(exc.status, exc.code)

    @app.exception_handler(CsvValidationError)
    async def invalid_csv(_request: Request, exc: CsvValidationError):
        return _error(413 if exc.code == "upload_too_large" else 422, exc.code)

    @app.get("/health")
    def health():
        return {"api_version": "v1", "storage": "available", "worker": worker_status(repository)}

    @app.post("/api/v1/datasets", response_model=DatasetResponse, status_code=201)
    async def upload_dataset(request: Request):
        upload = None
        try:
            root = await run_in_threadpool(upload_root, repository)
            upload = await run_in_threadpool(MultipartUpload, root, request.headers.get("content-type", ""), limits)
            async for chunk in request.stream():
                await run_in_threadpool(upload.write, chunk)
            await run_in_threadpool(upload.finish)
            return await run_in_threadpool(publish_upload, repository, upload.path, limits)
        except UploadError as exc:
            return _error(exc.status, exc.code)
        except CsvValidationError as exc:
            return _error(413 if exc.code == "upload_too_large" else 422, exc.code)
        except (MultipartParseError, UnicodeError):
            return _error(422, "invalid_multipart")
        finally:
            if upload is not None:
                await run_in_threadpool(upload.cleanup)

    @app.delete("/api/v1/datasets/{dataset_id}", status_code=202)
    async def delete_dataset(dataset_id: str, request: Request):
        body = await _body(request, DeletionInput)
        if not body.acknowledge:
            return _error(422, "acknowledgement_required")
        return await run_in_threadpool(request_deletion, repository, _opaque_id(dataset_id))

    @app.get("/api/v1/datasets/{dataset_id}/deletion-disclosure")
    def get_deletion_disclosure(dataset_id: str):
        return deletion_disclosure(repository, _opaque_id(dataset_id))

    @app.post("/api/v1/datasets/{dataset_id}/configurations", status_code=201)
    async def create_configuration(dataset_id: str, request: Request):
        dataset_id = _opaque_id(dataset_id)
        body = await _body(request, ConfigurationInput)
        try:
            config = DatasetConfig(body.task, body.target,
                FeatureSchema(tuple(FeatureColumn(c.name, c.dtype) for c in body.features)),
                body.positive_label, body.acknowledge_duplicates)
            return await run_in_threadpool(configure, repository, dataset_id, config, body.split_seed, limits)
        except ValueError as exc:
            if isinstance(exc, (ConflictError, CsvValidationError)):
                raise
            return _error(422, "invalid_configuration")

    @app.get("/api/v1/configurations/{configuration_id}/profile")
    def get_profile(configuration_id: str):
        return configuration(repository, _opaque_id(configuration_id))["profile"]

    @app.post("/api/v1/experiments", status_code=201)
    async def register_experiment(request: Request):
        body = await _body(request, ExperimentInput)
        try:
            result = await run_in_threadpool(create_experiment, repository, _opaque_id(body.configuration_id),
                                             body.primary_metric, body.model_seed, limits)
            return {"experiment_id": result}
        except ValueError as exc:
            if isinstance(exc, (ConflictError, CsvValidationError)):
                raise
            return _error(422, "invalid_experiment")

    @app.post("/api/v1/experiments/{experiment_id}/jobs", response_model=JobResponse, status_code=202)
    async def enqueue(experiment_id: str, request: Request):
        body = await _body(request, QueueInput)
        if not body.idempotency_key.strip():
            return _error(422, "invalid_request")
        job = await run_in_threadpool(repository.enqueue_cv, _opaque_id(experiment_id), body.idempotency_key)
        return await run_in_threadpool(_job, repository, job)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        return _job(repository, repository.get_job(_opaque_id(job_id)))

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse)
    async def cancel_job(job_id: str, request: Request):
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 1024:
                return _error(413, "body_too_large")
        if size:
            return _error(422, "unexpected_body")
        job = await run_in_threadpool(repository.request_cancel, _opaque_id(job_id))
        return await run_in_threadpool(_job, repository, job)

    @app.post("/api/v1/jobs/{job_id}/retry", response_model=JobResponse, status_code=202)
    async def retry_job(job_id: str, request: Request):
        body = await _body(request, QueueInput)
        if not body.idempotency_key.strip():
            return _error(422, "invalid_request")
        job = await run_in_threadpool(repository.retry_cv, _opaque_id(job_id), body.idempotency_key)
        return await run_in_threadpool(_job, repository, job)

    @app.get("/api/v1/experiments/{experiment_id}/leaderboard")
    def leaderboard(experiment_id: str):
        return asdict(get_leaderboard(repository, _opaque_id(experiment_id)))

    @app.get("/api/v1/candidates/{candidate_id}")
    def candidate(candidate_id: str):
        return get_candidate(repository, _opaque_id(candidate_id))

    return app
