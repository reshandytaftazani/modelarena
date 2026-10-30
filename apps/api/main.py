"""Restricted local transport for persisted job state and CV comparisons.

This factory does not start a server or worker. Upload, configuration, queueing
and deletion endpoints are intentionally absent until their storage boundaries
can be implemented and verified.
"""

from dataclasses import asdict
import hmac
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from apps.api.database.repository import ConflictError, JobRecord, JobRepository, NotFoundError
from apps.api.services.leaderboard import get_leaderboard


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


def _job(record: JobRecord) -> JobResponse:
    return JobResponse.model_validate(asdict(record))


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
               allowed_origins: tuple[str, ...] = ()) -> FastAPI:
    """Create an app with an externally generated, secret per-session token.

    The caller must bind the server to loopback and share the token only with the
    local client. No endpoint discloses the token.
    """
    if not isinstance(repository, JobRepository):
        raise TypeError("JobRepository required")
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
            if method not in {"GET", "POST"} or not headers <= {"x-modelarena-token"}:
                return _error(403, "preflight_rejected")
            return JSONResponse(content={}, headers={"Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST", "Access-Control-Allow-Headers": "X-ModelArena-Token",
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

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        return _job(repository.get_job(_opaque_id(job_id)))

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse)
    async def cancel_job(job_id: str, request: Request):
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 1024:
                return _error(413, "body_too_large")
        if size:
            return _error(422, "unexpected_body")
        return _job(await run_in_threadpool(repository.request_cancel, _opaque_id(job_id)))

    @app.get("/api/v1/experiments/{experiment_id}/leaderboard")
    def leaderboard(experiment_id: str):
        return asdict(get_leaderboard(repository, _opaque_id(experiment_id)))

    return app
