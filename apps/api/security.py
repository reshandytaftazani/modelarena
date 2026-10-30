"""Request correlation and sanitized errors, including trusted-host failures."""

from contextvars import ContextVar
from uuid import uuid4

from starlette.responses import JSONResponse


request_id: ContextVar[str | None] = ContextVar("modelarena_request_id", default=None)


def error_response(status: int, code: str, details: list[dict[str, str]] | None = None) -> JSONResponse:
    messages = {400: "Invalid request.", 403: "Request denied.", 404: "Resource not found.",
                405: "Method not allowed.", 409: "Request conflicts with current state.",
                413: "Request exceeds a size limit.", 422: "Request validation failed.",
                500: "The request could not be completed."}
    return JSONResponse(status_code=status, content={"error": {"code": code,
        "message": messages.get(status, "The request could not be completed."),
        "details": details or [], "request_id": request_id.get()}})


class RequestContextMiddleware:
    """Never reflect caller IDs or exception text; wrap host failures as JSON."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        identity = str(uuid4())
        token = request_id.set(identity)
        started = False
        replace_body = False

        async def send_correlated(message):
            nonlocal started, replace_body
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                replace_body = message["status"] == 400 and any(
                    name.lower() == b"content-type" and value.lower().startswith(b"text/plain")
                    for name, value in headers)
                if replace_body:
                    headers = [(name, value) for name, value in headers
                               if name.lower() not in (b"content-type", b"content-length")]
                    headers.append((b"content-type", b"application/json"))
                headers = [(name, value) for name, value in headers
                           if name.lower() not in (b"x-request-id", b"cache-control")]
                headers.extend([(b"x-request-id", identity.encode("ascii")), (b"cache-control", b"no-store")])
                message = {**message, "headers": headers}
                started = True
            elif message["type"] == "http.response.body" and replace_body:
                response = error_response(400, "host_rejected")
                message = {"type": "http.response.body", "body": response.body, "more_body": False}
                replace_body = False
            await send(message)

        try:
            hosts = [value for name, value in scope.get("headers", []) if name.lower() == b"host"]
            pieces = hosts[0].lower().split(b":") if len(hosts) == 1 else []
            valid_host = (len(pieces) in (1, 2) and pieces[0] in (b"localhost", b"127.0.0.1")
                          and (len(pieces) == 1 or (1 <= len(pieces[1]) <= 5 and pieces[1].isdigit()
                                                    and 1 <= int(pieces[1]) <= 65535)))
            if not valid_host:
                await error_response(400, "host_rejected")(scope, receive, send_correlated)
                return
            await self.app(scope, receive, send_correlated)
        except Exception:
            if started:
                raise
            await error_response(500, "internal_error")(scope, receive, send_correlated)
        finally:
            request_id.reset(token)
