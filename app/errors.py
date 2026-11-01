"""统一错误类型与 FastAPI 异常处理。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("tts_studio.errors")


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        *,
        request_id: str | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.detail = detail


def make_error(
    code: str,
    message: str,
    request_id: str | None,
    detail: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message, "request_id": request_id or ""}
    if detail is not None:
        payload["detail"] = detail
    return {"error": payload}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or ""


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        request_id = exc.request_id or _request_id(request)
        return JSONResponse(
            status_code=exc.status_code,
            content=make_error(exc.code, exc.message, request_id, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        detail = [
            {"loc": list(error.get("loc", [])), "msg": error.get("msg", ""), "type": error.get("type", "")}
            for error in errors
        ]
        code = (
            "INVALID_CONTROL_COMBINATION"
            if any(error.get("type") == "invalid_control_combination" for error in errors)
            else "INVALID_REQUEST"
        )
        return JSONResponse(
            status_code=400,
            content=make_error(code, "请求参数不合法", _request_id(request), detail),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(
            exc.status_code, "HTTP_ERROR"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=make_error(code, str(exc.detail), _request_id(request)),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception("unhandled error request_id=%s", request_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=make_error("GENERATION_FAILED", "服务内部错误，请查看日志", request_id),
        )
