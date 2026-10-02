"""Shared error contract for HTTP boundaries and application operations."""

import logging
import sqlite3

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def response(self) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if self.status == 401 else None
        return JSONResponse(
            {"error": {"code": self.code, "message": self.message}},
            status_code=self.status,
            headers=headers,
        )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(sqlite3.Error)
    async def storage_error(request: Request, exc: sqlite3.Error) -> JSONResponse:
        logging.getLogger(__name__).exception("Database operation failed", exc_info=exc)
        return APIError(
            503, "storage_unavailable", "Alert storage is temporarily unavailable."
        ).response()

    @app.exception_handler(APIError)
    async def application_error(request: Request, exc: APIError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Never reflect network features, unknown field names, or validation context.
        return APIError(
            422, "invalid_request", "Request does not match the schema or semantic constraints."
        ).response()

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        response = APIError(exc.status_code, "http_error", str(exc.detail)).response()
        response.headers.update(exc.headers or {})
        return response
