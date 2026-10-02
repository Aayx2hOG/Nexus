"""Transport limits independent of endpoint schemas and business logic."""

import asyncio
import json
import math
from dataclasses import dataclass

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nexus.errors import APIError

JSON_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


@dataclass(frozen=True)
class RequestLimits:
    max_body_bytes: int = 256 * 1024
    body_timeout_seconds: float = 10
    max_json_depth: int = 16


def check_json_depth(body: bytes, maximum: int) -> None:
    depth = 0
    quoted = escaped = False
    for char in body:
        if quoted:
            if escaped:
                escaped = False
            elif char == ord("\\"):
                escaped = True
            elif char == ord('"'):
                quoted = False
        elif char == ord('"'):
            quoted = True
        elif char in (ord("["), ord("{")):
            depth += 1
            if depth > maximum:
                raise ValueError("excessive JSON nesting")
        elif char in (ord("]"), ord("}")):
            depth -= 1


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def reject_constant(value: str) -> None:
    raise ValueError("nonstandard JSON constant")


def validate_json(body: bytes, limits: RequestLimits) -> None:
    try:
        check_json_depth(body, limits.max_json_depth)
        parsed = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
        if not isinstance(parsed, dict):
            raise ValueError("expected object")
    except (ValueError, RecursionError):
        raise APIError(
            400,
            "invalid_json",
            "Expected a bounded UTF-8 JSON object with unique keys and finite numbers.",
        ) from None


def validate_headers(scope: Scope, limits: RequestLimits) -> int | None:
    headers = Headers(scope=scope)
    for name in ("content-type", "content-length", "content-encoding", "transfer-encoding"):
        if len(headers.getlist(name)) > 1:
            raise APIError(400, "ambiguous_headers", "Duplicate content headers are not supported.")
    if "content-encoding" in headers:
        raise APIError(415, "unsupported_encoding", "Encoded bodies are not supported.")
    length = headers.get("content-length")
    if length is not None:
        if "transfer-encoding" in headers:
            raise APIError(400, "ambiguous_headers", "Conflicting body framing headers.")
        if not length.isascii() or not length.isdigit() or len(length) > 10:
            raise APIError(400, "invalid_length", "Invalid Content-Length.")
        if int(length) > limits.max_body_bytes:
            raise APIError(413, "body_too_large", "Request body exceeds the size limit.")
    if scope["method"] in JSON_BODY_METHODS:
        media = headers.get("content-type", "").lower().split(";")
        if (
            media[0].strip() != "application/json"
            or any(
                parameter.strip() not in {"charset=utf-8", 'charset="utf-8"'}
                for parameter in media[1:]
            )
            or len(media) > 2
        ):
            raise APIError(
                415, "unsupported_media_type", "Use application/json with UTF-8 encoding."
            )
    return int(length) if length is not None else None


async def read_body(receive: Receive, limits: RequestLimits) -> bytes | None:
    body = bytearray()
    try:
        async with asyncio.timeout(limits.body_timeout_seconds):
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return None
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > limits.max_body_bytes:
                    raise APIError(413, "body_too_large", "Request body exceeds the size limit.")
                body.extend(chunk)
                if not message.get("more_body", False):
                    return bytes(body)
    except TimeoutError:
        raise APIError(408, "body_timeout", "Request body exceeded the read deadline.") from None


class StrictRequests:
    def __init__(self, app: ASGIApp, limits: RequestLimits) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            length = validate_headers(scope, self.limits)
            body = await read_body(receive, self.limits)
            if body is None:
                return
            if length is not None and length != len(body):
                raise APIError(400, "invalid_length", "Content-Length does not match the body.")
            if scope["method"] in JSON_BODY_METHODS:
                validate_json(body, self.limits)
            elif body:
                raise APIError(400, "unexpected_body", "This method does not accept a body.")
        except APIError as exc:
            await exc.response()(scope, receive, send)
            return

        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)
