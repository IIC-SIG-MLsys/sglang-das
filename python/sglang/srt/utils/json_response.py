"""Utilities for JSON HTTP responses and SSE frames."""

from typing import Any, overload

import orjson
from fastapi.responses import Response
from pydantic import BaseModel

# Keep response serialization behavior consistent across endpoints:
# - Support non-string dictionary keys used in some metadata payloads.
# - Support numpy scalars/arrays without pre-conversion.
ORJSON_RESPONSE_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def dumps_json(content: Any) -> bytes:
    """Serialize content to JSON bytes using SGLang's ORJSON options."""
    return orjson.dumps(content, option=ORJSON_RESPONSE_OPTIONS)


@overload
def format_sse_json(payload: bytes) -> bytes: ...


@overload
def format_sse_json(payload: str) -> str: ...


def format_sse_json(payload: bytes | str) -> bytes | str:
    """Frame serialized JSON without triggering legacy PD routers' [DONE] scan."""
    # In valid JSON the match is always inside a string (including object keys).
    # RFC 8259 allows the colon to be written as \u003a with the same decoded value.
    if isinstance(payload, bytes):
        payload = payload.replace(b"data: [DONE]", rb"data\u003a [DONE]")
        return b"data: " + payload + b"\n\n"
    payload = payload.replace("data: [DONE]", r"data\u003a [DONE]")
    return f"data: {payload}\n\n"


class SGLangORJSONResponse(Response):
    """ORJSON response with SGLang-specific serialization options."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return dumps_json(content)


def orjson_response(content: Any, status_code: int = 200) -> Response:
    """Create a JSON response with stable ORJSON serialization options."""
    return SGLangORJSONResponse(content=content, status_code=status_code)


def model_json_response(content: Any) -> Any:
    """Serialize a pydantic result in one pass, skipping FastAPI's jsonable_encoder walk."""
    if not isinstance(content, BaseModel):
        return content
    return Response(
        content=content.model_dump_json(by_alias=True),
        media_type="application/json",
    )
