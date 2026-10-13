import asyncio

import pytest
from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Scope

from server.core.config import Settings
from server.main import create_app
from server.pipeline.spark import SparkUnavailableError, ensure_supported_python_version


def _request(
    app: ASGIApp,
    *,
    method: str,
    path: str,
    headers: list[tuple[bytes, bytes]],
    messages: list[Message] | None = None,
) -> list[Message]:
    messages = messages if messages is not None else []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }

    asyncio.run(app(scope, receive, send))
    return messages


def _response_headers(messages: list[Message]) -> dict[bytes, bytes]:
    response = next(message for message in messages if message["type"] == "http.response.start")
    return dict(response["headers"])


def test_local_spark_defaults_to_two_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPARK_MASTER", raising=False)

    settings = Settings.from_environment()

    assert settings.spark_master == "local[2]"
    assert settings.spark_driver_memory == "4g"


def test_spark_rejects_an_unsupported_python_version() -> None:
    with pytest.raises(SparkUnavailableError, match="requires Python 3.12"):
        ensure_supported_python_version((3, 14))


def test_cors_preflight_allows_the_local_client() -> None:
    messages = _request(
        create_app(),
        method="OPTIONS",
        path="/api/v1/health",
        headers=[
            (b"origin", b"http://localhost:3000"),
            (b"access-control-request-method", b"GET"),
        ],
    )

    response = next(message for message in messages if message["type"] == "http.response.start")
    assert response["status"] == 200
    headers = _response_headers(messages)
    assert headers[b"access-control-allow-origin"] == b"http://localhost:3000"
    assert headers[b"access-control-allow-credentials"] == b"true"


def test_cors_headers_are_present_on_an_unhandled_error() -> None:
    app = create_app()
    assert isinstance(app.app, FastAPI)

    @app.app.get("/__test_unhandled_error")
    def raise_unhandled_error() -> None:
        raise RuntimeError("expected test failure")

    messages: list[Message] = []
    with pytest.raises(RuntimeError, match="expected test failure"):
        _request(
            app,
            method="GET",
            path="/__test_unhandled_error",
            headers=[(b"origin", b"http://localhost:3000")],
            messages=messages,
        )

    headers = _response_headers(messages)
    assert headers[b"access-control-allow-origin"] == b"http://localhost:3000"
