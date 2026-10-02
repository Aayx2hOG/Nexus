import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nexus.api import create_app
from nexus.config import Settings
from nexus.demo import demo_alerts

TEST_TOKEN = "test-token-" + "a" * 32


@pytest.fixture
def app_factory(tmp_path):
    def factory(*, token=TEST_TOKEN, path=None, limits=None):
        return create_app(
            Settings(
                database_path=path or tmp_path / "nexus.sqlite3",
                api_token=token,
                reviewer_id="test-analyst",
            ),
            limits=limits,
        )

    return factory


@pytest.fixture
def app(app_factory):
    return app_factory()


@pytest.fixture
def client(app):
    with TestClient(app, headers={"Authorization": f"Bearer {TEST_TOKEN}"}) as instance:
        yield instance


@pytest.fixture
def store(app, client):
    return app.state.store


@pytest.fixture
def payload():
    return json.loads((Path(__file__).parent.parent / "examples/flow-batch.json").read_text())


@pytest.fixture
def alert(store):
    return store.record_alert(demo_alerts()[0])


@pytest.fixture
def feedback():
    return {
        "feedback_id": str(uuid4()),
        "expected_version": 0,
        "verdict": "confirmed_attack",
        "attack_category": "Exploits",
        "notes": "Reviewed flow.",
    }


@pytest.fixture
def asgi_request():
    """One reusable harness for streamed, slow, and disconnected HTTP clients."""

    def request(app, chunks, *, delay=0, headers=None):
        async def run():
            responses = []
            messages = iter(chunks)

            async def receive():
                if delay:
                    await asyncio.sleep(delay)
                return next(messages)

            async def send(message):
                responses.append(message)

            await app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/api/v1/flows/validate",
                    "raw_path": b"/api/v1/flows/validate",
                    "query_string": b"",
                    "headers": headers or [(b"content-type", b"application/json")],
                    "server": ("localhost", 80),
                    "client": ("localhost", 1234),
                },
                receive,
                send,
            )
            return responses

        return asyncio.run(run())

    return request
