"""Small-model local smoke tests for the Responses API.

These tests are intentionally narrow. They exist to keep a single-GPU local
development loop healthy before adding heavier semantic or compatibility suites.
"""

from __future__ import annotations

import asyncio
import json

import pytest


def _gateway_ws_url(base_url: str) -> str:
    """Convert the router base URL into a websocket endpoint URL."""
    if base_url.startswith("https://"):
        return f"wss://{base_url.removeprefix('https://')}/v1/responses"
    return f"ws://{base_url.removeprefix('http://')}/v1/responses"


async def _collect_ws_events(ws_url: str, model: str) -> list[dict]:
    """Send one `response.create` request and collect events until terminal state."""
    import websockets

    request = {
        "type": "response.create",
        "response": {
            "model": model,
            "input": "Reply with the single word: hello",
            "temperature": 0,
            "max_output_tokens": 16,
            "store": False,
        },
    }

    events: list[dict] = []
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        await websocket.send(json.dumps(request))

        while True:
            payload = await asyncio.wait_for(websocket.recv(), timeout=90)
            event = json.loads(payload)
            events.append(event)

            if event.get("type") == "error":
                raise AssertionError(f"Unexpected websocket error: {event}")

            if event.get("type") == "response.completed":
                break

    return events


def _response_output_text(completed_event: dict) -> str:
    """Best-effort extraction of assistant text from a completed response event."""
    for item in completed_event.get("response", {}).get("output", []):
        if item.get("type") != "message":
            continue
        for content_part in item.get("content", []):
            text = content_part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


@pytest.mark.e2e
@pytest.mark.model("qwen-0.5b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestResponsesLocalSmoke:
    """Minimal local Responses checks on a small cached model."""

    def test_basic_response_creation(self, setup_backend):
        """Basic non-streaming response creation should succeed locally."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model,
            input="Reply with the single word: hello",
            temperature=0,
            max_output_tokens=16,
        )

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert resp.usage is not None
        assert len(resp.output_text) > 0

    def test_streaming_response(self, setup_backend):
        """Streaming should emit a normal response event flow."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model,
            input="Count from 1 to 3.",
            stream=True,
            temperature=0,
            max_output_tokens=32,
        )

        events = list(resp)
        assert len(events) > 0

        created = [event for event in events if event.type == "response.created"]
        completed = [event for event in events if event.type == "response.completed"]

        assert created, "Expected at least one response.created event"
        assert completed, "Expected exactly one terminal completed event"
        assert len(completed) == 1

    def test_websocket_response_create(self, setup_backend):
        """WebSocket Responses should complete end-to-end on the local smoke model."""
        _, model, _, gateway = setup_backend

        events = asyncio.run(_collect_ws_events(_gateway_ws_url(gateway.base_url), model))

        event_types = [event["type"] for event in events]
        completed = events[-1]

        assert "response.created" in event_types
        assert completed["type"] == "response.completed"
        assert completed["response"]["status"] == "completed"
        assert len(completed["response"]["output"]) > 0
        assert _response_output_text(completed).strip()
