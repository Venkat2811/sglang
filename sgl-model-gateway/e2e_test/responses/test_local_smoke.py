"""Small-model local smoke tests for the Responses API.

These tests are intentionally narrow. They exist to keep a single-GPU local
development loop healthy before adding heavier semantic or compatibility suites.
"""

from __future__ import annotations

import pytest


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
