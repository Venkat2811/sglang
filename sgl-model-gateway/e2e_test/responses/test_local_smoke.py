"""Small-model local smoke tests for the Responses API.

These tests are intentionally narrow. They exist to keep a single-GPU local
development loop healthy before adding heavier semantic or compatibility suites.
"""

from __future__ import annotations

import asyncio
import json

import openai
import pytest


def _gateway_ws_url(base_url: str) -> str:
    """Convert the router base URL into a websocket endpoint URL."""
    if base_url.startswith("https://"):
        return f"wss://{base_url.removeprefix('https://')}/v1/responses"
    return f"ws://{base_url.removeprefix('http://')}/v1/responses"


def _ws_request(
    model: str,
    *,
    input,
    store: bool,
    previous_response_id: str | None = None,
    generate: bool | None = None,
    temperature: float = 0,
    max_output_tokens: int = 16,
) -> dict:
    request = {
        "type": "response.create",
        "model": model,
        "input": input,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "store": store,
    }
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    if generate is not None:
        request["generate"] = generate
    return request


def _ws_error_code(event: dict) -> str | None:
    error = event.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str):
            return code
    code = event.get("code")
    return code if isinstance(code, str) else None


def _tool_output_chain_turn_input(turn_index: int) -> list[dict]:
    return [
        {
            "type": "function_call_output",
            "call_id": f"call_ws_smoke_{turn_index}",
            "output": json.dumps(
                {
                    "step": turn_index,
                    "status": "ok",
                    "summary": f"tool result {turn_index}",
                }
            ),
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Continue from this tool result and reply with hello.",
                }
            ],
        },
    ]


async def _send_ws_request_and_collect(
    websocket,
    request: dict,
    *,
    fail_on_error: bool = True,
) -> list[dict]:
    await websocket.send(json.dumps(request))

    events: list[dict] = []
    while True:
        payload = await asyncio.wait_for(websocket.recv(), timeout=90)
        event = json.loads(payload)
        events.append(event)

        if event.get("type") == "error":
            if fail_on_error:
                raise AssertionError(f"Unexpected websocket error: {event}")
            break

        if event.get("type") == "response.completed":
            break

    return events


async def _collect_ws_events(ws_url: str, model: str) -> list[dict]:
    """Send one `response.create` request and collect events until terminal state."""
    import websockets

    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        return await _send_ws_request_and_collect(
            websocket,
            _ws_request(
                model,
                input="Reply with the single word: hello",
                store=False,
            ),
        )


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


def _collect_http_event_types(client, model: str) -> list[str]:
    """Collect the logical event sequence from the HTTP streaming Responses path."""
    response = client.responses.create(
        model=model,
        input="Reply with the single word: hello",
        stream=True,
        temperature=0,
        max_output_tokens=16,
        store=False,
    )
    return [event.type for event in response]


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

    def test_http_ws_event_type_parity(self, setup_backend):
        """HTTP SSE and WS should expose the same logical event sequence locally."""
        _, model, client, gateway = setup_backend

        http_event_types = _collect_http_event_types(client, model)
        ws_event_types = [
            event["type"]
            for event in asyncio.run(
                _collect_ws_events(_gateway_ws_url(gateway.base_url), model)
            )
        ]

        assert ws_event_types == http_event_types

    def test_websocket_store_false_continuation_is_connection_local(self, setup_backend):
        """`store=false` should continue on the same socket and fail after reconnect."""
        _, model, _, gateway = setup_backend

        async def run() -> tuple[list[dict], list[dict], list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                first_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="First websocket turn. Reply with hello.",
                        store=False,
                    ),
                )
                response_id = first_events[-1]["response"]["id"]
                second_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Second websocket turn. Reply with hello again.",
                        previous_response_id=response_id,
                        store=False,
                    ),
                )

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                reconnect_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Reconnect websocket turn.",
                        previous_response_id=response_id,
                        store=False,
                    ),
                    fail_on_error=False,
                )

            return first_events, second_events, reconnect_events

        first_events, second_events, reconnect_events = asyncio.run(run())

        assert first_events[-1]["type"] == "response.completed"
        assert second_events[-1]["type"] == "response.completed"
        assert reconnect_events[-1]["type"] == "error"
        assert _ws_error_code(reconnect_events[-1]) == "previous_response_not_found"

    def test_websocket_only_latest_store_false_response_is_cached_per_connection(
        self, setup_backend
    ):
        """WS keeps only the most recent `store=false` response in connection-local cache."""
        _, model, _, gateway = setup_backend

        async def run() -> tuple[list[dict], list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                first_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="First store-false websocket turn.",
                        store=False,
                    ),
                )
                first_response_id = first_events[-1]["response"]["id"]

                second_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Second store-false websocket turn.",
                        previous_response_id=first_response_id,
                        store=False,
                    ),
                )
                second_response_id = second_events[-1]["response"]["id"]

                stale_retry_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Retry the stale first response id.",
                        previous_response_id=first_response_id,
                        store=False,
                    ),
                    fail_on_error=False,
                )

                latest_retry_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Retry the latest response id.",
                        previous_response_id=second_response_id,
                        store=False,
                    ),
                )

            return stale_retry_events, latest_retry_events

        stale_retry_events, latest_retry_events = asyncio.run(run())

        assert stale_retry_events[-1]["type"] == "error"
        assert _ws_error_code(stale_retry_events[-1]) == "previous_response_not_found"
        assert latest_retry_events[-1]["type"] == "response.completed"

    def test_websocket_invalid_previous_response_id_fails(self, setup_backend):
        """Fresh websocket connections should reject unknown previous_response_id values."""
        _, model, _, gateway = setup_backend

        async def run() -> list[dict]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                return await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="This should fail.",
                        previous_response_id="resp_missing_ws",
                        store=False,
                    ),
                    fail_on_error=False,
                )

        events = asyncio.run(run())

        assert events[-1]["type"] == "error"
        assert _ws_error_code(events[-1]) == "previous_response_not_found"

    def test_websocket_store_true_continuation_survives_reconnect_and_is_retrievable(
        self, setup_backend
    ):
        """Stored WS responses should survive reconnect and be retrievable over HTTP."""
        _, model, client, gateway = setup_backend

        async def run() -> tuple[dict, list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                first_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Persist this websocket turn.",
                        store=True,
                    ),
                )
                first_completed = first_events[-1]
                response_id = first_completed["response"]["id"]

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                reconnect_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Continue after reconnect.",
                        previous_response_id=response_id,
                        store=False,
                    ),
                )

            return first_completed, reconnect_events

        first_completed, reconnect_events = asyncio.run(run())
        stored_response_id = first_completed["response"]["id"]

        assert reconnect_events[-1]["type"] == "response.completed"

        retrieved = client.responses.retrieve(response_id=stored_response_id)
        assert retrieved.id == stored_response_id
        assert retrieved.status == "completed"
        assert retrieved.store is True

    def test_websocket_store_false_response_is_not_retrievable_over_http(
        self, setup_backend
    ):
        """WS `store=false` responses should not become retrievable over HTTP APIs."""
        _, model, client, gateway = setup_backend

        async def run() -> str:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Do not persist this websocket response.",
                        store=False,
                    ),
                )
            return events[-1]["response"]["id"]

        response_id = asyncio.run(run())

        with pytest.raises(openai.NotFoundError):
            client.responses.retrieve(response_id=response_id)

    def test_websocket_function_call_output_continuation_survives_reconnect_when_stored(
        self, setup_backend
    ):
        """Stored WS chains should accept incremental `function_call_output` items."""
        _, model, client, gateway = setup_backend

        async def run() -> tuple[str, dict]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                seed_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Seed a tool-output websocket chain.",
                        store=True,
                    ),
                )
                seed_response_id = seed_events[-1]["response"]["id"]

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                follow_up_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input=_tool_output_chain_turn_input(1),
                        previous_response_id=seed_response_id,
                        store=True,
                    ),
                )

            return seed_response_id, follow_up_events[-1]["response"]

        seed_response_id, continued_response = asyncio.run(run())
        continued_response_id = continued_response["id"]

        seed_response = client.responses.retrieve(response_id=seed_response_id)
        assert seed_response.id == seed_response_id
        assert seed_response.store is True

        retrieved = client.responses.retrieve(response_id=continued_response_id)
        assert retrieved.id == continued_response_id
        assert retrieved.status == "completed"
        assert retrieved.store is True
        assert len(retrieved.output) > 0

    def test_websocket_generate_false_returns_chainable_response_id(self, setup_backend):
        """`generate=false` should return a response id that the same socket can chain from."""
        _, model, _, gateway = setup_backend

        async def run() -> tuple[list[dict], list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                warmup_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Warm up this websocket request state.",
                        store=False,
                        generate=False,
                    ),
                )
                warmup_response_id = warmup_events[-1]["response"]["id"]
                follow_up_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Now answer with the single word: warmed",
                        previous_response_id=warmup_response_id,
                        store=False,
                    ),
                )
            return warmup_events, follow_up_events

        warmup_events, follow_up_events = asyncio.run(run())

        assert warmup_events[-1]["type"] == "response.completed"
        assert warmup_events[-1]["response"]["output"] == []
        assert follow_up_events[-1]["type"] == "response.completed"
        assert _response_output_text(follow_up_events[-1]).strip()

    def test_websocket_generate_false_store_false_is_not_retrievable_or_reconnectable(
        self, setup_backend
    ):
        """Warmup responses with `store=false` stay socket-local and non-durable."""
        _, model, client, gateway = setup_backend

        async def run() -> tuple[str, list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                warmup_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Warm up without durable storage.",
                        store=False,
                        generate=False,
                    ),
                )
                warmup_response_id = warmup_events[-1]["response"]["id"]

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                reconnect_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Continue from non-durable warmup.",
                        previous_response_id=warmup_response_id,
                        store=False,
                    ),
                    fail_on_error=False,
                )

            return warmup_response_id, reconnect_events

        warmup_response_id, reconnect_events = asyncio.run(run())

        assert reconnect_events[-1]["type"] == "error"
        assert _ws_error_code(reconnect_events[-1]) == "previous_response_not_found"

        with pytest.raises(openai.NotFoundError):
            client.responses.retrieve(response_id=warmup_response_id)

    def test_websocket_generate_false_store_true_survives_reconnect_and_is_retrievable(
        self, setup_backend
    ):
        """Warmup responses with `store=true` should persist like normal WS responses."""
        _, model, client, gateway = setup_backend

        async def run() -> tuple[str, list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                warmup_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Warm up with durable storage.",
                        store=True,
                        generate=False,
                    ),
                )
                warmup_response_id = warmup_events[-1]["response"]["id"]

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                reconnect_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Continue from stored warmup.",
                        previous_response_id=warmup_response_id,
                        store=False,
                    ),
                )

            return warmup_response_id, reconnect_events

        warmup_response_id, reconnect_events = asyncio.run(run())

        retrieved = client.responses.retrieve(response_id=warmup_response_id)
        assert retrieved.id == warmup_response_id
        assert retrieved.status == "completed"
        assert retrieved.store is True
        assert retrieved.output == []

        assert reconnect_events[-1]["type"] == "response.completed"
        assert _response_output_text(reconnect_events[-1]).strip()

    def test_http_invalid_previous_response_id_is_not_found(self, setup_backend):
        """HTTP Responses should reject unknown previous_response_id values."""
        _, model, client, _ = setup_backend

        with pytest.raises(openai.NotFoundError):
            client.responses.create(
                model=model,
                input="This should fail.",
                previous_response_id="resp_missing_http",
                max_output_tokens=16,
            )
