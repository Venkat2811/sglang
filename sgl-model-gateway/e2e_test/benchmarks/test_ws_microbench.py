"""Lightweight WebSocket microbenchmark for local gateway iteration.

This is intentionally separate from the existing genai-bench HTTP controls.
It measures the WebSocket Responses path directly on a small local model and
writes a JSON summary for repeatable local regression checks.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import operator as op
import os
import statistics
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

CALCULATE_FUNCTION = {
    "type": "function",
    "name": "calculate",
    "description": "Perform a mathematical calculation.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate.",
            }
        },
        "required": ["expression"],
    },
}

CALCULATE_TOOL_REQUEST_INSTRUCTIONS = (
    "You are a calculator assistant. "
    "When the user asks for a calculation, call the calculate tool exactly once "
    "and do not answer until the tool result is provided."
)

CALCULATE_TOOL_RESULT_INSTRUCTIONS = (
    "The calculate tool result has been provided. "
    "Answer briefly with the final numeric result."
)


def _gateway_ws_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return f"wss://{base_url.removeprefix('https://')}/v1/responses"
    return f"ws://{base_url.removeprefix('http://')}/v1/responses"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _benchmark_request_body(model: str) -> dict:
    return {
        "model": model,
        "input": "Reply with the single word: hello",
        "temperature": 0,
        "max_output_tokens": 16,
        "store": False,
    }


def _ws_request(**request_fields) -> dict:
    return {"type": "response.create", **request_fields}


def _chain_turn_input(turn_index: int) -> str:
    return (
        f"Continuation turn {turn_index}. "
        "Reply with the single word: hello."
    )


def _tool_output_chain_turn_input(turn_index: int) -> list[dict]:
    return [
        {
            "type": "function_call_output",
            "call_id": f"call_chain_{turn_index}",
            "output": json.dumps(
                {
                    "step": turn_index,
                    "status": "ok",
                    "summary": f"tool result {turn_index}",
                    "artifacts": [f"chunk_{turn_index}_{index}" for index in range(3)],
                }
            ),
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"Tool step {turn_index} is complete. "
                        "Reply with the single word: hello."
                    ),
                }
            ],
        },
    ]


def _model_generated_tool_prompt(turn_index: int) -> str:
    left = 15 + turn_index
    right = 23 + (turn_index * 2)
    return f"Calculate {left} * {right}. Use the tool."


def _response_function_calls(output) -> list:
    return [item for item in output if item.type == "function_call"]


def _completed_response_function_calls(completed_event: dict) -> list[dict]:
    return [
        item
        for item in completed_event.get("response", {}).get("output", [])
        if item.get("type") == "function_call"
    ]


_SAFE_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}


def _evaluate_expression(expression: str) -> int | float:
    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
            return _SAFE_BINOPS[type(node.op)](
                eval_node(node.left), eval_node(node.right)
            )
        raise ValueError(f"unsupported expression: {expression}")

    parsed = ast.parse(expression, mode="eval")
    return eval_node(parsed)


def _tool_output_from_http_function_call(function_call) -> list[dict]:
    arguments = json.loads(function_call.arguments)
    expression = arguments["expression"]
    result = _evaluate_expression(expression)
    return [
        {
            "type": "function_call_output",
            "call_id": function_call.call_id,
            "output": json.dumps({"result": result}),
        }
    ]


def _tool_output_from_ws_function_call(function_call: dict) -> list[dict]:
    arguments = json.loads(function_call["arguments"])
    expression = arguments["expression"]
    result = _evaluate_expression(expression)
    return [
        {
            "type": "function_call_output",
            "call_id": function_call["call_id"],
            "output": json.dumps({"result": result}),
        }
    ]


def _summarize_samples(samples: list[dict[str, float | int]]) -> dict[str, float | int]:
    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in samples]

    summary = {
        "samples": len(samples),
        "request_to_first_event_ms_p50": _percentile(
            values("request_to_first_event_ms"), 0.50
        ),
        "request_to_first_event_ms_p95": _percentile(
            values("request_to_first_event_ms"), 0.95
        ),
        "request_to_first_content_ms_p50": _percentile(
            values("request_to_first_content_ms"), 0.50
        ),
        "request_to_first_content_ms_p95": _percentile(
            values("request_to_first_content_ms"), 0.95
        ),
        "request_to_completed_ms_p50": _percentile(
            values("request_to_completed_ms"), 0.50
        ),
        "request_to_completed_ms_p95": _percentile(
            values("request_to_completed_ms"), 0.95
        ),
        "output_tokens_per_second_mean": statistics.fmean(
            values("output_tokens_per_second")
        ),
    }

    if "connect_ms" in samples[0]:
        summary["connect_ms_p50"] = _percentile(values("connect_ms"), 0.50)
        summary["connect_ms_p95"] = _percentile(values("connect_ms"), 0.95)

    return summary


async def _run_single_ws_sample(ws_url: str, model: str) -> dict[str, float | int]:
    import websockets

    request = _ws_request(**_benchmark_request_body(model))

    connect_started_at = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        connected_at = time.perf_counter()
        await websocket.send(json.dumps(request))
        request_sent_at = time.perf_counter()

        first_event_ms: float | None = None
        first_content_ms: float | None = None
        completed_ms: float | None = None
        output_tokens = 0

        while True:
            payload = await asyncio.wait_for(websocket.recv(), timeout=90)
            event = json.loads(payload)
            now = time.perf_counter()

            if first_event_ms is None:
                first_event_ms = (now - request_sent_at) * 1000

            if (
                first_content_ms is None
                and event.get("type") == "response.output_text.delta"
                and isinstance(event.get("delta"), str)
                and event["delta"]
            ):
                first_content_ms = (now - request_sent_at) * 1000

            if event.get("type") == "error":
                raise AssertionError(f"Unexpected websocket benchmark error: {event}")

            if event.get("type") == "response.completed":
                completed_ms = (now - request_sent_at) * 1000
                usage = event.get("response", {}).get("usage", {})
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                if first_content_ms is None:
                    first_content_ms = completed_ms
                break

    total_connect_ms = (connected_at - connect_started_at) * 1000
    tokens_per_second = 0.0
    if output_tokens > 0 and completed_ms and completed_ms > 0:
        tokens_per_second = output_tokens / (completed_ms / 1000)

    return {
        "connect_ms": total_connect_ms,
        "request_to_first_event_ms": first_event_ms or 0.0,
        "request_to_first_content_ms": first_content_ms or 0.0,
        "request_to_completed_ms": completed_ms or 0.0,
        "output_tokens": output_tokens,
        "output_tokens_per_second": tokens_per_second,
    }


def _run_single_http_sample(client, model: str) -> dict[str, float | int]:
    request_started_at = time.perf_counter()
    response = client.responses.create(stream=True, **_benchmark_request_body(model))

    first_event_ms: float | None = None
    first_content_ms: float | None = None
    completed_ms: float | None = None
    output_tokens = 0

    for event in response:
        now = time.perf_counter()

        if first_event_ms is None:
            first_event_ms = (now - request_started_at) * 1000

        if (
            first_content_ms is None
            and event.type == "response.output_text.delta"
            and isinstance(getattr(event, "delta", None), str)
            and event.delta
        ):
            first_content_ms = (now - request_started_at) * 1000

        if event.type == "response.completed":
            completed_ms = (now - request_started_at) * 1000
            usage = getattr(getattr(event, "response", None), "usage", None)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            if first_content_ms is None:
                first_content_ms = completed_ms
            break

    tokens_per_second = 0.0
    if output_tokens > 0 and completed_ms and completed_ms > 0:
        tokens_per_second = output_tokens / (completed_ms / 1000)

    return {
        "request_to_first_event_ms": first_event_ms or 0.0,
        "request_to_first_content_ms": first_content_ms or 0.0,
        "request_to_completed_ms": completed_ms or 0.0,
        "output_tokens": output_tokens,
        "output_tokens_per_second": tokens_per_second,
    }


async def _run_concurrency_profile(
    ws_url: str,
    model: str,
    concurrency_levels: list[int],
    samples_per_concurrency: int,
) -> dict:
    results = []
    for concurrency in concurrency_levels:
        samples: list[dict[str, float | int]] = []
        for _ in range(samples_per_concurrency):
            batch = await asyncio.gather(
                *[_run_single_ws_sample(ws_url, model) for _ in range(concurrency)]
            )
            samples.extend(batch)

        summary = _summarize_samples(samples)
        logger.info("WS microbench concurrency=%s summary=%s", concurrency, summary)
        results.append(
            {
                "concurrency": concurrency,
                "samples": samples,
                "summary": summary,
            }
        )

    return {
        "profile": {
            "concurrency_levels": concurrency_levels,
            "samples_per_concurrency": samples_per_concurrency,
        },
        "results": results,
    }


async def _run_ws_sample_batch(
    ws_url: str, model: str, samples: int
) -> list[dict[str, float | int]]:
    return await asyncio.gather(
        *[_run_single_ws_sample(ws_url, model) for _ in range(samples)]
    )


async def _collect_ws_terminal_event(websocket, request: dict) -> tuple[dict, float]:
    await websocket.send(json.dumps(request))
    request_started_at = time.perf_counter()

    while True:
        payload = await asyncio.wait_for(websocket.recv(), timeout=90)
        event = json.loads(payload)
        now = time.perf_counter()

        if event.get("type") == "error":
            raise AssertionError(f"Unexpected websocket chain benchmark error: {event}")

        if event.get("type") == "response.completed":
            return event, (now - request_started_at) * 1000


async def _run_ws_continuation_chain_sample(
    ws_url: str, model: str, turns: int
) -> dict[str, float | int | list[float]]:
    import websockets

    connect_started_at = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        connected_at = time.perf_counter()

        per_turn_ms: list[float] = []
        response, first_turn_ms = await _collect_ws_terminal_event(
            websocket,
            _ws_request(
                model=model,
                input=_chain_turn_input(1),
                temperature=0,
                max_output_tokens=16,
                store=True,
            ),
        )
        per_turn_ms.append(first_turn_ms)
        previous_response_id = response["response"]["id"]

        for turn_index in range(2, turns + 1):
            response, completed_ms = await _collect_ws_terminal_event(
                websocket,
                _ws_request(
                    model=model,
                    input=_chain_turn_input(turn_index),
                    temperature=0,
                    max_output_tokens=16,
                    store=True,
                    previous_response_id=previous_response_id,
                ),
            )
            per_turn_ms.append(completed_ms)
            previous_response_id = response["response"]["id"]

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )
    total_chain_ms = sum(per_turn_ms)

    return {
        "connect_ms": (connected_at - connect_started_at) * 1000,
        "turns": turns,
        "total_chain_ms": total_chain_ms,
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
    }


async def _run_ws_tool_output_chain_sample(
    ws_url: str, model: str, tool_turns: int
) -> dict[str, float | int | list[float]]:
    import websockets

    connect_started_at = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        connected_at = time.perf_counter()

        per_turn_ms: list[float] = []
        response, seed_turn_ms = await _collect_ws_terminal_event(
            websocket,
            _ws_request(
                model=model,
                input="Seed the tool-output continuation chain. Reply with hello.",
                temperature=0,
                max_output_tokens=16,
                store=True,
            ),
        )
        per_turn_ms.append(seed_turn_ms)
        previous_response_id = response["response"]["id"]

        for turn_index in range(1, tool_turns + 1):
            response, completed_ms = await _collect_ws_terminal_event(
                websocket,
                _ws_request(
                    model=model,
                    input=_tool_output_chain_turn_input(turn_index),
                    temperature=0,
                    max_output_tokens=16,
                    store=True,
                    previous_response_id=previous_response_id,
                ),
            )
            per_turn_ms.append(completed_ms)
            previous_response_id = response["response"]["id"]

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )
    total_chain_ms = sum(per_turn_ms)

    return {
        "connect_ms": (connected_at - connect_started_at) * 1000,
        "turns": tool_turns + 1,
        "tool_turns": tool_turns,
        "total_chain_ms": total_chain_ms,
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
    }


async def _run_ws_model_generated_tool_chain_sample(
    ws_url: str, model: str, turns: int
) -> dict[str, float | int | list[float] | list[dict[str, float]]]:
    import websockets

    connect_started_at = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        connected_at = time.perf_counter()

        per_turn_ms: list[float] = []
        per_turn_requests_ms: list[dict[str, float]] = []
        previous_response_id: str | None = None

        for turn_index in range(1, turns + 1):
            first_completed, tool_request_ms = await _collect_ws_terminal_event(
                websocket,
                _ws_request(
                    model=model,
                    input=_model_generated_tool_prompt(turn_index),
                    instructions=CALCULATE_TOOL_REQUEST_INSTRUCTIONS,
                    temperature=0,
                    max_output_tokens=128,
                    store=True,
                    tools=[CALCULATE_FUNCTION],
                    tool_choice="required",
                    previous_response_id=previous_response_id,
                ),
            )

            function_calls = _completed_response_function_calls(first_completed)
            if not function_calls:
                raise AssertionError("expected function call in websocket tool benchmark")

            second_completed, tool_output_ms = await _collect_ws_terminal_event(
                websocket,
                _ws_request(
                    model=model,
                    input=_tool_output_from_ws_function_call(function_calls[0]),
                    instructions=CALCULATE_TOOL_RESULT_INSTRUCTIONS,
                    temperature=0,
                    max_output_tokens=128,
                    store=True,
                    tools=[CALCULATE_FUNCTION],
                    tool_choice="auto",
                    previous_response_id=first_completed["response"]["id"],
                ),
            )

            turn_total_ms = tool_request_ms + tool_output_ms
            per_turn_ms.append(turn_total_ms)
            per_turn_requests_ms.append(
                {
                    "tool_request_ms": tool_request_ms,
                    "tool_output_ms": tool_output_ms,
                    "turn_total_ms": turn_total_ms,
                }
            )
            previous_response_id = second_completed["response"]["id"]

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )

    return {
        "connect_ms": (connected_at - connect_started_at) * 1000,
        "turns": turns,
        "total_chain_ms": sum(per_turn_ms),
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
        "per_turn_requests_ms": per_turn_requests_ms,
    }


def _collect_http_completed_ms(response_stream) -> tuple[object, float]:
    request_started_at = time.perf_counter()
    event_types: list[str] = []

    for event in response_stream:
        event_types.append(event.type)
        if event.type in {"error", "response.failed", "response.incomplete"}:
            raise AssertionError(
                f"HTTP continuation benchmark terminated with {event.type}: {event_types}"
            )
        if event.type == "response.completed":
            return event.response, (time.perf_counter() - request_started_at) * 1000

    raise AssertionError(
        "HTTP continuation benchmark ended without response.completed; "
        f"events={event_types}"
    )


def _run_http_continuation_chain_sample(
    client, model: str, turns: int
) -> dict[str, float | int | list[float]]:
    per_turn_ms: list[float] = []

    response_stream = client.responses.create(
        model=model,
        input=_chain_turn_input(1),
        temperature=0,
        max_output_tokens=16,
        store=True,
        stream=True,
    )
    response, completed_ms = _collect_http_completed_ms(response_stream)
    per_turn_ms.append(completed_ms)
    previous_response_id = response.id

    for turn_index in range(2, turns + 1):
        response_stream = client.responses.create(
            model=model,
            input=_chain_turn_input(turn_index),
            previous_response_id=previous_response_id,
            temperature=0,
            max_output_tokens=16,
            store=True,
            stream=True,
        )
        response, completed_ms = _collect_http_completed_ms(response_stream)
        per_turn_ms.append(completed_ms)
        previous_response_id = response.id

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )
    total_chain_ms = sum(per_turn_ms)

    return {
        "turns": turns,
        "total_chain_ms": total_chain_ms,
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
    }


def _run_http_tool_output_chain_sample(
    client, model: str, tool_turns: int
) -> dict[str, float | int | list[float]]:
    per_turn_ms: list[float] = []

    response_stream = client.responses.create(
        model=model,
        input="Seed the tool-output continuation chain. Reply with hello.",
        temperature=0,
        max_output_tokens=16,
        store=True,
        stream=True,
    )
    response, completed_ms = _collect_http_completed_ms(response_stream)
    per_turn_ms.append(completed_ms)
    previous_response_id = response.id

    for turn_index in range(1, tool_turns + 1):
        response_stream = client.responses.create(
            model=model,
            input=_tool_output_chain_turn_input(turn_index),
            previous_response_id=previous_response_id,
            temperature=0,
            max_output_tokens=16,
            store=True,
            stream=True,
        )
        response, completed_ms = _collect_http_completed_ms(response_stream)
        per_turn_ms.append(completed_ms)
        previous_response_id = response.id

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )
    total_chain_ms = sum(per_turn_ms)

    return {
        "turns": tool_turns + 1,
        "tool_turns": tool_turns,
        "total_chain_ms": total_chain_ms,
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
    }


def _run_http_model_generated_tool_chain_sample(
    client, model: str, turns: int
) -> dict[str, float | int | list[float] | list[dict[str, float]]]:
    per_turn_ms: list[float] = []
    per_turn_requests_ms: list[dict[str, float]] = []
    previous_response_id: str | None = None

    for turn_index in range(1, turns + 1):
        first_stream = client.responses.create(
            model=model,
            input=_model_generated_tool_prompt(turn_index),
            previous_response_id=previous_response_id,
            instructions=CALCULATE_TOOL_REQUEST_INSTRUCTIONS,
            temperature=0,
            max_output_tokens=128,
            store=True,
            stream=True,
            tools=[CALCULATE_FUNCTION],
            tool_choice="required",
        )
        first_response, tool_request_ms = _collect_http_completed_ms(first_stream)

        function_calls = _response_function_calls(first_response.output)
        if not function_calls:
            output_types = [item.type for item in first_response.output]
            raise AssertionError(
                "expected function call in http tool benchmark "
                f"at turn {turn_index}; output_types={output_types}"
            )

        second_stream = client.responses.create(
            model=model,
            input=_tool_output_from_http_function_call(function_calls[0]),
            previous_response_id=first_response.id,
            instructions=CALCULATE_TOOL_RESULT_INSTRUCTIONS,
            temperature=0,
            max_output_tokens=128,
            store=True,
            stream=True,
            tools=[CALCULATE_FUNCTION],
            tool_choice="auto",
        )
        second_response, tool_output_ms = _collect_http_completed_ms(second_stream)

        turn_total_ms = tool_request_ms + tool_output_ms
        per_turn_ms.append(turn_total_ms)
        per_turn_requests_ms.append(
            {
                "tool_request_ms": tool_request_ms,
                "tool_output_ms": tool_output_ms,
                "turn_total_ms": turn_total_ms,
            }
        )
        previous_response_id = second_response.id

    continuation_turns = per_turn_ms[1:]
    continuation_mean_ms = (
        statistics.fmean(continuation_turns) if continuation_turns else 0.0
    )

    return {
        "turns": turns,
        "total_chain_ms": sum(per_turn_ms),
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
        "per_turn_requests_ms": per_turn_requests_ms,
    }


def _write_summary(experiment_folder: str, payload: dict) -> Path:
    out_dir = Path.cwd() / experiment_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _transport_ratios(http_summary: dict, ws_summary: dict) -> dict[str, float]:
    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    return {
        "ws_over_http_first_event_p50": ratio(
            float(ws_summary["request_to_first_event_ms_p50"]),
            float(http_summary["request_to_first_event_ms_p50"]),
        ),
        "ws_over_http_first_content_p50": ratio(
            float(ws_summary["request_to_first_content_ms_p50"]),
            float(http_summary["request_to_first_content_ms_p50"]),
        ),
        "ws_over_http_completed_p50": ratio(
            float(ws_summary["request_to_completed_ms_p50"]),
            float(http_summary["request_to_completed_ms_p50"]),
        ),
        "ws_over_http_output_tps_mean": ratio(
            float(ws_summary["output_tokens_per_second_mean"]),
            float(http_summary["output_tokens_per_second_mean"]),
        ),
    }


def _chain_transport_ratios(http_sample: dict, ws_sample: dict) -> dict[str, float]:
    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    total_ratio = ratio(
        float(ws_sample["total_chain_ms"]), float(http_sample["total_chain_ms"])
    )
    continuation_ratio = ratio(
        float(ws_sample["continuation_only_total_ms"]),
        float(http_sample["continuation_only_total_ms"]),
    )

    return {
        "ws_over_http_total_chain": total_ratio,
        "ws_over_http_continuation_only": continuation_ratio,
        "ws_vs_http_total_chain_delta_pct": (1.0 - total_ratio) * 100.0,
        "ws_vs_http_continuation_only_delta_pct": (1.0 - continuation_ratio)
            * 100.0,
    }


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model("qwen-0.5b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestWsMicrobench:
    """WebSocket benchmark for the Responses route on a single small model."""

    def test_ws_microbench(self, setup_backend):
        _, model, _, gateway = setup_backend

        concurrency_levels = [
            int(value)
            for value in os.environ.get("SGLANG_WS_BENCH_CONCURRENCY", "1,2,4").split(",")
            if value.strip()
        ]
        samples_per_concurrency = int(
            os.environ.get("SGLANG_WS_BENCH_SAMPLES_PER_CONCURRENCY", "2")
        )
        experiment_folder = os.environ.get(
            "SGLANG_WS_BENCH_EXPERIMENT",
            f"benchmark_ws_microbench_{model.replace('/', '_')}",
        )

        payload = asyncio.run(
            _run_concurrency_profile(
                _gateway_ws_url(gateway.base_url),
                model,
                concurrency_levels,
                samples_per_concurrency,
            )
        )
        payload["router_url"] = gateway.base_url
        payload["model"] = model
        payload["experiment_folder"] = experiment_folder

        summary_path = _write_summary(experiment_folder, payload)
        logger.info("WS microbenchmark summary written to %s", summary_path)

        for result in payload["results"]:
            summary = result["summary"]
            assert summary["samples"] > 0
            assert summary["request_to_first_event_ms_p50"] > 0
            assert summary["request_to_first_content_ms_p50"] > 0
            assert summary["request_to_completed_ms_p50"] > 0


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model("qwen-0.5b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestResponsesTransportCompare:
    """Small-model transport comparison for HTTP SSE vs WebSocket Responses."""

    def test_http_vs_ws_transport_compare(self, setup_backend):
        _, model, client, gateway = setup_backend

        samples = int(os.environ.get("SGLANG_HTTP_WS_COMPARE_SAMPLES", "2"))
        experiment_folder = os.environ.get(
            "SGLANG_HTTP_WS_COMPARE_EXPERIMENT",
            f"benchmark_http_ws_compare_{model.replace('/', '_')}",
        )

        http_samples = [_run_single_http_sample(client, model) for _ in range(samples)]
        ws_samples = asyncio.run(
            _run_ws_sample_batch(_gateway_ws_url(gateway.base_url), model, samples)
        )

        http_summary = _summarize_samples(http_samples)
        ws_summary = _summarize_samples(ws_samples)

        payload = {
            "router_url": gateway.base_url,
            "model": model,
            "experiment_folder": experiment_folder,
            "samples_per_transport": samples,
            "http": {
                "transport": "http_sse",
                "samples": http_samples,
                "summary": http_summary,
            },
            "websocket": {
                "transport": "websocket",
                "samples": ws_samples,
                "summary": ws_summary,
            },
            "ratios": _transport_ratios(http_summary, ws_summary),
        }

        summary_path = _write_summary(experiment_folder, payload)
        logger.info("HTTP-vs-WS transport comparison summary written to %s", summary_path)

        assert http_summary["samples"] > 0
        assert ws_summary["samples"] > 0
        assert http_summary["request_to_first_event_ms_p50"] > 0
        assert ws_summary["request_to_first_event_ms_p50"] > 0
        assert http_summary["request_to_completed_ms_p50"] > 0
        assert ws_summary["request_to_completed_ms_p50"] > 0


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model("qwen-0.5b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestResponsesContinuationChainCompare:
    """Long-chain continuation comparison for HTTP vs persistent WS."""

    def test_http_vs_ws_continuation_chain_compare(self, setup_backend):
        _, model, client, gateway = setup_backend

        turns = int(os.environ.get("SGLANG_HTTP_WS_CHAIN_TURNS", "20"))
        samples = int(os.environ.get("SGLANG_HTTP_WS_CHAIN_SAMPLES", "1"))
        experiment_folder = os.environ.get(
            "SGLANG_HTTP_WS_CHAIN_EXPERIMENT",
            f"benchmark_http_ws_chain_compare_{model.replace('/', '_')}_{turns}turns",
        )

        http_samples = [
            _run_http_continuation_chain_sample(client, model, turns)
            for _ in range(samples)
        ]
        ws_samples = [
            asyncio.run(
                _run_ws_continuation_chain_sample(
                    _gateway_ws_url(gateway.base_url), model, turns
                )
            )
            for _ in range(samples)
        ]

        payload = {
            "router_url": gateway.base_url,
            "model": model,
            "turns": turns,
            "samples": samples,
            "experiment_folder": experiment_folder,
            "http": {
                "transport": "http",
                "samples": http_samples,
            },
            "websocket": {
                "transport": "websocket",
                "samples": ws_samples,
            },
            "ratios": _chain_transport_ratios(http_samples[0], ws_samples[0]),
        }

        summary_path = _write_summary(experiment_folder, payload)
        logger.info("HTTP-vs-WS chain comparison summary written to %s", summary_path)

        assert http_samples[0]["turns"] == turns
        assert ws_samples[0]["turns"] == turns
        assert float(http_samples[0]["total_chain_ms"]) > 0
        assert float(ws_samples[0]["total_chain_ms"]) > 0


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model("qwen-0.5b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestResponsesToolOutputChainCompare:
    """Tool-output-heavy continuation comparison for HTTP vs persistent WS."""

    def test_http_vs_ws_tool_output_chain_compare(self, setup_backend):
        _, model, client, gateway = setup_backend

        tool_turns = int(os.environ.get("SGLANG_HTTP_WS_TOOL_CHAIN_TURNS", "20"))
        samples = int(os.environ.get("SGLANG_HTTP_WS_TOOL_CHAIN_SAMPLES", "1"))
        experiment_folder = os.environ.get(
            "SGLANG_HTTP_WS_TOOL_CHAIN_EXPERIMENT",
            (
                "benchmark_http_ws_tool_chain_compare_"
                f"{model.replace('/', '_')}_{tool_turns}toolturns"
            ),
        )

        http_samples = [
            _run_http_tool_output_chain_sample(client, model, tool_turns)
            for _ in range(samples)
        ]
        ws_samples = [
            asyncio.run(
                _run_ws_tool_output_chain_sample(
                    _gateway_ws_url(gateway.base_url), model, tool_turns
                )
            )
            for _ in range(samples)
        ]

        payload = {
            "router_url": gateway.base_url,
            "model": model,
            "tool_turns": tool_turns,
            "samples": samples,
            "experiment_folder": experiment_folder,
            "http": {
                "transport": "http",
                "samples": http_samples,
            },
            "websocket": {
                "transport": "websocket",
                "samples": ws_samples,
            },
            "ratios": _chain_transport_ratios(http_samples[0], ws_samples[0]),
        }

        summary_path = _write_summary(experiment_folder, payload)
        logger.info(
            "HTTP-vs-WS tool-output chain comparison summary written to %s",
            summary_path,
        )

        assert http_samples[0]["tool_turns"] == tool_turns
        assert ws_samples[0]["tool_turns"] == tool_turns
        assert float(http_samples[0]["total_chain_ms"]) > 0
        assert float(ws_samples[0]["total_chain_ms"]) > 0


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model("qwen-3b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestResponsesModelGeneratedToolChainCompare:
    """Model-generated tool-call comparison for HTTP vs persistent WS."""

    def test_http_vs_ws_model_generated_tool_chain_compare(self, setup_backend):
        _, model, client, gateway = setup_backend

        turn_profiles = [
            int(value)
            for value in os.environ.get(
                "SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_TURNS", "5"
            ).split(",")
            if value.strip()
        ]
        samples = int(os.environ.get("SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_SAMPLES", "1"))
        experiment_folder = os.environ.get(
            "SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_EXPERIMENT",
            f"benchmark_http_ws_model_tool_chain_compare_{model.replace('/', '_')}",
        )

        profile_results = []
        for turns in turn_profiles:
            http_samples = [
                _run_http_model_generated_tool_chain_sample(client, model, turns)
                for _ in range(samples)
            ]
            ws_samples = [
                asyncio.run(
                    _run_ws_model_generated_tool_chain_sample(
                        _gateway_ws_url(gateway.base_url), model, turns
                    )
                )
                for _ in range(samples)
            ]

            profile_results.append(
                {
                    "turns": turns,
                    "http": {
                        "transport": "http",
                        "samples": http_samples,
                    },
                    "websocket": {
                        "transport": "websocket",
                        "samples": ws_samples,
                    },
                    "ratios": _chain_transport_ratios(http_samples[0], ws_samples[0]),
                }
            )

        payload = {
            "router_url": gateway.base_url,
            "model": model,
            "turn_profiles": turn_profiles,
            "samples": samples,
            "experiment_folder": experiment_folder,
            "results": profile_results,
        }

        summary_path = _write_summary(experiment_folder, payload)
        logger.info(
            "HTTP-vs-WS model-generated tool-chain summary written to %s",
            summary_path,
        )

        assert profile_results
        for result in profile_results:
            assert result["turns"] > 0
            assert float(result["http"]["samples"][0]["total_chain_ms"]) > 0
            assert float(result["websocket"]["samples"][0]["total_chain_ms"]) > 0
