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
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from benchmarks.bfcl_subset import BfclScenario
from benchmarks.bfcl_subset import BfclWorkspace
from benchmarks.bfcl_subset import bfcl_subset_tools
from benchmarks.bfcl_subset import load_bfcl_subset_scenarios
from benchmarks.bfcl_subset import materialize_bfcl_scenario

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

DEFAULT_MODEL_TOOL_REQUEST_MAX_OUTPUT_TOKENS = 128
DEFAULT_MODEL_TOOL_RESULT_MAX_OUTPUT_TOKENS = 128
BFCL_TOOL_REQUEST_INSTRUCTIONS = (
    "You are handling a BFCL-derived multi-turn filesystem task. "
    "Paths are always relative to the scenario root. "
    "Use the provided filesystem tools only. "
    "For each turn, call the requested tool once and do not answer normally "
    "until the tool result is provided."
)
BFCL_TOOL_RESULT_INSTRUCTIONS = (
    "The filesystem tool result has been provided. "
    "Reply briefly with the outcome and do not call any more tools."
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


def _model_tool_request_instructions() -> str:
    return os.environ.get(
        "SGLANG_HTTP_WS_MODEL_TOOL_REQUEST_INSTRUCTIONS",
        CALCULATE_TOOL_REQUEST_INSTRUCTIONS,
    )


def _model_tool_result_instructions() -> str:
    return os.environ.get(
        "SGLANG_HTTP_WS_MODEL_TOOL_RESULT_INSTRUCTIONS",
        CALCULATE_TOOL_RESULT_INSTRUCTIONS,
    )


def _model_tool_request_max_output_tokens() -> int:
    return int(
        os.environ.get(
            "SGLANG_HTTP_WS_MODEL_TOOL_REQUEST_MAX_OUTPUT_TOKENS",
            str(DEFAULT_MODEL_TOOL_REQUEST_MAX_OUTPUT_TOKENS),
        )
    )


def _model_tool_result_max_output_tokens() -> int:
    return int(
        os.environ.get(
            "SGLANG_HTTP_WS_MODEL_TOOL_RESULT_MAX_OUTPUT_TOKENS",
            str(DEFAULT_MODEL_TOOL_RESULT_MAX_OUTPUT_TOKENS),
        )
    )


def _bfcl_tool_request_instructions() -> str:
    return os.environ.get(
        "SGLANG_HTTP_WS_BFCL_TOOL_REQUEST_INSTRUCTIONS",
        BFCL_TOOL_REQUEST_INSTRUCTIONS,
    )


def _bfcl_tool_result_instructions() -> str:
    return os.environ.get(
        "SGLANG_HTTP_WS_BFCL_TOOL_RESULT_INSTRUCTIONS",
        BFCL_TOOL_RESULT_INSTRUCTIONS,
    )


def _selected_bfcl_subset_scenarios() -> list[BfclScenario]:
    scenarios = load_bfcl_subset_scenarios()
    selected_ids = [
        value.strip()
        for value in os.environ.get("SGLANG_HTTP_WS_BFCL_SUBSET_SCENARIOS", "").split(",")
        if value.strip()
    ]
    if not selected_ids:
        return scenarios

    requested = set(selected_ids)
    selected = [scenario for scenario in scenarios if scenario.id in requested]
    if not selected:
        raise AssertionError(
            f"BFCL subset selection matched no scenarios: requested={selected_ids}"
        )
    return selected


def _bfcl_tool_choice(expected_tool: str) -> dict:
    return {"type": "function", "function": {"name": expected_tool}}


def _bfcl_request_instructions_for_scenario(scenario: BfclScenario) -> str:
    return (
        f"{_bfcl_tool_request_instructions()} "
        f"Scenario {scenario.id}: {scenario.description}"
    )


def _bfcl_tool_output_from_http_function_call(
    workspace: BfclWorkspace, function_call
) -> tuple[list[dict], dict]:
    result = workspace.execute(function_call.name, json.loads(function_call.arguments))
    return (
        [
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": json.dumps(result),
            }
        ],
        result,
    )


def _bfcl_tool_output_from_ws_function_call(
    workspace: BfclWorkspace, function_call: dict
) -> tuple[list[dict], dict]:
    result = workspace.execute(
        function_call["name"],
        json.loads(function_call["arguments"]),
    )
    return (
        [
            {
                "type": "function_call_output",
                "call_id": function_call["call_id"],
                "output": json.dumps(result),
            }
        ],
        result,
    )


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


def _tool_output_from_http_function_call_dict(function_call: dict) -> list[dict]:
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


def _http_response_output_types(response_body: dict) -> list[str]:
    return [
        item_type
        for item in response_body.get("output", [])
        if isinstance(item, dict)
        for item_type in [item.get("type")]
        if isinstance(item_type, str)
    ]


def _http_response_function_calls(response_body: dict) -> list[dict]:
    return [
        item
        for item in response_body.get("output", [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def _http_response_output_text(response_body: dict) -> str:
    for item in response_body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content_part in item.get("content", []):
            if isinstance(content_part, dict):
                text = content_part.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    return ""


def _http_post_json(
    base_url: str, path: str, payload: dict, timeout_secs: float
) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        body = exc.read().decode("utf-8", errors="replace")

    return status_code, json.loads(body)


def _http_stream_events(base_url: str, path: str, payload: dict, timeout_secs: float):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        response = urllib.request.urlopen(request, timeout=timeout_secs)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(
            f"HTTP SSE benchmark request failed status={exc.code} body={body}"
        ) from exc

    with response:
        event_lines: list[str] = []

        def parse_event(lines: list[str]) -> dict | None:
            data_lines = [
                line.removeprefix("data:").lstrip()
                for line in lines
                if line.startswith("data:")
            ]
            if not data_lines:
                return None
            payload_text = "\n".join(data_lines)
            if payload_text == "[DONE]":
                return None
            return json.loads(payload_text)

        while True:
            raw_line = response.readline()
            if not raw_line:
                if event_lines:
                    event = parse_event(event_lines)
                    if event is not None:
                        yield event
                break

            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if not event_lines:
                    continue
                event = parse_event(event_lines)
                event_lines = []
                if event is not None:
                    yield event
                continue

            if line.startswith(":"):
                continue

            event_lines.append(line)


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


def _summarize_chain_samples(
    samples: list[dict[str, float | int | list[float] | list[dict[str, float]]]],
) -> dict[str, float | int]:
    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in samples]

    summary: dict[str, float | int] = {"samples": len(samples)}

    for key in (
        "total_chain_ms",
        "first_turn_completed_ms",
        "continuation_only_total_ms",
    ):
        summary[f"{key}_mean"] = statistics.fmean(values(key))
        summary[f"{key}_p50"] = _percentile(values(key), 0.50)
        summary[f"{key}_p95"] = _percentile(values(key), 0.95)

    if "connect_ms" in samples[0]:
        summary["connect_ms_mean"] = statistics.fmean(values("connect_ms"))
        summary["connect_ms_p50"] = _percentile(values("connect_ms"), 0.50)
        summary["connect_ms_p95"] = _percentile(values("connect_ms"), 0.95)

    return summary


def _summarize_bfcl_suite_samples(samples: list[dict]) -> dict[str, float | int]:
    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in samples]

    summary: dict[str, float | int] = {
        "samples": len(samples),
        "scenario_count": int(samples[0]["scenario_count"]),
        "turns": int(samples[0]["total_turns"]),
        "all_expected_tools_matched": int(
            all(bool(sample["all_expected_tools_matched"]) for sample in samples)
        ),
    }

    for key in (
        "total_suite_ms",
        "total_tool_request_ms",
        "total_tool_output_ms",
    ):
        summary[f"{key}_mean"] = statistics.fmean(values(key))
        summary[f"{key}_p50"] = _percentile(values(key), 0.50)
        summary[f"{key}_p95"] = _percentile(values(key), 0.95)

    if "connect_ms_total" in samples[0]:
        summary["connect_ms_total_mean"] = statistics.fmean(values("connect_ms_total"))
        summary["connect_ms_total_p50"] = _percentile(values("connect_ms_total"), 0.50)
        summary["connect_ms_total_p95"] = _percentile(values("connect_ms_total"), 0.95)

    return summary


def _summarize_bfcl_free_choice_samples(samples: list[dict]) -> dict[str, float | int]:
    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in samples]

    def ratio(sample: dict, numerator_key: str, denominator_key: str) -> float:
        denominator = float(sample[denominator_key])
        if denominator <= 0:
            return 0.0
        return float(sample[numerator_key]) / denominator

    summary: dict[str, float | int] = {
        "samples": len(samples),
        "scenario_count": int(samples[0]["scenario_count"]),
        "turns_requested": int(samples[0]["total_turns_requested"]),
    }

    for key in (
        "matched_turns",
        "total_turns_executed",
        "missing_tool_turns",
        "mismatched_tool_turns",
        "completed_scenarios",
        "terminated_scenarios",
    ):
        summary[f"{key}_mean"] = statistics.fmean(values(key))

    for key in (
        "total_suite_ms",
        "total_tool_request_ms",
        "total_tool_output_ms",
    ):
        summary[f"{key}_mean"] = statistics.fmean(values(key))
        summary[f"{key}_p50"] = _percentile(values(key), 0.50)
        summary[f"{key}_p95"] = _percentile(values(key), 0.95)

    summary["matched_turn_rate_mean"] = statistics.fmean(
        ratio(sample, "matched_turns", "total_turns_requested") for sample in samples
    )
    summary["turn_execution_rate_mean"] = statistics.fmean(
        ratio(sample, "total_turns_executed", "total_turns_requested")
        for sample in samples
    )
    summary["completed_scenario_rate_mean"] = statistics.fmean(
        ratio(sample, "completed_scenarios", "scenario_count") for sample in samples
    )

    if "connect_ms_total" in samples[0]:
        summary["connect_ms_total_mean"] = statistics.fmean(values("connect_ms_total"))
        summary["connect_ms_total_p50"] = _percentile(values("connect_ms_total"), 0.50)
        summary["connect_ms_total_p95"] = _percentile(values("connect_ms_total"), 0.95)

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


def _run_single_http_sample(
    base_url: str, model: str, timeout_secs: float = 90
) -> dict[str, float | int]:
    request_started_at = time.perf_counter()

    first_event_ms: float | None = None
    first_content_ms: float | None = None
    completed_ms: float | None = None
    output_tokens = 0

    for event in _http_stream_events(
        base_url,
        "/v1/responses",
        {
            **_benchmark_request_body(model),
            "stream": True,
        },
        timeout_secs,
    ):
        now = time.perf_counter()
        event_type = event.get("type")

        if first_event_ms is None:
            first_event_ms = (now - request_started_at) * 1000

        if (
            first_content_ms is None
            and event_type == "response.output_text.delta"
            and isinstance(event.get("delta"), str)
            and event["delta"]
        ):
            first_content_ms = (now - request_started_at) * 1000

        if event_type in {"error", "response.failed", "response.incomplete"}:
            raise AssertionError(
                f"HTTP transport benchmark terminated with {event_type}: {event}"
            )

        if event_type == "response.completed":
            completed_ms = (now - request_started_at) * 1000
            usage = event.get("response", {}).get("usage", {})
            output_tokens = int(usage.get("output_tokens", 0) or 0)
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
    return await _collect_ws_terminal_event_with_timeout(websocket, request, timeout_secs=90)


async def _collect_ws_terminal_event_with_timeout(
    websocket, request: dict, *, timeout_secs: float
) -> tuple[dict, float]:
    await websocket.send(json.dumps(request))
    request_started_at = time.perf_counter()

    while True:
        payload = await asyncio.wait_for(websocket.recv(), timeout=timeout_secs)
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

    request_instructions = _model_tool_request_instructions()
    result_instructions = _model_tool_result_instructions()
    request_max_output_tokens = _model_tool_request_max_output_tokens()
    result_max_output_tokens = _model_tool_result_max_output_tokens()
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
                    instructions=request_instructions,
                    temperature=0,
                    max_output_tokens=request_max_output_tokens,
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
                    instructions=result_instructions,
                    temperature=0,
                    max_output_tokens=result_max_output_tokens,
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
        "request_instructions": request_instructions,
        "result_instructions": result_instructions,
        "request_max_output_tokens": request_max_output_tokens,
        "result_max_output_tokens": result_max_output_tokens,
        "total_chain_ms": sum(per_turn_ms),
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
        "per_turn_requests_ms": per_turn_requests_ms,
    }


def _collect_http_completed_ms(
    base_url: str, payload: dict, timeout_secs: float = 90
) -> tuple[dict, float]:
    request_started_at = time.perf_counter()
    event_types: list[str] = []

    for event in _http_stream_events(base_url, "/v1/responses", payload, timeout_secs):
        event_type = str(event.get("type"))
        event_types.append(event_type)
        if event_type in {"error", "response.failed", "response.incomplete"}:
            raise AssertionError(
                f"HTTP continuation benchmark terminated with {event_type}: {event_types}"
            )
        if event_type == "response.completed":
            return event["response"], (time.perf_counter() - request_started_at) * 1000

    raise AssertionError(
        "HTTP continuation benchmark ended without response.completed; "
        f"events={event_types}"
    )


def _run_http_continuation_chain_sample(
    base_url: str, model: str, turns: int, timeout_secs: float = 90
) -> dict[str, float | int | list[float]]:
    per_turn_ms: list[float] = []

    response, completed_ms = _collect_http_completed_ms(
        base_url,
        {
            "model": model,
            "input": _chain_turn_input(1),
            "temperature": 0,
            "max_output_tokens": 16,
            "store": True,
            "stream": True,
        },
        timeout_secs,
    )
    per_turn_ms.append(completed_ms)
    previous_response_id = response["id"]

    for turn_index in range(2, turns + 1):
        response, completed_ms = _collect_http_completed_ms(
            base_url,
            {
                "model": model,
                "input": _chain_turn_input(turn_index),
                "previous_response_id": previous_response_id,
                "temperature": 0,
                "max_output_tokens": 16,
                "store": True,
                "stream": True,
            },
            timeout_secs,
        )
        per_turn_ms.append(completed_ms)
        previous_response_id = response["id"]

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
    base_url: str, model: str, tool_turns: int, timeout_secs: float = 90
) -> dict[str, float | int | list[float]]:
    per_turn_ms: list[float] = []

    response, completed_ms = _collect_http_completed_ms(
        base_url,
        {
            "model": model,
            "input": "Seed the tool-output continuation chain. Reply with hello.",
            "temperature": 0,
            "max_output_tokens": 16,
            "store": True,
            "stream": True,
        },
        timeout_secs,
    )
    per_turn_ms.append(completed_ms)
    previous_response_id = response["id"]

    for turn_index in range(1, tool_turns + 1):
        response, completed_ms = _collect_http_completed_ms(
            base_url,
            {
                "model": model,
                "input": _tool_output_chain_turn_input(turn_index),
                "previous_response_id": previous_response_id,
                "temperature": 0,
                "max_output_tokens": 16,
                "store": True,
                "stream": True,
            },
            timeout_secs,
        )
        per_turn_ms.append(completed_ms)
        previous_response_id = response["id"]

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
    request_instructions = _model_tool_request_instructions()
    result_instructions = _model_tool_result_instructions()
    request_max_output_tokens = _model_tool_request_max_output_tokens()
    result_max_output_tokens = _model_tool_result_max_output_tokens()
    per_turn_ms: list[float] = []
    per_turn_requests_ms: list[dict[str, float]] = []
    previous_response_id: str | None = None

    for turn_index in range(1, turns + 1):
        first_stream = client.responses.create(
            model=model,
            input=_model_generated_tool_prompt(turn_index),
            previous_response_id=previous_response_id,
            instructions=request_instructions,
            temperature=0,
            max_output_tokens=request_max_output_tokens,
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
            instructions=result_instructions,
            temperature=0,
            max_output_tokens=result_max_output_tokens,
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
        "request_instructions": request_instructions,
        "result_instructions": result_instructions,
        "request_max_output_tokens": request_max_output_tokens,
        "result_max_output_tokens": result_max_output_tokens,
        "total_chain_ms": sum(per_turn_ms),
        "first_turn_completed_ms": per_turn_ms[0],
        "continuation_turn_completed_ms_mean": continuation_mean_ms,
        "continuation_turn_completed_ms_p50": _percentile(continuation_turns, 0.50),
        "continuation_turn_completed_ms_p95": _percentile(continuation_turns, 0.95),
        "continuation_only_total_ms": sum(continuation_turns),
        "per_turn_completed_ms": per_turn_ms,
        "per_turn_requests_ms": per_turn_requests_ms,
    }


async def _run_ws_model_generated_tool_chain_diagnostic(
    ws_url: str, model: str, turns: int, timeout_secs: float
) -> dict:
    import websockets

    request_instructions = _model_tool_request_instructions()
    result_instructions = _model_tool_result_instructions()
    request_max_output_tokens = _model_tool_request_max_output_tokens()
    result_max_output_tokens = _model_tool_result_max_output_tokens()
    result = {
        "transport": "websocket",
        "mode": "persistent_ws_diagnostic",
        "turns_requested": turns,
        "request_instructions": request_instructions,
        "result_instructions": result_instructions,
        "request_max_output_tokens": request_max_output_tokens,
        "result_max_output_tokens": result_max_output_tokens,
        "success": True,
        "failed_at_turn": None,
        "failed_stage": None,
        "failure_reason": None,
        "failure_detail": None,
        "records": [],
    }

    connect_started_at = time.perf_counter()
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=5) as websocket:
        result["connect_ms"] = (time.perf_counter() - connect_started_at) * 1000
        previous_response_id: str | None = None

        for turn_index in range(1, turns + 1):
            first_request = _ws_request(
                model=model,
                input=_model_generated_tool_prompt(turn_index),
                instructions=request_instructions,
                temperature=0,
                max_output_tokens=request_max_output_tokens,
                store=True,
                tools=[CALCULATE_FUNCTION],
                tool_choice="required",
                previous_response_id=previous_response_id,
            )

            try:
                first_completed, tool_request_ms = await _collect_ws_terminal_event_with_timeout(
                    websocket, first_request, timeout_secs=timeout_secs
                )
            except Exception as exc:
                result.update(
                    {
                        "success": False,
                        "failed_at_turn": turn_index,
                        "failed_stage": "tool_request",
                        "failure_reason": "exception",
                        "failure_detail": str(exc),
                    }
                )
                return result

            first_function_calls = _completed_response_function_calls(first_completed)
            first_record = {
                "turn": turn_index,
                "stage": "tool_request",
                "elapsed_ms": tool_request_ms,
                "response_id": first_completed.get("response", {}).get("id"),
                "response_status": first_completed.get("response", {}).get("status"),
                "output_types": [
                    item.get("type")
                    for item in first_completed.get("response", {}).get("output", [])
                    if isinstance(item, dict)
                ],
                "function_call_count": len(first_function_calls),
            }
            result["records"].append(first_record)

            if not first_function_calls:
                result.update(
                    {
                        "success": False,
                        "failed_at_turn": turn_index,
                        "failed_stage": "tool_request",
                        "failure_reason": "missing_function_call",
                        "failure_detail": first_record,
                    }
                )
                return result

            second_request = _ws_request(
                model=model,
                input=_tool_output_from_ws_function_call(first_function_calls[0]),
                instructions=result_instructions,
                temperature=0,
                max_output_tokens=result_max_output_tokens,
                store=True,
                tools=[CALCULATE_FUNCTION],
                tool_choice="auto",
                previous_response_id=first_completed["response"]["id"],
            )

            try:
                second_completed, tool_output_ms = await _collect_ws_terminal_event_with_timeout(
                    websocket, second_request, timeout_secs=timeout_secs
                )
            except Exception as exc:
                result.update(
                    {
                        "success": False,
                        "failed_at_turn": turn_index,
                        "failed_stage": "tool_output",
                        "failure_reason": "exception",
                        "failure_detail": str(exc),
                    }
                )
                return result

            second_record = {
                "turn": turn_index,
                "stage": "tool_output",
                "elapsed_ms": tool_output_ms,
                "response_id": second_completed.get("response", {}).get("id"),
                "response_status": second_completed.get("response", {}).get("status"),
                "output_types": [
                    item.get("type")
                    for item in second_completed.get("response", {}).get("output", [])
                    if isinstance(item, dict)
                ],
                "output_text_present": bool(
                    any(
                        isinstance(item, dict)
                        and item.get("type") == "message"
                        for item in second_completed.get("response", {}).get("output", [])
                    )
                ),
            }
            result["records"].append(second_record)
            previous_response_id = second_completed["response"]["id"]

    return result


def _run_http_model_generated_tool_chain_diagnostic(
    base_url: str, model: str, turns: int, timeout_secs: float
) -> dict:
    request_instructions = _model_tool_request_instructions()
    result_instructions = _model_tool_result_instructions()
    request_max_output_tokens = _model_tool_request_max_output_tokens()
    result_max_output_tokens = _model_tool_result_max_output_tokens()
    result = {
        "transport": "http_json",
        "mode": "non_streaming_http_diagnostic",
        "turns_requested": turns,
        "request_instructions": request_instructions,
        "result_instructions": result_instructions,
        "request_max_output_tokens": request_max_output_tokens,
        "result_max_output_tokens": result_max_output_tokens,
        "success": True,
        "failed_at_turn": None,
        "failed_stage": None,
        "failure_reason": None,
        "failure_detail": None,
        "records": [],
    }

    previous_response_id: str | None = None

    for turn_index in range(1, turns + 1):
        first_payload = {
            "model": model,
            "input": _model_generated_tool_prompt(turn_index),
            "previous_response_id": previous_response_id,
            "instructions": request_instructions,
            "temperature": 0,
            "max_output_tokens": request_max_output_tokens,
            "store": True,
            "tools": [CALCULATE_FUNCTION],
            "tool_choice": "required",
        }

        try:
            started_at = time.perf_counter()
            first_status_code, first_body = _http_post_json(
                base_url, "/v1/responses", first_payload, timeout_secs
            )
            tool_request_ms = (time.perf_counter() - started_at) * 1000
        except Exception as exc:
            result.update(
                {
                    "success": False,
                    "failed_at_turn": turn_index,
                    "failed_stage": "tool_request",
                    "failure_reason": "exception",
                    "failure_detail": str(exc),
                }
            )
            return result

        first_function_calls = _http_response_function_calls(first_body)
        first_record = {
            "turn": turn_index,
            "stage": "tool_request",
            "elapsed_ms": tool_request_ms,
            "http_status": first_status_code,
            "response_id": first_body.get("id"),
            "response_status": first_body.get("status"),
            "output_types": _http_response_output_types(first_body),
            "function_call_count": len(first_function_calls),
            "output_text": _http_response_output_text(first_body),
        }
        result["records"].append(first_record)

        if first_status_code != 200:
            result.update(
                {
                    "success": False,
                    "failed_at_turn": turn_index,
                    "failed_stage": "tool_request",
                    "failure_reason": "http_error",
                    "failure_detail": first_record,
                }
            )
            return result

        if not first_function_calls:
            result.update(
                {
                    "success": False,
                    "failed_at_turn": turn_index,
                    "failed_stage": "tool_request",
                    "failure_reason": "missing_function_call",
                    "failure_detail": first_record,
                }
            )
            return result

        second_payload = {
            "model": model,
            "input": _tool_output_from_http_function_call_dict(first_function_calls[0]),
            "previous_response_id": first_body["id"],
            "instructions": result_instructions,
            "temperature": 0,
            "max_output_tokens": result_max_output_tokens,
            "store": True,
            "tools": [CALCULATE_FUNCTION],
            "tool_choice": "auto",
        }

        try:
            started_at = time.perf_counter()
            second_status_code, second_body = _http_post_json(
                base_url, "/v1/responses", second_payload, timeout_secs
            )
            tool_output_ms = (time.perf_counter() - started_at) * 1000
        except Exception as exc:
            result.update(
                {
                    "success": False,
                    "failed_at_turn": turn_index,
                    "failed_stage": "tool_output",
                    "failure_reason": "exception",
                    "failure_detail": str(exc),
                }
            )
            return result

        second_record = {
            "turn": turn_index,
            "stage": "tool_output",
            "elapsed_ms": tool_output_ms,
            "http_status": second_status_code,
            "response_id": second_body.get("id"),
            "response_status": second_body.get("status"),
            "output_types": _http_response_output_types(second_body),
            "output_text": _http_response_output_text(second_body),
        }
        result["records"].append(second_record)

        if second_status_code != 200:
            result.update(
                {
                    "success": False,
                    "failed_at_turn": turn_index,
                    "failed_stage": "tool_output",
                    "failure_reason": "http_error",
                    "failure_detail": second_record,
                }
            )
            return result

        previous_response_id = second_body["id"]

    return result


def _write_summary(experiment_folder: str, payload: dict) -> Path:
    out_dir = Path.cwd() / experiment_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _worker_transport_for_backend(backend_name: str) -> str:
    if backend_name == "http":
        return "http"
    if backend_name == "grpc":
        return "grpc"
    if backend_name == "pd":
        return "http"
    raise ValueError(f"Unsupported benchmark backend: {backend_name}")


def _router_topology_for_backend(backend_name: str) -> str:
    if backend_name == "http":
        return "regular_http_worker"
    if backend_name == "grpc":
        return "regular_grpc_worker"
    if backend_name == "pd":
        return "pd_http_workers"
    raise ValueError(f"Unsupported benchmark backend: {backend_name}")


def _topology_overlay_for_backend(backend_name: str) -> str:
    if backend_name == "pd":
        return "pd"
    return "none"


def _benchmark_context(
    *,
    benchmark_family: str,
    run_class: str,
    backend_name: str,
    model: str,
    store_mode: str,
    workload_kind: str,
) -> dict[str, str]:
    return {
        "benchmark_family": benchmark_family,
        "run_class": run_class,
        "worker_transport": _worker_transport_for_backend(backend_name),
        "router_topology": _router_topology_for_backend(backend_name),
        "model_id": model,
        "topology_overlay": _topology_overlay_for_backend(backend_name),
        "store_mode": store_mode,
        "workload_kind": workload_kind,
    }


def _benchmark_contract(*, client_transport: str, **context: str) -> dict[str, str]:
    return {
        **context,
        "client_transport": client_transport,
    }


def _transport_result(
    *,
    context: dict[str, str],
    client_transport: str,
    samples: list[dict],
    summary: dict,
) -> dict:
    return {
        "transport": client_transport,
        "benchmark_contract": _benchmark_contract(
            client_transport=client_transport,
            **context,
        ),
        "samples": samples,
        "summary": summary,
    }


def _scoped_experiment_folder(base_name: str, backend_name: str) -> str:
    return f"{base_name}_{_router_topology_for_backend(backend_name)}"


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


def _chain_transport_ratios(http_summary: dict, ws_summary: dict) -> dict[str, float]:
    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    total_ratio = ratio(
        float(ws_summary["total_chain_ms_p50"]), float(http_summary["total_chain_ms_p50"])
    )
    continuation_ratio = ratio(
        float(ws_summary["continuation_only_total_ms_p50"]),
        float(http_summary["continuation_only_total_ms_p50"]),
    )

    return {
        "ws_over_http_total_chain": total_ratio,
        "ws_over_http_continuation_only": continuation_ratio,
        "ws_vs_http_total_chain_delta_pct": (1.0 - total_ratio) * 100.0,
        "ws_vs_http_continuation_only_delta_pct": (1.0 - continuation_ratio)
            * 100.0,
    }


def _bfcl_transport_ratios(http_summary: dict, ws_summary: dict) -> dict[str, float]:
    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    total_ratio = ratio(
        float(ws_summary["total_suite_ms_p50"]),
        float(http_summary["total_suite_ms_p50"]),
    )
    request_ratio = ratio(
        float(ws_summary["total_tool_request_ms_p50"]),
        float(http_summary["total_tool_request_ms_p50"]),
    )
    output_ratio = ratio(
        float(ws_summary["total_tool_output_ms_p50"]),
        float(http_summary["total_tool_output_ms_p50"]),
    )

    return {
        "ws_over_http_total_suite": total_ratio,
        "ws_over_http_tool_request_total": request_ratio,
        "ws_over_http_tool_output_total": output_ratio,
        "ws_vs_http_total_suite_delta_pct": (1.0 - total_ratio) * 100.0,
        "ws_vs_http_tool_request_total_delta_pct": (1.0 - request_ratio) * 100.0,
        "ws_vs_http_tool_output_total_delta_pct": (1.0 - output_ratio) * 100.0,
    }


def _bfcl_free_choice_transport_ratios(
    http_summary: dict, ws_summary: dict
) -> dict[str, float]:
    ratios = _bfcl_transport_ratios(http_summary, ws_summary)
    ratios.update(
        {
            "ws_minus_http_matched_turn_rate": float(
                ws_summary["matched_turn_rate_mean"]
            )
            - float(http_summary["matched_turn_rate_mean"]),
            "ws_minus_http_turn_execution_rate": float(
                ws_summary["turn_execution_rate_mean"]
            )
            - float(http_summary["turn_execution_rate_mean"]),
            "ws_minus_http_completed_scenario_rate": float(
                ws_summary["completed_scenario_rate_mean"]
            )
            - float(http_summary["completed_scenario_rate_mean"]),
        }
    )
    return ratios


def _run_http_bfcl_subset_suite_sample(
    client, model: str, scenarios: list[BfclScenario]
) -> dict:
    request_max_output_tokens = _model_tool_request_max_output_tokens()
    result_max_output_tokens = _model_tool_result_max_output_tokens()
    total_tool_request_ms = 0.0
    total_tool_output_ms = 0.0
    total_turns = 0
    all_expected_tools_matched = True
    scenario_results = []

    for scenario in scenarios:
        with tempfile.TemporaryDirectory(prefix=f"{scenario.id}_") as temp_dir:
            workspace_root = Path(temp_dir)
            materialize_bfcl_scenario(workspace_root, scenario)
            workspace = BfclWorkspace(workspace_root)
            previous_response_id: str | None = None
            per_turn_ms: list[float] = []
            per_turn_requests_ms: list[dict[str, float | int | str]] = []
            observed_tools: list[str] = []

            for turn_index, turn in enumerate(scenario.turns, start=1):
                first_stream = client.responses.create(
                    model=model,
                    input=turn.prompt,
                    previous_response_id=previous_response_id,
                    instructions=_bfcl_request_instructions_for_scenario(scenario),
                    temperature=0,
                    max_output_tokens=request_max_output_tokens,
                    store=True,
                    stream=True,
                    tools=bfcl_subset_tools(),
                    tool_choice=_bfcl_tool_choice(turn.expected_tool),
                )
                first_response, tool_request_ms = _collect_http_completed_ms(first_stream)
                function_calls = _response_function_calls(first_response.output)
                if not function_calls:
                    raise AssertionError(
                        f"expected BFCL tool call in http scenario={scenario.id} turn={turn_index}"
                    )

                observed_tool = function_calls[0].name
                observed_tools.append(observed_tool)
                if observed_tool != turn.expected_tool:
                    all_expected_tools_matched = False
                    raise AssertionError(
                        "unexpected BFCL http tool selection "
                        f"scenario={scenario.id} turn={turn_index} "
                        f"expected={turn.expected_tool} observed={observed_tool}"
                    )

                tool_output_items, _ = _bfcl_tool_output_from_http_function_call(
                    workspace, function_calls[0]
                )
                second_stream = client.responses.create(
                    model=model,
                    input=tool_output_items,
                    previous_response_id=first_response.id,
                    instructions=_bfcl_tool_result_instructions(),
                    temperature=0,
                    max_output_tokens=result_max_output_tokens,
                    store=True,
                    stream=True,
                    tools=bfcl_subset_tools(),
                    tool_choice="auto",
                )
                second_response, tool_output_ms = _collect_http_completed_ms(second_stream)

                turn_total_ms = tool_request_ms + tool_output_ms
                per_turn_ms.append(turn_total_ms)
                per_turn_requests_ms.append(
                    {
                        "turn_index": turn_index,
                        "expected_tool": turn.expected_tool,
                        "observed_tool": observed_tool,
                        "tool_request_ms": tool_request_ms,
                        "tool_output_ms": tool_output_ms,
                        "turn_total_ms": turn_total_ms,
                    }
                )
                previous_response_id = second_response.id
                total_tool_request_ms += tool_request_ms
                total_tool_output_ms += tool_output_ms

            total_turns += len(scenario.turns)
            scenario_results.append(
                {
                    "scenario_id": scenario.id,
                    "source_dataset": scenario.source_dataset,
                    "source_id": scenario.source_id,
                    "turns": len(scenario.turns),
                    "total_chain_ms": sum(per_turn_ms),
                    "per_turn_completed_ms": per_turn_ms,
                    "per_turn_requests_ms": per_turn_requests_ms,
                    "expected_tools": [turn.expected_tool for turn in scenario.turns],
                    "observed_tools": observed_tools,
                    "final_files": workspace.list_files(),
                }
            )

    return {
        "scenario_count": len(scenarios),
        "scenario_results": scenario_results,
        "total_turns": total_turns,
        "total_suite_ms": total_tool_request_ms + total_tool_output_ms,
        "total_tool_request_ms": total_tool_request_ms,
        "total_tool_output_ms": total_tool_output_ms,
        "all_expected_tools_matched": all_expected_tools_matched,
    }


async def _run_ws_bfcl_subset_suite_sample(
    ws_url: str, model: str, scenarios: list[BfclScenario]
) -> dict:
    import websockets

    request_max_output_tokens = _model_tool_request_max_output_tokens()
    result_max_output_tokens = _model_tool_result_max_output_tokens()
    total_tool_request_ms = 0.0
    total_tool_output_ms = 0.0
    total_turns = 0
    all_expected_tools_matched = True
    connect_ms_total = 0.0
    scenario_results = []

    for scenario in scenarios:
        with tempfile.TemporaryDirectory(prefix=f"{scenario.id}_") as temp_dir:
            workspace_root = Path(temp_dir)
            materialize_bfcl_scenario(workspace_root, scenario)
            workspace = BfclWorkspace(workspace_root)

            connect_started_at = time.perf_counter()
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                connect_ms_total += (time.perf_counter() - connect_started_at) * 1000
                previous_response_id: str | None = None
                per_turn_ms: list[float] = []
                per_turn_requests_ms: list[dict[str, float | int | str]] = []
                observed_tools: list[str] = []

                for turn_index, turn in enumerate(scenario.turns, start=1):
                    first_completed, tool_request_ms = await _collect_ws_terminal_event(
                        websocket,
                        _ws_request(
                            model=model,
                            input=turn.prompt,
                            instructions=_bfcl_request_instructions_for_scenario(scenario),
                            temperature=0,
                            max_output_tokens=request_max_output_tokens,
                            store=True,
                            tools=bfcl_subset_tools(),
                            tool_choice=_bfcl_tool_choice(turn.expected_tool),
                            previous_response_id=previous_response_id,
                        ),
                    )
                    function_calls = _completed_response_function_calls(first_completed)
                    if not function_calls:
                        raise AssertionError(
                            f"expected BFCL tool call in websocket scenario={scenario.id} turn={turn_index}"
                        )

                    observed_tool = function_calls[0]["name"]
                    observed_tools.append(observed_tool)
                    if observed_tool != turn.expected_tool:
                        all_expected_tools_matched = False
                        raise AssertionError(
                            "unexpected BFCL websocket tool selection "
                            f"scenario={scenario.id} turn={turn_index} "
                            f"expected={turn.expected_tool} observed={observed_tool}"
                        )

                    tool_output_items, _ = _bfcl_tool_output_from_ws_function_call(
                        workspace, function_calls[0]
                    )
                    second_completed, tool_output_ms = await _collect_ws_terminal_event(
                        websocket,
                        _ws_request(
                            model=model,
                            input=tool_output_items,
                            instructions=_bfcl_tool_result_instructions(),
                            temperature=0,
                            max_output_tokens=result_max_output_tokens,
                            store=True,
                            tools=bfcl_subset_tools(),
                            tool_choice="auto",
                            previous_response_id=first_completed["response"]["id"],
                        ),
                    )

                    turn_total_ms = tool_request_ms + tool_output_ms
                    per_turn_ms.append(turn_total_ms)
                    per_turn_requests_ms.append(
                        {
                            "turn_index": turn_index,
                            "expected_tool": turn.expected_tool,
                            "observed_tool": observed_tool,
                            "tool_request_ms": tool_request_ms,
                            "tool_output_ms": tool_output_ms,
                            "turn_total_ms": turn_total_ms,
                        }
                    )
                    previous_response_id = second_completed["response"]["id"]
                    total_tool_request_ms += tool_request_ms
                    total_tool_output_ms += tool_output_ms

                total_turns += len(scenario.turns)
                scenario_results.append(
                    {
                        "scenario_id": scenario.id,
                        "source_dataset": scenario.source_dataset,
                        "source_id": scenario.source_id,
                        "turns": len(scenario.turns),
                        "total_chain_ms": sum(per_turn_ms),
                        "per_turn_completed_ms": per_turn_ms,
                        "per_turn_requests_ms": per_turn_requests_ms,
                        "expected_tools": [turn.expected_tool for turn in scenario.turns],
                        "observed_tools": observed_tools,
                        "final_files": workspace.list_files(),
                    }
                )

    return {
        "scenario_count": len(scenarios),
        "scenario_results": scenario_results,
        "total_turns": total_turns,
        "connect_ms_total": connect_ms_total,
        "total_suite_ms": total_tool_request_ms + total_tool_output_ms,
        "total_tool_request_ms": total_tool_request_ms,
        "total_tool_output_ms": total_tool_output_ms,
        "all_expected_tools_matched": all_expected_tools_matched,
    }


def _run_http_bfcl_subset_free_choice_suite_sample(
    client, model: str, scenarios: list[BfclScenario]
) -> dict:
    request_max_output_tokens = _model_tool_request_max_output_tokens()
    result_max_output_tokens = _model_tool_result_max_output_tokens()
    total_tool_request_ms = 0.0
    total_tool_output_ms = 0.0
    total_turns_requested = 0
    total_turns_executed = 0
    matched_turns = 0
    missing_tool_turns = 0
    mismatched_tool_turns = 0
    completed_scenarios = 0
    terminated_scenarios = 0
    scenario_results = []

    for scenario in scenarios:
        with tempfile.TemporaryDirectory(prefix=f"{scenario.id}_") as temp_dir:
            workspace_root = Path(temp_dir)
            materialize_bfcl_scenario(workspace_root, scenario)
            workspace = BfclWorkspace(workspace_root)
            previous_response_id: str | None = None
            scenario_records: list[dict[str, float | int | str | bool | None]] = []
            scenario_completed = True
            scenario_failure_reason: str | None = None

            for turn_index, turn in enumerate(scenario.turns, start=1):
                total_turns_requested += 1

                try:
                    first_stream = client.responses.create(
                        model=model,
                        input=turn.prompt,
                        previous_response_id=previous_response_id,
                        instructions=_bfcl_request_instructions_for_scenario(scenario),
                        temperature=0,
                        max_output_tokens=request_max_output_tokens,
                        store=True,
                        stream=True,
                        tools=bfcl_subset_tools(),
                        tool_choice="auto",
                    )
                    first_response, tool_request_ms = _collect_http_completed_ms(first_stream)
                except Exception as exc:
                    scenario_completed = False
                    scenario_failure_reason = "tool_request_exception"
                    scenario_records.append(
                        {
                            "turn_index": turn_index,
                            "expected_tool": turn.expected_tool,
                            "observed_tool": None,
                            "matched_expected_tool": False,
                            "tool_request_ms": 0.0,
                            "tool_output_ms": None,
                            "turn_total_ms": 0.0,
                            "status": "tool_request_exception",
                            "error": str(exc),
                        }
                    )
                    break

                total_tool_request_ms += tool_request_ms
                function_calls = _response_function_calls(first_response.output)
                observed_tool = function_calls[0].name if function_calls else None
                matched_expected_tool = observed_tool == turn.expected_tool
                record: dict[str, float | int | str | bool | None] = {
                    "turn_index": turn_index,
                    "expected_tool": turn.expected_tool,
                    "observed_tool": observed_tool,
                    "matched_expected_tool": matched_expected_tool,
                    "tool_request_ms": tool_request_ms,
                    "tool_output_ms": None,
                    "turn_total_ms": tool_request_ms,
                    "status": "tool_request_completed",
                    "response_id": first_response.id,
                    "output_types": ",".join(item.type for item in first_response.output),
                }

                if not function_calls:
                    missing_tool_turns += 1
                    scenario_completed = False
                    scenario_failure_reason = "missing_function_call"
                    record["status"] = "missing_function_call"
                    scenario_records.append(record)
                    break

                if not matched_expected_tool:
                    mismatched_tool_turns += 1
                    scenario_completed = False
                    scenario_failure_reason = "unexpected_tool"
                    record["status"] = "unexpected_tool"
                    scenario_records.append(record)
                    break

                matched_turns += 1

                try:
                    tool_output_items, _ = _bfcl_tool_output_from_http_function_call(
                        workspace, function_calls[0]
                    )
                    second_stream = client.responses.create(
                        model=model,
                        input=tool_output_items,
                        previous_response_id=first_response.id,
                        instructions=_bfcl_tool_result_instructions(),
                        temperature=0,
                        max_output_tokens=result_max_output_tokens,
                        store=True,
                        stream=True,
                        tools=bfcl_subset_tools(),
                        tool_choice="auto",
                    )
                    second_response, tool_output_ms = _collect_http_completed_ms(second_stream)
                except Exception as exc:
                    scenario_completed = False
                    scenario_failure_reason = "tool_output_exception"
                    record["status"] = "tool_output_exception"
                    record["error"] = str(exc)
                    scenario_records.append(record)
                    break

                total_tool_output_ms += tool_output_ms
                total_turns_executed += 1
                record["tool_output_ms"] = tool_output_ms
                record["turn_total_ms"] = tool_request_ms + tool_output_ms
                record["status"] = "matched_completed"
                previous_response_id = second_response.id
                scenario_records.append(record)

            if scenario_completed and len(scenario_records) == len(scenario.turns):
                completed_scenarios += 1
            else:
                terminated_scenarios += 1

            scenario_results.append(
                {
                    "scenario_id": scenario.id,
                    "source_dataset": scenario.source_dataset,
                    "source_id": scenario.source_id,
                    "turns_requested": len(scenario.turns),
                    "turns_executed": sum(
                        1
                        for record in scenario_records
                        if record.get("status") == "matched_completed"
                    ),
                    "matched_turns": sum(
                        1
                        for record in scenario_records
                        if bool(record.get("matched_expected_tool"))
                    ),
                    "completed": scenario_completed
                    and len(scenario_records) == len(scenario.turns),
                    "failure_reason": scenario_failure_reason,
                    "turn_records": scenario_records,
                    "final_files": workspace.list_files(),
                }
            )

    return {
        "scenario_count": len(scenarios),
        "scenario_results": scenario_results,
        "total_turns_requested": total_turns_requested,
        "total_turns_executed": total_turns_executed,
        "matched_turns": matched_turns,
        "missing_tool_turns": missing_tool_turns,
        "mismatched_tool_turns": mismatched_tool_turns,
        "completed_scenarios": completed_scenarios,
        "terminated_scenarios": terminated_scenarios,
        "total_suite_ms": total_tool_request_ms + total_tool_output_ms,
        "total_tool_request_ms": total_tool_request_ms,
        "total_tool_output_ms": total_tool_output_ms,
    }


async def _run_ws_bfcl_subset_free_choice_suite_sample(
    ws_url: str, model: str, scenarios: list[BfclScenario]
) -> dict:
    import websockets

    request_max_output_tokens = _model_tool_request_max_output_tokens()
    result_max_output_tokens = _model_tool_result_max_output_tokens()
    total_tool_request_ms = 0.0
    total_tool_output_ms = 0.0
    total_turns_requested = 0
    total_turns_executed = 0
    matched_turns = 0
    missing_tool_turns = 0
    mismatched_tool_turns = 0
    completed_scenarios = 0
    terminated_scenarios = 0
    connect_ms_total = 0.0
    scenario_results = []

    for scenario in scenarios:
        with tempfile.TemporaryDirectory(prefix=f"{scenario.id}_") as temp_dir:
            workspace_root = Path(temp_dir)
            materialize_bfcl_scenario(workspace_root, scenario)
            workspace = BfclWorkspace(workspace_root)
            scenario_records: list[dict[str, float | int | str | bool | None]] = []
            scenario_completed = True
            scenario_failure_reason: str | None = None

            connect_started_at = time.perf_counter()
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                connect_ms_total += (time.perf_counter() - connect_started_at) * 1000
                previous_response_id: str | None = None

                for turn_index, turn in enumerate(scenario.turns, start=1):
                    total_turns_requested += 1

                    try:
                        first_completed, tool_request_ms = await _collect_ws_terminal_event(
                            websocket,
                            _ws_request(
                                model=model,
                                input=turn.prompt,
                                instructions=_bfcl_request_instructions_for_scenario(
                                    scenario
                                ),
                                temperature=0,
                                max_output_tokens=request_max_output_tokens,
                                store=True,
                                tools=bfcl_subset_tools(),
                                tool_choice="auto",
                                previous_response_id=previous_response_id,
                            ),
                        )
                    except Exception as exc:
                        scenario_completed = False
                        scenario_failure_reason = "tool_request_exception"
                        scenario_records.append(
                            {
                                "turn_index": turn_index,
                                "expected_tool": turn.expected_tool,
                                "observed_tool": None,
                                "matched_expected_tool": False,
                                "tool_request_ms": 0.0,
                                "tool_output_ms": None,
                                "turn_total_ms": 0.0,
                                "status": "tool_request_exception",
                                "error": str(exc),
                            }
                        )
                        break

                    total_tool_request_ms += tool_request_ms
                    function_calls = _completed_response_function_calls(first_completed)
                    observed_tool = function_calls[0]["name"] if function_calls else None
                    matched_expected_tool = observed_tool == turn.expected_tool
                    record: dict[str, float | int | str | bool | None] = {
                        "turn_index": turn_index,
                        "expected_tool": turn.expected_tool,
                        "observed_tool": observed_tool,
                        "matched_expected_tool": matched_expected_tool,
                        "tool_request_ms": tool_request_ms,
                        "tool_output_ms": None,
                        "turn_total_ms": tool_request_ms,
                        "status": "tool_request_completed",
                        "response_id": first_completed.get("response", {}).get("id"),
                    }

                    if not function_calls:
                        missing_tool_turns += 1
                        scenario_completed = False
                        scenario_failure_reason = "missing_function_call"
                        record["status"] = "missing_function_call"
                        scenario_records.append(record)
                        break

                    if not matched_expected_tool:
                        mismatched_tool_turns += 1
                        scenario_completed = False
                        scenario_failure_reason = "unexpected_tool"
                        record["status"] = "unexpected_tool"
                        scenario_records.append(record)
                        break

                    matched_turns += 1

                    try:
                        tool_output_items, _ = _bfcl_tool_output_from_ws_function_call(
                            workspace, function_calls[0]
                        )
                        second_completed, tool_output_ms = await _collect_ws_terminal_event(
                            websocket,
                            _ws_request(
                                model=model,
                                input=tool_output_items,
                                instructions=_bfcl_tool_result_instructions(),
                                temperature=0,
                                max_output_tokens=result_max_output_tokens,
                                store=True,
                                tools=bfcl_subset_tools(),
                                tool_choice="auto",
                                previous_response_id=first_completed["response"]["id"],
                            ),
                        )
                    except Exception as exc:
                        scenario_completed = False
                        scenario_failure_reason = "tool_output_exception"
                        record["status"] = "tool_output_exception"
                        record["error"] = str(exc)
                        scenario_records.append(record)
                        break

                    total_tool_output_ms += tool_output_ms
                    total_turns_executed += 1
                    record["tool_output_ms"] = tool_output_ms
                    record["turn_total_ms"] = tool_request_ms + tool_output_ms
                    record["status"] = "matched_completed"
                    previous_response_id = second_completed["response"]["id"]
                    scenario_records.append(record)

            if scenario_completed and len(scenario_records) == len(scenario.turns):
                completed_scenarios += 1
            else:
                terminated_scenarios += 1

            scenario_results.append(
                {
                    "scenario_id": scenario.id,
                    "source_dataset": scenario.source_dataset,
                    "source_id": scenario.source_id,
                    "turns_requested": len(scenario.turns),
                    "turns_executed": sum(
                        1
                        for record in scenario_records
                        if record.get("status") == "matched_completed"
                    ),
                    "matched_turns": sum(
                        1
                        for record in scenario_records
                        if bool(record.get("matched_expected_tool"))
                    ),
                    "completed": scenario_completed
                    and len(scenario_records) == len(scenario.turns),
                    "failure_reason": scenario_failure_reason,
                    "turn_records": scenario_records,
                    "final_files": workspace.list_files(),
                }
            )

    return {
        "scenario_count": len(scenarios),
        "scenario_results": scenario_results,
        "total_turns_requested": total_turns_requested,
        "total_turns_executed": total_turns_executed,
        "matched_turns": matched_turns,
        "missing_tool_turns": missing_tool_turns,
        "mismatched_tool_turns": mismatched_tool_turns,
        "completed_scenarios": completed_scenarios,
        "terminated_scenarios": terminated_scenarios,
        "connect_ms_total": connect_ms_total,
        "total_suite_ms": total_tool_request_ms + total_tool_output_ms,
        "total_tool_request_ms": total_tool_request_ms,
        "total_tool_output_ms": total_tool_output_ms,
    }


def test_bfcl_free_choice_transport_ratios_track_quality_deltas():
    ratios = _bfcl_free_choice_transport_ratios(
        {
            "total_suite_ms_p50": 100.0,
            "total_tool_request_ms_p50": 60.0,
            "total_tool_output_ms_p50": 40.0,
            "matched_turn_rate_mean": 0.50,
            "turn_execution_rate_mean": 0.60,
            "completed_scenario_rate_mean": 0.40,
        },
        {
            "total_suite_ms_p50": 90.0,
            "total_tool_request_ms_p50": 50.0,
            "total_tool_output_ms_p50": 40.0,
            "matched_turn_rate_mean": 0.75,
            "turn_execution_rate_mean": 0.80,
            "completed_scenario_rate_mean": 0.55,
        },
    )

    assert ratios["ws_over_http_total_suite"] == pytest.approx(0.9)
    assert ratios["ws_minus_http_matched_turn_rate"] == pytest.approx(0.25)
    assert ratios["ws_minus_http_turn_execution_rate"] == pytest.approx(0.20)
    assert ratios["ws_minus_http_completed_scenario_rate"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("backend_name", "expected_worker_transport", "expected_router_topology"),
    [
        ("http", "http", "regular_http_worker"),
        ("grpc", "grpc", "regular_grpc_worker"),
        ("pd", "http", "pd_http_workers"),
    ],
)
def test_benchmark_contract_maps_backend_axes(
    backend_name: str,
    expected_worker_transport: str,
    expected_router_topology: str,
):
    contract = _benchmark_contract(
        **_benchmark_context(
            benchmark_family="transport_qos",
            run_class="test_contract",
            backend_name=backend_name,
            model="Qwen/Qwen2.5-72B-Instruct",
            store_mode="store_false",
            workload_kind="single_turn_text",
        ),
        client_transport="websocket",
    )

    assert contract["worker_transport"] == expected_worker_transport
    assert contract["router_topology"] == expected_router_topology
    assert contract["client_transport"] == "websocket"
    assert contract["model_id"] == "Qwen/Qwen2.5-72B-Instruct"


def test_benchmark_context_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported benchmark backend"):
        _benchmark_context(
            benchmark_family="transport_qos",
            run_class="test_contract",
            backend_name="ws_worker",
            model="Qwen/Qwen2.5-72B-Instruct",
            store_mode="store_false",
            workload_kind="single_turn_text",
        )


def test_scoped_experiment_folder_uses_router_topology_suffix():
    assert _scoped_experiment_folder("benchmark_http_ws_compare", "grpc") == (
        "benchmark_http_ws_compare_regular_grpc_worker"
    )


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model("qwen-0.5b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestWsMicrobench:
    """WebSocket benchmark for the Responses route on a single small model."""

    def test_ws_microbench(self, setup_backend):
        backend_name, model, _, gateway = setup_backend

        concurrency_levels = [
            int(value)
            for value in os.environ.get("SGLANG_WS_BENCH_CONCURRENCY", "1,2,4").split(",")
            if value.strip()
        ]
        samples_per_concurrency = int(
            os.environ.get("SGLANG_WS_BENCH_SAMPLES_PER_CONCURRENCY", "2")
        )
        experiment_folder = _scoped_experiment_folder(
            os.environ.get(
                "SGLANG_WS_BENCH_EXPERIMENT",
                f"benchmark_ws_microbench_{model.replace('/', '_')}",
            ),
            backend_name,
        )
        benchmark_context = _benchmark_context(
            benchmark_family="transport_qos",
            run_class="ws_microbench_profile",
            backend_name=backend_name,
            model=model,
            store_mode="store_false",
            workload_kind="single_turn_text",
        )

        payload = asyncio.run(
            _run_concurrency_profile(
                _gateway_ws_url(gateway.base_url),
                model,
                concurrency_levels,
                samples_per_concurrency,
            )
        )
        payload["benchmark_contract"] = _benchmark_contract(
            client_transport="websocket",
            **benchmark_context,
        )
        payload["worker_backend"] = backend_name
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
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestResponsesTransportCompare:
    """Small-model transport comparison for HTTP SSE vs WebSocket Responses."""

    def test_http_vs_ws_transport_compare(self, setup_backend):
        backend_name, model, client, gateway = setup_backend

        samples = int(os.environ.get("SGLANG_HTTP_WS_COMPARE_SAMPLES", "2"))
        experiment_folder = _scoped_experiment_folder(
            os.environ.get(
                "SGLANG_HTTP_WS_COMPARE_EXPERIMENT",
                f"benchmark_http_ws_compare_{model.replace('/', '_')}",
            ),
            backend_name,
        )
        benchmark_context = _benchmark_context(
            benchmark_family="transport_qos",
            run_class="http_vs_ws_transport_compare",
            backend_name=backend_name,
            model=model,
            store_mode="store_false",
            workload_kind="single_turn_text",
        )

        timeout_secs = float(
            os.environ.get("SGLANG_HTTP_WS_COMPARE_TIMEOUT_SECS", "90")
        )
        http_samples = [
            _run_single_http_sample(gateway.base_url, model, timeout_secs)
            for _ in range(samples)
        ]
        ws_samples = asyncio.run(
            _run_ws_sample_batch(_gateway_ws_url(gateway.base_url), model, samples)
        )

        http_summary = _summarize_samples(http_samples)
        ws_summary = _summarize_samples(ws_samples)

        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
            "router_url": gateway.base_url,
            "model": model,
            "experiment_folder": experiment_folder,
            "samples_per_transport": samples,
            "http": _transport_result(
                context=benchmark_context,
                client_transport="http_sse",
                samples=http_samples,
                summary=http_summary,
            ),
            "websocket": _transport_result(
                context=benchmark_context,
                client_transport="websocket",
                samples=ws_samples,
                summary=ws_summary,
            ),
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
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestResponsesContinuationChainCompare:
    """Long-chain continuation comparison for HTTP vs persistent WS."""

    def test_http_vs_ws_continuation_chain_compare(self, setup_backend):
        backend_name, model, client, gateway = setup_backend

        turns = int(os.environ.get("SGLANG_HTTP_WS_CHAIN_TURNS", "20"))
        samples = int(os.environ.get("SGLANG_HTTP_WS_CHAIN_SAMPLES", "1"))
        experiment_folder = _scoped_experiment_folder(
            os.environ.get(
                "SGLANG_HTTP_WS_CHAIN_EXPERIMENT",
                (
                    "benchmark_http_ws_chain_compare_"
                    f"{model.replace('/', '_')}_{turns}turns"
                ),
            ),
            backend_name,
        )
        benchmark_context = _benchmark_context(
            benchmark_family="continuation_qos",
            run_class="http_vs_ws_continuation_compare",
            backend_name=backend_name,
            model=model,
            store_mode="store_true",
            workload_kind="incremental_text_continuation",
        )

        timeout_secs = float(
            os.environ.get("SGLANG_HTTP_WS_CHAIN_TIMEOUT_SECS", "90")
        )
        http_samples = [
            _run_http_continuation_chain_sample(
                gateway.base_url, model, turns, timeout_secs
            )
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
        http_summary = _summarize_chain_samples(http_samples)
        ws_summary = _summarize_chain_samples(ws_samples)

        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
            "router_url": gateway.base_url,
            "model": model,
            "turns": turns,
            "samples": samples,
            "experiment_folder": experiment_folder,
            "http": _transport_result(
                context=benchmark_context,
                client_transport="http_sse",
                samples=http_samples,
                summary=http_summary,
            ),
            "websocket": _transport_result(
                context=benchmark_context,
                client_transport="websocket",
                samples=ws_samples,
                summary=ws_summary,
            ),
            "ratios": _chain_transport_ratios(http_summary, ws_summary),
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
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestResponsesToolOutputChainCompare:
    """Tool-output-heavy continuation comparison for HTTP vs persistent WS."""

    def test_http_vs_ws_tool_output_chain_compare(self, setup_backend):
        backend_name, model, client, gateway = setup_backend

        tool_turns = int(os.environ.get("SGLANG_HTTP_WS_TOOL_CHAIN_TURNS", "20"))
        samples = int(os.environ.get("SGLANG_HTTP_WS_TOOL_CHAIN_SAMPLES", "1"))
        experiment_folder = _scoped_experiment_folder(
            os.environ.get(
                "SGLANG_HTTP_WS_TOOL_CHAIN_EXPERIMENT",
                (
                    "benchmark_http_ws_tool_chain_compare_"
                    f"{model.replace('/', '_')}_{tool_turns}toolturns"
                ),
            ),
            backend_name,
        )
        benchmark_context = _benchmark_context(
            benchmark_family="continuation_qos",
            run_class="http_vs_ws_tool_output_compare",
            backend_name=backend_name,
            model=model,
            store_mode="store_true",
            workload_kind="incremental_tool_output_continuation",
        )

        timeout_secs = float(
            os.environ.get("SGLANG_HTTP_WS_TOOL_CHAIN_TIMEOUT_SECS", "90")
        )
        http_samples = [
            _run_http_tool_output_chain_sample(
                gateway.base_url, model, tool_turns, timeout_secs
            )
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
        http_summary = _summarize_chain_samples(http_samples)
        ws_summary = _summarize_chain_samples(ws_samples)

        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
            "router_url": gateway.base_url,
            "model": model,
            "tool_turns": tool_turns,
            "samples": samples,
            "experiment_folder": experiment_folder,
            "http": _transport_result(
                context=benchmark_context,
                client_transport="http_sse",
                samples=http_samples,
                summary=http_summary,
            ),
            "websocket": _transport_result(
                context=benchmark_context,
                client_transport="websocket",
                samples=ws_samples,
                summary=ws_summary,
            ),
            "ratios": _chain_transport_ratios(http_summary, ws_summary),
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
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestResponsesModelGeneratedToolChainCompare:
    """Model-generated tool-call comparison for HTTP vs persistent WS."""

    def test_http_vs_ws_model_generated_tool_chain_compare(self, setup_backend):
        backend_name, model, client, gateway = setup_backend

        turn_profiles = [
            int(value)
            for value in os.environ.get(
                "SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_TURNS", "5"
            ).split(",")
            if value.strip()
        ]
        samples = int(os.environ.get("SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_SAMPLES", "1"))
        experiment_folder = _scoped_experiment_folder(
            os.environ.get(
                "SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_EXPERIMENT",
                f"benchmark_http_ws_model_tool_chain_compare_{model.replace('/', '_')}",
            ),
            backend_name,
        )
        benchmark_context = _benchmark_context(
            benchmark_family="tool_loop_transport_qos",
            run_class="http_vs_ws_model_tool_compare",
            backend_name=backend_name,
            model=model,
            store_mode="store_true",
            workload_kind="model_generated_tool_loop_required",
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
            http_summary = _summarize_chain_samples(http_samples)
            ws_summary = _summarize_chain_samples(ws_samples)

            profile_results.append(
                {
                    "turns": turns,
                    "http": _transport_result(
                        context=benchmark_context,
                        client_transport="http_sse",
                        samples=http_samples,
                        summary=http_summary,
                    ),
                    "websocket": _transport_result(
                        context=benchmark_context,
                        client_transport="websocket",
                        samples=ws_samples,
                        summary=ws_summary,
                    ),
                    "ratios": _chain_transport_ratios(http_summary, ws_summary),
                }
            )

        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
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


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Diagnostics are only meaningful sequentially.")
@pytest.mark.model("qwen-3b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestResponsesModelGeneratedToolChainDiagnostic:
    """Turn-level diagnostic harness for long-chain local tool-loop failures."""

    def test_http_vs_ws_model_generated_tool_chain_diagnostic(self, setup_backend):
        backend_name, model, _, gateway = setup_backend

        turns = int(
            os.environ.get("SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_DIAGNOSTIC_TURNS", "20")
        )
        timeout_secs = float(
            os.environ.get("SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_DIAGNOSTIC_TIMEOUT_SECS", "45")
        )
        experiment_folder = _scoped_experiment_folder(
            os.environ.get(
                "SGLANG_HTTP_WS_MODEL_TOOL_CHAIN_DIAGNOSTIC_EXPERIMENT",
                (
                    "benchmark_http_ws_model_tool_chain_diagnostic_"
                    f"{model.replace('/', '_')}_{turns}turns"
                ),
            ),
            backend_name,
        )
        benchmark_context = _benchmark_context(
            benchmark_family="tool_loop_transport_qos",
            run_class="http_vs_ws_model_tool_diagnostic",
            backend_name=backend_name,
            model=model,
            store_mode="store_true",
            workload_kind="model_generated_tool_loop_required_diagnostic",
        )

        http_result = _run_http_model_generated_tool_chain_diagnostic(
            gateway.base_url, model, turns, timeout_secs
        )
        ws_result = asyncio.run(
            _run_ws_model_generated_tool_chain_diagnostic(
                _gateway_ws_url(gateway.base_url), model, turns, timeout_secs
            )
        )

        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
            "router_url": gateway.base_url,
            "model": model,
            "turns": turns,
            "timeout_secs": timeout_secs,
            "experiment_folder": experiment_folder,
            "http_contract": _benchmark_contract(
                client_transport="http_json",
                **benchmark_context,
            ),
            "websocket_contract": _benchmark_contract(
                client_transport="websocket",
                **benchmark_context,
            ),
            "http": http_result,
            "websocket": ws_result,
        }

        summary_path = _write_summary(experiment_folder, payload)
        logger.info(
            "HTTP-vs-WS model-generated tool-chain diagnostic summary written to %s",
            summary_path,
        )

        assert payload["http"]["turns_requested"] == turns
        assert payload["websocket"]["turns_requested"] == turns
        assert payload["http"]["records"] or payload["websocket"]["records"]


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="Benchmark timing is only meaningful sequentially.")
@pytest.mark.model("qwen-3b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
class TestResponsesBfclSubsetCompare:
    """BFCL-derived multi-turn filesystem workload for HTTP vs persistent WS."""

    def test_http_vs_ws_bfcl_subset_compare(self, setup_backend):
        backend_name, model, client, gateway = setup_backend

        scenarios = _selected_bfcl_subset_scenarios()
        samples = int(os.environ.get("SGLANG_HTTP_WS_BFCL_SUBSET_SAMPLES", "1"))
        experiment_folder = _scoped_experiment_folder(
            os.environ.get(
                "SGLANG_HTTP_WS_BFCL_SUBSET_EXPERIMENT",
                f"benchmark_http_ws_bfcl_subset_compare_{model.replace('/', '_')}",
            ),
            backend_name,
        )
        benchmark_context = _benchmark_context(
            benchmark_family="tool_loop_transport_qos",
            run_class="http_vs_ws_bfcl_subset_compare",
            backend_name=backend_name,
            model=model,
            store_mode="store_true",
            workload_kind="bfcl_multi_turn_forced_tool",
        )

        http_samples = [
            _run_http_bfcl_subset_suite_sample(client, model, scenarios)
            for _ in range(samples)
        ]
        ws_samples = [
            asyncio.run(
                _run_ws_bfcl_subset_suite_sample(
                    _gateway_ws_url(gateway.base_url), model, scenarios
                )
            )
            for _ in range(samples)
        ]

        http_summary = _summarize_bfcl_suite_samples(http_samples)
        ws_summary = _summarize_bfcl_suite_samples(ws_samples)
        payload = {
            "benchmark_context": benchmark_context,
            "worker_backend": backend_name,
            "router_url": gateway.base_url,
            "model": model,
            "samples": samples,
            "experiment_folder": experiment_folder,
            "scenarios": [
                {
                    "id": scenario.id,
                    "source_dataset": scenario.source_dataset,
                    "source_id": scenario.source_id,
                    "turns": len(scenario.turns),
                    "description": scenario.description,
                }
                for scenario in scenarios
            ],
            "http": _transport_result(
                context=benchmark_context,
                client_transport="http_sse",
                samples=http_samples,
                summary=http_summary,
            ),
            "websocket": _transport_result(
                context=benchmark_context,
                client_transport="websocket",
                samples=ws_samples,
                summary=ws_summary,
            ),
            "ratios": _bfcl_transport_ratios(http_summary, ws_summary),
        }

        summary_path = _write_summary(experiment_folder, payload)
        logger.info("HTTP-vs-WS BFCL subset summary written to %s", summary_path)

        assert payload["scenarios"]
        assert http_summary["all_expected_tools_matched"] == 1
        assert ws_summary["all_expected_tools_matched"] == 1
        assert float(http_samples[0]["total_suite_ms"]) > 0
        assert float(ws_samples[0]["total_suite_ms"]) > 0
