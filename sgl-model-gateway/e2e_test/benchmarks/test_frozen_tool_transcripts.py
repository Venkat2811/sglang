"""Unit tests for the frozen tool transcript loader."""

from __future__ import annotations

import pytest

from benchmarks.frozen_tool_transcripts import load_frozen_tool_transcript_scenarios
from benchmarks import test_ws_microbench as ws_bench


def test_load_frozen_tool_transcript_scenarios_has_expected_shape():
    scenarios = load_frozen_tool_transcript_scenarios()

    assert len(scenarios) == 3
    assert sum(len(scenario.turns) for scenario in scenarios) == 9
    assert scenarios[0].id == "repo_refactor_path"
    assert scenarios[1].turns[0].tool_name == "read_test_report"


def test_frozen_transcript_turn_input_contains_function_call_output_and_message():
    scenario = load_frozen_tool_transcript_scenarios()[0]
    turn = scenario.turns[0]

    items = ws_bench._frozen_transcript_turn_input(turn)

    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == turn.call_id
    assert items[1]["type"] == "message"
    assert items[1]["content"][0]["text"] == turn.user_text


def test_summarize_frozen_transcript_samples_tracks_payloads_and_tokens():
    summary = ws_bench._summarize_frozen_transcript_samples(
        [
            {
                "scenario_count": 1,
                "total_turns": 2,
                "total_suite_ms": 120.0,
                "seed_setup_ms_total": 9.0,
                "total_request_payload_bytes": 500,
                "total_response_payload_bytes": 900,
                "input_tokens_total": 64,
                "output_tokens_total": 20,
                "nonempty_output_turns": 2,
                "turn_results": [
                    {
                        "request_to_first_event_ms": 5.0,
                        "request_to_first_content_ms": 15.0,
                        "request_to_completed_ms": 60.0,
                        "first_event_to_first_content_ms": 10.0,
                        "first_content_to_completed_ms": 45.0,
                        "request_payload_bytes": 200,
                        "response_payload_bytes": 400,
                        "input_tokens": 30,
                        "output_tokens": 10,
                    },
                    {
                        "request_to_first_event_ms": 6.0,
                        "request_to_first_content_ms": 16.0,
                        "request_to_completed_ms": 60.0,
                        "first_event_to_first_content_ms": 10.0,
                        "first_content_to_completed_ms": 44.0,
                        "request_payload_bytes": 300,
                        "response_payload_bytes": 500,
                        "input_tokens": 34,
                        "output_tokens": 10,
                    },
                ],
            }
        ]
    )

    assert summary["samples"] == 1
    assert summary["turns"] == 2
    assert summary["all_turns_nonempty"] == 1
    assert summary["total_request_payload_bytes_p50"] == pytest.approx(500.0)
    assert summary["request_to_first_content_ms_p50"] == pytest.approx(15.5)
    assert summary["output_tokens_total_mean"] == pytest.approx(20.0)


def test_frozen_transcript_transport_ratios_report_payload_and_latency_deltas():
    ratios = ws_bench._frozen_transcript_transport_ratios(
        {
            "total_suite_ms_p50": 100.0,
            "request_to_first_event_ms_p50": 10.0,
            "request_to_first_content_ms_p50": 20.0,
            "request_to_completed_ms_p50": 50.0,
            "total_request_payload_bytes_p50": 1000.0,
            "total_response_payload_bytes_p50": 2000.0,
        },
        {
            "total_suite_ms_p50": 90.0,
            "request_to_first_event_ms_p50": 12.0,
            "request_to_first_content_ms_p50": 18.0,
            "request_to_completed_ms_p50": 45.0,
            "total_request_payload_bytes_p50": 800.0,
            "total_response_payload_bytes_p50": 1800.0,
        },
    )

    assert ratios["ws_over_http_total_suite"] == pytest.approx(0.9)
    assert ratios["ws_vs_http_total_suite_delta_pct"] == pytest.approx(10.0)
    assert ratios["ws_over_http_first_content_p50"] == pytest.approx(0.9)
    assert ratios["ws_over_http_request_payload_total_p50"] == pytest.approx(0.8)
