from __future__ import annotations

from argparse import Namespace

import pytest

from benchmark.multi_turn_chat import bench_router_responses as bench


def test_validate_args_rejects_http_previous_response_without_store():
    args = Namespace(
        parallel=1,
        client_transport="http_sse",
        chain_mode="previous_response_id",
        store_mode="false",
    )

    with pytest.raises(
        ValueError,
        match="HTTP SSE with --chain-mode previous_response_id requires --store-mode true",
    ):
        bench._validate_args(args)


def test_benchmark_contract_carries_router_axes():
    args = Namespace(
        model="Qwen/Qwen2.5-72B-Instruct",
        worker_transport="grpc",
        router_topology="regular_grpc_worker",
        store_mode="true",
        chain_mode="full_replay",
    )

    contract = bench._benchmark_contract(args, "websocket")

    assert contract == {
        "benchmark_family": "long_context_multiturn_qos",
        "run_class": "router_multiturn_adapter",
        "client_transport": "websocket",
        "worker_transport": "grpc",
        "router_topology": "regular_grpc_worker",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "topology_overlay": "none",
        "store_mode": "store_true",
        "workload_kind": "multi_turn_chat_full_replay",
    }


def test_prepare_turn_input_modes():
    conversation = [bench._user_message("first"), bench._assistant_message("reply")]

    assert bench._prepare_turn_input("previous_response_id", conversation, "next") == "next"
    assert bench._prepare_turn_input("full_replay", conversation, "next") == [
        *conversation,
        bench._user_message("next"),
    ]
