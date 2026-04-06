"""Run the retained B200 WS benchmark matrix and persist structured artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks import test_ws_microbench as ws_bench
from infra.constants import ConnectionMode
from infra.gateway import Gateway
from infra.model_pool import ModelPool
from infra.model_specs import get_model_spec

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_SGLANG_ROOT = Path(__file__).resolve().parents[3]
_AI_CHAT_EXPORTS_ROOT = _WORKSPACE_ROOT / "ai-chat-exports"
_DEFAULT_ARTIFACTS_ROOT = (
    _AI_CHAT_EXPORTS_ROOT
    / ".0_agentic_engineering"
    / "1_sglang"
    / "0_sgl-router-ws"
    / "2_artifacts"
)

_ALL_FAMILIES = (
    "transport_compare",
    "continuation_compare",
    "tool_output_compare",
    "frozen_transcript_compare",
    "router_multiturn_full_replay",
    "router_multiturn_previous_response_id",
)


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    logs: Path
    raw_benchmarks: Path
    raw_system_info: Path
    reports_benchmarks: Path
    reports_system_info: Path
    tools: Path


@dataclass(frozen=True)
class BackendConfig:
    name: str
    mode: ConnectionMode

    @property
    def worker_transport(self) -> str:
        return ws_bench._worker_transport_for_backend(self.name)

    @property
    def router_topology(self) -> str:
        return ws_bench._router_topology_for_backend(self.name)


def _timestamp_slug() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())


def _default_artifact_root(model_id: str) -> Path:
    model_label = model_id.replace("-", "").replace("_", "")
    return _DEFAULT_ARTIFACTS_ROOT / (
        f"{_timestamp_slug()}_b200_{model_label}_ws_matrix_artifacts"
    )


def _artifact_paths(root: Path) -> ArtifactPaths:
    paths = ArtifactPaths(
        root=root,
        logs=root / "runtime" / "logs",
        raw_benchmarks=root / "runtime" / "raw" / "benchmarks",
        raw_system_info=root / "runtime" / "raw" / "system_info",
        reports_benchmarks=root / "runtime" / "reports" / "benchmarks",
        reports_system_info=root / "runtime" / "reports" / "system_info",
        tools=root / "tools",
    )
    for path in (
        paths.root,
        paths.logs,
        paths.raw_benchmarks,
        paths.raw_system_info,
        paths.reports_benchmarks,
        paths.reports_system_info,
        paths.tools,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _configure_logging(log_path: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _run_command(path: Path, cmd: list[str], cwd: Path | None = None) -> str:
    logger.info("Capturing command: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
    content = textwrap.dedent(
        f"""\
        $ {' '.join(cmd)}
        [exit_code] {completed.returncode}

        [stdout]
        {completed.stdout}

        [stderr]
        {completed.stderr}
        """
    )
    _write_text(path, content)
    return completed.stdout


def _capture_system_info(paths: ArtifactPaths) -> dict[str, str]:
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "sglang_git_rev": _run_command(
            paths.raw_system_info / "sglang_git_rev.txt",
            ["git", "-C", str(_SGLANG_ROOT), "rev-parse", "HEAD"],
        ).strip(),
        "ai_chat_exports_git_rev": _run_command(
            paths.raw_system_info / "ai_chat_exports_git_rev.txt",
            ["git", "-C", str(_AI_CHAT_EXPORTS_ROOT), "rev-parse", "HEAD"],
        ).strip(),
    }

    _run_command(paths.raw_system_info / "uname.txt", ["uname", "-a"])
    _run_command(paths.raw_system_info / "python_version.txt", [sys.executable, "--version"])
    _run_command(paths.raw_system_info / "uv_version.txt", ["uv", "--version"])
    _run_command(paths.raw_system_info / "os_release.txt", ["bash", "-lc", "cat /etc/os-release"])
    _run_command(paths.raw_system_info / "nvidia_smi.txt", ["nvidia-smi"])
    _run_command(paths.raw_system_info / "nvidia_topology.txt", ["nvidia-smi", "topo", "-m"])
    _run_command(paths.raw_system_info / "memory.txt", ["free", "-h"])
    _run_command(paths.raw_system_info / "disk.txt", ["bash", "-lc", "df -h /root /root/.cache/huggingface"])
    _run_command(
        paths.raw_system_info / "hf_cache_sizes.txt",
        [
            "bash",
            "-lc",
            "du -sh /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct "
            "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct 2>/dev/null",
        ],
    )
    _run_command(paths.raw_system_info / "env.txt", ["bash", "-lc", "env | sort"])

    report = textwrap.dedent(
        f"""\
        Timestamp UTC: {summary['timestamp_utc']}
        Python: {summary['python_executable']}
        sglang git rev: {summary['sglang_git_rev']}
        ai-chat-exports git rev: {summary['ai_chat_exports_git_rev']}
        """
    )
    _write_text(paths.reports_system_info / "README.md", report)
    return summary


def _run_logged_samples(
    *,
    suite_name: str,
    transport: str,
    samples: int,
    runner: Any,
) -> list[Any]:
    results = []
    for sample_index in range(1, samples + 1):
        logger.info(
            "Running sample suite=%s transport=%s sample=%s/%s",
            suite_name,
            transport,
            sample_index,
            samples,
        )
        started_at = time.perf_counter()
        results.append(runner())
        logger.info(
            "Completed sample suite=%s transport=%s sample=%s/%s elapsed=%.2fs",
            suite_name,
            transport,
            sample_index,
            samples,
            time.perf_counter() - started_at,
        )
    return results


def _transport_compare_payload(
    backend_name: str,
    model_path: str,
    gateway: Gateway,
    samples: int,
    request_timeout: int,
) -> dict[str, Any]:
    benchmark_context = ws_bench._benchmark_context(
        benchmark_family="transport_qos",
        run_class="b200_transport_compare",
        backend_name=backend_name,
        model=model_path,
        store_mode="store_false",
        workload_kind="single_turn_text",
    )
    http_samples = _run_logged_samples(
        suite_name="transport_compare",
        transport="http_sse",
        samples=samples,
        runner=lambda: ws_bench._run_single_http_sample(
            gateway.base_url,
            model_path,
            request_timeout,
        ),
    )
    ws_samples = _run_logged_samples(
        suite_name="transport_compare",
        transport="websocket",
        samples=samples,
        runner=lambda: asyncio.run(
            ws_bench._run_single_ws_sample(
                ws_bench._gateway_ws_url(gateway.base_url), model_path
            )
        ),
    )
    http_summary = ws_bench._summarize_samples(http_samples)
    ws_summary = ws_bench._summarize_samples(ws_samples)
    return {
        "benchmark_context": benchmark_context,
        "worker_backend": backend_name,
        "router_url": gateway.base_url,
        "model": model_path,
        "samples": samples,
        "http": ws_bench._transport_result(
            context=benchmark_context,
            client_transport="http_sse",
            samples=http_samples,
            summary=http_summary,
        ),
        "websocket": ws_bench._transport_result(
            context=benchmark_context,
            client_transport="websocket",
            samples=ws_samples,
            summary=ws_summary,
        ),
        "ratios": ws_bench._transport_ratios(http_summary, ws_summary),
    }


def _continuation_compare_payload(
    backend_name: str,
    model_path: str,
    gateway: Gateway,
    turns: int,
    samples: int,
    request_timeout: int,
) -> dict[str, Any]:
    benchmark_context = ws_bench._benchmark_context(
        benchmark_family="continuation_qos",
        run_class="b200_continuation_compare",
        backend_name=backend_name,
        model=model_path,
        store_mode="store_true",
        workload_kind="incremental_text_continuation",
    )
    http_samples = _run_logged_samples(
        suite_name="continuation_compare",
        transport="http_sse",
        samples=samples,
        runner=lambda: ws_bench._run_http_continuation_chain_sample(
            gateway.base_url,
            model_path,
            turns,
            request_timeout,
        ),
    )
    ws_samples = _run_logged_samples(
        suite_name="continuation_compare",
        transport="websocket",
        samples=samples,
        runner=lambda: asyncio.run(
            ws_bench._run_ws_continuation_chain_sample(
                ws_bench._gateway_ws_url(gateway.base_url), model_path, turns
            )
        ),
    )
    http_summary = ws_bench._summarize_chain_samples(http_samples)
    ws_summary = ws_bench._summarize_chain_samples(ws_samples)
    return {
        "benchmark_context": benchmark_context,
        "worker_backend": backend_name,
        "router_url": gateway.base_url,
        "model": model_path,
        "turns": turns,
        "samples": samples,
        "http": ws_bench._transport_result(
            context=benchmark_context,
            client_transport="http_sse",
            samples=http_samples,
            summary=http_summary,
        ),
        "websocket": ws_bench._transport_result(
            context=benchmark_context,
            client_transport="websocket",
            samples=ws_samples,
            summary=ws_summary,
        ),
        "ratios": ws_bench._chain_transport_ratios(http_summary, ws_summary),
    }


def _tool_output_compare_payload(
    backend_name: str,
    model_path: str,
    gateway: Gateway,
    tool_turns: int,
    samples: int,
    request_timeout: int,
) -> dict[str, Any]:
    benchmark_context = ws_bench._benchmark_context(
        benchmark_family="continuation_qos",
        run_class="b200_tool_output_compare",
        backend_name=backend_name,
        model=model_path,
        store_mode="store_true",
        workload_kind="incremental_tool_output_continuation",
    )
    http_samples = _run_logged_samples(
        suite_name="tool_output_compare",
        transport="http_sse",
        samples=samples,
        runner=lambda: ws_bench._run_http_tool_output_chain_sample(
            gateway.base_url,
            model_path,
            tool_turns,
            request_timeout,
        ),
    )
    ws_samples = _run_logged_samples(
        suite_name="tool_output_compare",
        transport="websocket",
        samples=samples,
        runner=lambda: asyncio.run(
            ws_bench._run_ws_tool_output_chain_sample(
                ws_bench._gateway_ws_url(gateway.base_url), model_path, tool_turns
            )
        ),
    )
    http_summary = ws_bench._summarize_chain_samples(http_samples)
    ws_summary = ws_bench._summarize_chain_samples(ws_samples)
    return {
        "benchmark_context": benchmark_context,
        "worker_backend": backend_name,
        "router_url": gateway.base_url,
        "model": model_path,
        "tool_turns": tool_turns,
        "samples": samples,
        "http": ws_bench._transport_result(
            context=benchmark_context,
            client_transport="http_sse",
            samples=http_samples,
            summary=http_summary,
        ),
        "websocket": ws_bench._transport_result(
            context=benchmark_context,
            client_transport="websocket",
            samples=ws_samples,
            summary=ws_summary,
        ),
        "ratios": ws_bench._chain_transport_ratios(http_summary, ws_summary),
    }


def _frozen_transcript_compare_payload(
    backend_name: str,
    model_path: str,
    gateway: Gateway,
    samples: int,
    request_timeout: int,
) -> dict[str, Any]:
    scenarios = ws_bench._selected_frozen_tool_transcript_scenarios()
    benchmark_context = ws_bench._benchmark_context(
        benchmark_family="agentic_transcript_qos",
        run_class="b200_frozen_tool_transcript_compare",
        backend_name=backend_name,
        model=model_path,
        store_mode="store_true",
        workload_kind="incremental_frozen_tool_transcript",
    )
    http_samples = _run_logged_samples(
        suite_name="frozen_transcript_compare",
        transport="http_sse",
        samples=samples,
        runner=lambda: ws_bench._run_http_frozen_tool_transcript_sample(
            gateway.base_url,
            model_path,
            scenarios,
            request_timeout,
        ),
    )
    ws_samples = _run_logged_samples(
        suite_name="frozen_transcript_compare",
        transport="websocket",
        samples=samples,
        runner=lambda: asyncio.run(
            ws_bench._run_ws_frozen_tool_transcript_sample(
                ws_bench._gateway_ws_url(gateway.base_url),
                model_path,
                scenarios,
                request_timeout,
            )
        ),
    )
    http_summary = ws_bench._summarize_frozen_transcript_samples(http_samples)
    ws_summary = ws_bench._summarize_frozen_transcript_samples(ws_samples)
    return {
        "benchmark_context": benchmark_context,
        "worker_backend": backend_name,
        "router_url": gateway.base_url,
        "model": model_path,
        "samples": samples,
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
        "http": ws_bench._transport_result(
            context=benchmark_context,
            client_transport="http_sse",
            samples=http_samples,
            summary=http_summary,
        ),
        "websocket": ws_bench._transport_result(
            context=benchmark_context,
            client_transport="websocket",
            samples=ws_samples,
            summary=ws_summary,
        ),
        "ratios": ws_bench._frozen_transcript_transport_ratios(
            http_summary, ws_summary
        ),
    }


def _router_multiturn_payload(
    *,
    backend: BackendConfig,
    paths: ArtifactPaths,
    gateway: Gateway,
    model_path: str,
    chain_mode: str,
    store_mode: str,
    turns: int,
    num_qa: int,
    parallel: int,
) -> dict[str, Any]:
    raw_dir = paths.raw_benchmarks / backend.router_topology
    log_dir = paths.logs / backend.router_topology
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = raw_dir / f"router_multiturn_{chain_mode}.summary.json"
    result_path = raw_dir / f"router_multiturn_{chain_mode}.results.jsonl"
    log_path = log_dir / f"router_multiturn_{chain_mode}.log"
    cmd = [
        sys.executable,
        "bench_router_responses.py",
        "--base-url",
        gateway.base_url,
        "--model",
        model_path,
        "--tokenizer",
        model_path,
        "--client-transport",
        "both",
        "--chain-mode",
        chain_mode,
        "--store-mode",
        store_mode,
        "--worker-transport",
        backend.worker_transport,
        "--router-topology",
        backend.router_topology,
        "--parallel",
        str(parallel),
        "--turns",
        str(turns),
        "--num-qa",
        str(num_qa),
        "--min-len-q",
        "256",
        "--max-len-q",
        "512",
        "--min-len-a",
        "32",
        "--max-len-a",
        "96",
        "--summary-file",
        str(summary_path),
        "--result-file",
        str(result_path),
    ]
    logger.info(
        "Running router multiturn adapter backend=%s chain_mode=%s turns=%s num_qa=%s",
        backend.name,
        chain_mode,
        turns,
        num_qa,
    )
    completed = subprocess.run(
        cmd,
        cwd=_SGLANG_ROOT / "benchmark" / "multi_turn_chat",
        check=False,
        capture_output=True,
        text=True,
    )
    _write_text(
        log_path,
        textwrap.dedent(
            f"""\
            $ {' '.join(cmd)}
            [exit_code] {completed.returncode}

            [stdout]
            {completed.stdout}

            [stderr]
            {completed.stderr}
            """
        ),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"router multiturn adapter failed for backend={backend.name} chain_mode={chain_mode}"
        )
    return json.loads(summary_path.read_text())


def _report_markdown(payload: dict[str, Any]) -> str:
    sections = ["# WS Matrix Summary", ""]
    for backend_name, backend_payload in payload["backends"].items():
        sections.append(f"## {backend_name}")
        startup_phases = backend_payload.get("startup_phases")
        if startup_phases:
            sections.append(
                f"- worker acquire elapsed s: {startup_phases['worker_acquire_elapsed_s']:.2f}"
            )
            sections.append(
                f"- gateway start elapsed s: {startup_phases['gateway_start_elapsed_s']:.2f}"
            )
            sections.append(
                f"- backend setup elapsed s: {startup_phases['backend_setup_elapsed_s']:.2f}"
            )
        transport_payload = backend_payload.get("transport_compare", {})
        if "ratios" in transport_payload:
            transport = transport_payload["ratios"]
            sections.append(
                f"- transport ws/http completed p50 ratio: {transport['ws_over_http_completed_p50']:.4f}"
            )
        elif "error" in transport_payload:
            sections.append(
                f"- transport compare failed: {transport_payload['error']['message']}"
            )

        continuation_payload = backend_payload.get("continuation_compare", {})
        if "ratios" in continuation_payload:
            continuation = continuation_payload["ratios"]
            sections.append(
                f"- continuation ws/http total-chain delta pct: {continuation['ws_vs_http_total_chain_delta_pct']:.2f}"
            )
        elif "error" in continuation_payload:
            sections.append(
                f"- continuation compare failed: {continuation_payload['error']['message']}"
            )

        tool_output_payload = backend_payload.get("tool_output_compare", {})
        if "ratios" in tool_output_payload:
            tool_output = tool_output_payload["ratios"]
            sections.append(
                f"- tool-output ws/http continuation delta pct: {tool_output['ws_vs_http_total_chain_delta_pct']:.2f}"
            )
        elif "error" in tool_output_payload:
            sections.append(
                f"- tool-output compare failed: {tool_output_payload['error']['message']}"
            )

        frozen_transcript_payload = backend_payload.get("frozen_transcript_compare", {})
        if "ratios" in frozen_transcript_payload:
            frozen_transcript = frozen_transcript_payload["ratios"]
            sections.append(
                "- frozen transcript ws/http total-suite delta pct: "
                f"{frozen_transcript['ws_vs_http_total_suite_delta_pct']:.2f}"
            )
        elif "error" in frozen_transcript_payload:
            sections.append(
                "frozen transcript compare failed: "
                f"{frozen_transcript_payload['error']['message']}"
            )

        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def _root_readme(payload: dict[str, Any]) -> str:
    artifact_root = payload["artifact_root"]
    system = payload["system_summary"]
    startup_sections: list[str] = []
    for backend_name, backend_payload in payload["backends"].items():
        startup_phases = backend_payload.get("startup_phases")
        if not startup_phases:
            continue
        startup_sections.append(
            textwrap.dedent(
                f"""\
                ### {backend_name}

                - worker acquire elapsed s: {startup_phases['worker_acquire_elapsed_s']:.2f}
                - gateway start elapsed s: {startup_phases['gateway_start_elapsed_s']:.2f}
                - backend setup elapsed s: {startup_phases['backend_setup_elapsed_s']:.2f}
                """
            ).rstrip()
        )
    sections = [
        "# WS Matrix Artifacts",
        "",
        "This artifact bundle captures the retained WebSocket benchmark matrix",
        f"for `{payload['model_id']}`.",
        "",
        "## Machine Snapshot",
        "",
        f"- timestamp UTC: {system['timestamp_utc']}",
        f"- python: `{system['python_executable']}`",
        f"- sglang git rev: `{system['sglang_git_rev']}`",
        f"- ai-chat-exports git rev: `{system['ai_chat_exports_git_rev']}`",
        "",
        "## Artifact Layout",
        "",
        "- `runtime/raw/benchmarks`: raw JSON benchmark payloads",
        "- `runtime/raw/system_info`: captured machine and repo state",
        "- `runtime/reports/benchmarks`: matrix summaries",
        "- `runtime/reports/system_info`: condensed system report",
        "- `runtime/logs`: runner and subprocess logs",
        "- `tools`: commands and manifests for reruns",
    ]
    if startup_sections:
        sections.extend(["", "## Startup Phases", ""])
        for entry in startup_sections:
            sections.append(entry)
            sections.append("")
    sections.extend(
        [
            "## Primary Report",
            "",
            "- `runtime/reports/benchmarks/matrix_summary.md`",
            "- `runtime/reports/benchmarks/matrix_summary.json`",
            "",
            f"Artifact root: `{artifact_root}`",
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def _run_backend(
    *,
    backend: BackendConfig,
    paths: ArtifactPaths,
    pool: ModelPool,
    model_id: str,
    startup_timeout: int,
    router_timeout: int,
    transport_samples: int,
    chain_turns: int,
    chain_samples: int,
    tool_turns: int,
    transcript_samples: int,
    multiturn_turns: int,
    multiturn_num_qa: int,
    multiturn_parallel: int,
    request_timeout: int,
    families: set[str],
) -> dict[str, Any]:
    logger.info("Starting backend slice: %s", backend.name)
    backend_started_at = time.perf_counter()
    logger.info(
        "Acquiring worker backend=%s model_id=%s mode=%s",
        backend.name,
        model_id,
        backend.mode.value,
    )
    worker_acquire_started_at = time.perf_counter()
    instance = pool.get(model_id, backend.mode, gpu_wait_timeout=startup_timeout)
    worker_acquire_elapsed_s = time.perf_counter() - worker_acquire_started_at
    logger.info(
        "Worker acquired backend=%s worker_url=%s elapsed=%.2fs",
        backend.name,
        instance.worker_url,
        worker_acquire_elapsed_s,
    )
    try:
        gateway = Gateway()
        backend_payload: dict[str, Any] = {
            "model_id": model_id,
            "model_path": instance.model_path,
            "worker_transport": backend.worker_transport,
            "router_topology": backend.router_topology,
        }
        try:
            gateway_start_started_at = time.perf_counter()
            gateway.start(
                worker_urls=[instance.worker_url],
                model_path=instance.model_path,
                timeout=router_timeout,
                extra_args=["--history-backend", "memory"],
            )
            gateway_start_elapsed_s = time.perf_counter() - gateway_start_started_at
            backend_payload["router_url"] = gateway.base_url
            backend_payload["startup_phases"] = {
                "worker_acquire_elapsed_s": worker_acquire_elapsed_s,
                "gateway_start_elapsed_s": gateway_start_elapsed_s,
                "backend_setup_elapsed_s": time.perf_counter() - backend_started_at,
            }
            logger.info(
                "Gateway ready backend=%s router_url=%s elapsed=%.2fs setup_total=%.2fs",
                backend.name,
                gateway.base_url,
                gateway_start_elapsed_s,
                backend_payload["startup_phases"]["backend_setup_elapsed_s"],
            )

            raw_path = paths.raw_benchmarks / f"{backend.router_topology}.json"

            def persist_partial() -> None:
                _write_json(raw_path, backend_payload)

            suite_runners: list[tuple[str, Any]] = [
                (
                    "transport_compare",
                    lambda: _transport_compare_payload(
                        backend.name,
                        instance.model_path,
                        gateway,
                        transport_samples,
                        request_timeout,
                    ),
                ),
                (
                    "continuation_compare",
                    lambda: _continuation_compare_payload(
                        backend.name,
                        instance.model_path,
                        gateway,
                        chain_turns,
                        chain_samples,
                        request_timeout,
                    ),
                ),
                (
                    "tool_output_compare",
                    lambda: _tool_output_compare_payload(
                        backend.name,
                        instance.model_path,
                        gateway,
                        tool_turns,
                        chain_samples,
                        request_timeout,
                    ),
                ),
                (
                    "frozen_transcript_compare",
                    lambda: _frozen_transcript_compare_payload(
                        backend.name,
                        instance.model_path,
                        gateway,
                        transcript_samples,
                        request_timeout,
                    ),
                ),
                (
                    "router_multiturn_full_replay",
                    lambda: _router_multiturn_payload(
                        backend=backend,
                        paths=paths,
                        gateway=gateway,
                        model_path=instance.model_path,
                        chain_mode="full_replay",
                        store_mode="true",
                        turns=multiturn_turns,
                        num_qa=multiturn_num_qa,
                        parallel=multiturn_parallel,
                    ),
                ),
                (
                    "router_multiturn_previous_response_id",
                    lambda: _router_multiturn_payload(
                        backend=backend,
                        paths=paths,
                        gateway=gateway,
                        model_path=instance.model_path,
                        chain_mode="previous_response_id",
                        store_mode="true",
                        turns=multiturn_turns,
                        num_qa=multiturn_num_qa,
                        parallel=multiturn_parallel,
                    ),
                ),
            ]

            persist_partial()
            for suite_name, runner in suite_runners:
                if suite_name not in families:
                    continue
                logger.info("Running suite backend=%s suite=%s", backend.name, suite_name)
                try:
                    backend_payload[suite_name] = runner()
                except Exception as exc:
                    logger.exception(
                        "Suite failed backend=%s suite=%s",
                        backend.name,
                        suite_name,
                    )
                    backend_payload[suite_name] = {
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    }
                persist_partial()
        finally:
            gateway.shutdown()
    finally:
        instance.release()

    return backend_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the retained B200 WS benchmark matrix with artifact capture."
    )
    parser.add_argument("--model-id", default="qwen-72b")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Artifact directory root. Defaults to a timestamped ai-chat-exports path.",
    )
    parser.add_argument("--transport-samples", type=int, default=3)
    parser.add_argument("--chain-turns", type=int, default=20)
    parser.add_argument("--chain-samples", type=int, default=1)
    parser.add_argument("--tool-turns", type=int, default=20)
    parser.add_argument("--transcript-samples", type=int, default=1)
    parser.add_argument("--multiturn-turns", type=int, default=8)
    parser.add_argument("--multiturn-num-qa", type=int, default=8)
    parser.add_argument("--multiturn-parallel", type=int, default=1)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--router-timeout", type=int, default=180)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["http", "grpc"],
        default=["http", "grpc"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(_ALL_FAMILIES),
        default=list(_ALL_FAMILIES),
        help="Benchmark families to run. Use a subset for cheaper incremental passes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root or _default_artifact_root(args.model_id)
    paths = _artifact_paths(artifact_root)
    _configure_logging(paths.logs / "runner.log")
    _write_text(
        paths.tools / "invocation.txt",
        "$ " + " ".join(sys.argv) + "\n",
    )

    logger.info("Artifacts will be written to %s", artifact_root)
    system_summary = _capture_system_info(paths)
    model_spec = get_model_spec(args.model_id)
    backend_configs = [
        BackendConfig(name=name, mode=ConnectionMode(name)) for name in args.backends
    ]

    payload: dict[str, Any] = {
        "artifact_root": str(artifact_root),
        "model_id": args.model_id,
        "model_path": model_spec["model"],
        "system_summary": system_summary,
        "runner_config": {
            "transport_samples": args.transport_samples,
            "chain_turns": args.chain_turns,
            "chain_samples": args.chain_samples,
            "tool_turns": args.tool_turns,
            "transcript_samples": args.transcript_samples,
            "multiturn_turns": args.multiturn_turns,
            "multiturn_num_qa": args.multiturn_num_qa,
            "multiturn_parallel": args.multiturn_parallel,
            "startup_timeout": args.startup_timeout,
            "router_timeout": args.router_timeout,
            "request_timeout": args.request_timeout,
            "backends": [backend.name for backend in backend_configs],
            "families": args.families,
        },
        "backends": {},
    }

    with ModelPool() as pool:
        # No eager workers here, but this sets the health-check timeout used by
        # later on-demand launches from pool.get().
        pool.startup(requirements=[], startup_timeout=args.startup_timeout)
        for backend in backend_configs:
            payload["backends"][backend.name] = _run_backend(
                backend=backend,
                paths=paths,
                pool=pool,
                model_id=args.model_id,
                startup_timeout=args.startup_timeout,
                router_timeout=args.router_timeout,
                transport_samples=args.transport_samples,
                chain_turns=args.chain_turns,
                chain_samples=args.chain_samples,
                tool_turns=args.tool_turns,
                transcript_samples=args.transcript_samples,
                multiturn_turns=args.multiturn_turns,
                multiturn_num_qa=args.multiturn_num_qa,
                multiturn_parallel=args.multiturn_parallel,
                request_timeout=args.request_timeout,
                families=set(args.families),
            )

    _write_json(paths.reports_benchmarks / "matrix_summary.json", payload)
    _write_text(paths.reports_benchmarks / "matrix_summary.md", _report_markdown(payload))
    _write_text(paths.root / "README.md", _root_readme(payload))
    logger.info("Completed B200 WS matrix run")
    print(json.dumps({"artifact_root": str(artifact_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
