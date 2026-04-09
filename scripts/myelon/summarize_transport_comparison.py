#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def fmt_num(value: float, scale: float = 1.0, decimals: int = 2, unit: str = "") -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * scale:.{decimals}f}{unit}"


def fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    sep = "  ".join("-" * width for width in widths)
    lines = [render_row(headers), sep]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def is_request_signature(signature: str | None) -> bool:
    if not signature:
        return False
    return "GenerateReqInput" in signature


def collect_gloo_request_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        if row.get("kind") != "broadcast_pyobj" or row.get("role") != "src":
            continue
        caller_file = row.get("caller_file") or ""
        if not caller_file.endswith("/sglang/srt/managers/scheduler.py"):
            continue
        if row.get("caller_func") not in {"recv_requests", "_broadcast_tp_scheduler_reqs"}:
            continue
        if not is_request_signature(row.get("payload_signature")):
            continue
        rows.append(row)
    return rows


def collect_mq_source_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        if row.get("event") != "broadcast_object" or row.get("role") != "src":
            continue
        if row.get("unique_name") != "tp:0":
            continue
        if not is_request_signature(row.get("payload_signature")):
            continue
        rows.append(row)
    return rows


def collect_mq_writer_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        if row.get("kind") != "myelon_ipc_event":
            continue
        if row.get("queue_role") != "writer" or row.get("event") != "enqueue":
            continue
        if not is_request_signature(row.get("payload_signature")):
            continue
        rows.append(row)
    return rows


def collect_mq_reader_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        if row.get("kind") != "myelon_ipc_event":
            continue
        if row.get("event") != "dequeue":
            continue
        if not is_request_signature(row.get("payload_signature")):
            continue
        rows.append(row)
    return rows


def drop_warmup_events(rows: list[dict[str, Any]], warmup_repeats: int) -> list[dict[str, Any]]:
    if warmup_repeats <= 0 or len(rows) <= warmup_repeats:
        return rows
    rows = sorted(rows, key=lambda row: row.get("trace_time_ns") or 0)
    return rows[warmup_repeats:]


def summarize_path(label: str, artifact: Path) -> dict[str, Any]:
    bench = load_json(artifact / "bench_repeat_summary.json")
    pyobj_path = artifact / "pyobj_broadcast.jsonl"
    parallel_path = artifact / "parallel_state.jsonl"
    ipc_events_path = artifact / "ipc_events.jsonl"
    warmup_repeats = int(bench.get("warmup_repeats") or 0)

    if label == "Gloo TCP":
        request_events = drop_warmup_events(
            collect_gloo_request_events(pyobj_path), warmup_repeats
        )
        serialize_values = [
            (row.get("pickle_ns") or 0) + (row.get("tensorize_ns") or 0)
            for row in request_events
        ]
        transport_values = [
            (row.get("size_broadcast_ns") or 0) + (row.get("payload_broadcast_ns") or 0)
            for row in request_events
        ]
        wait_values: list[int] = []
        source_values = [row.get("elapsed_ns") or 0 for row in request_events]
        payload_values = [row.get("payload_bytes") or 0 for row in request_events]
        phase_sum = sum(serialize_values) + sum(transport_values)
        phase_mix = {
            "serde_pct": (100.0 * sum(serialize_values) / phase_sum) if phase_sum else 0.0,
            "transport_pct": (100.0 * sum(transport_values) / phase_sum) if phase_sum else 0.0,
            "wait_pct": 0.0,
        }
        reader_events: list[dict[str, Any]] = []
    else:
        request_events = drop_warmup_events(
            collect_mq_source_events(parallel_path), warmup_repeats
        )
        writer_events = drop_warmup_events(
            collect_mq_writer_events(ipc_events_path), warmup_repeats
        )
        serialize_values = [row.get("pickle_ns") or 0 for row in writer_events]
        transport_values = [row.get("transport_ns") or 0 for row in writer_events]
        wait_values = [row.get("wait_ns") or 0 for row in writer_events]
        source_values = [row.get("elapsed_ns") or 0 for row in request_events]
        payload_values = [row.get("payload_bytes") or 0 for row in request_events]
        phase_sum = sum(serialize_values) + sum(transport_values) + sum(wait_values)
        phase_mix = {
            "serde_pct": (100.0 * sum(serialize_values) / phase_sum) if phase_sum else 0.0,
            "transport_pct": (100.0 * sum(transport_values) / phase_sum) if phase_sum else 0.0,
            "wait_pct": (100.0 * sum(wait_values) / phase_sum) if phase_sum else 0.0,
        }
        reader_events = collect_mq_reader_events(ipc_events_path)
        if reader_events and request_events:
            readers_per_request = max(1, len(reader_events) // (len(request_events) + warmup_repeats))
            reader_events = sorted(reader_events, key=lambda row: row.get("trace_time_ns") or 0)[
                warmup_repeats * readers_per_request :
            ]

    wall_ns = bench["total_wall_s"] * 1_000_000_000.0
    source_sum_ns = float(sum(source_values))
    source_pct_wall = 100.0 * source_sum_ns / wall_ns if wall_ns else float("nan")

    reader_transport_values = [row.get("transport_ns") or 0 for row in reader_events]
    reader_unpickle_values = [row.get("unpickle_ns") or 0 for row in reader_events]

    return {
        "label": label,
        "artifact": str(artifact),
        "bench": bench,
        "request_events": len(request_events),
        "payload_bytes": stats(payload_values),
        "source_ns": stats(source_values),
        "source_pct_wall": source_pct_wall,
        "serde_ns": stats(serialize_values),
        "transport_ns": stats(transport_values),
        "wait_ns": stats(wait_values),
        "phase_mix": phase_mix,
        "reader_transport_ns": stats(reader_transport_values),
        "reader_unpickle_ns": stats(reader_unpickle_values),
    }


def build_report(paths: list[dict[str, Any]]) -> str:
    end_to_end_rows = []
    source_rows = []
    phase_rows = []
    reader_rows = []
    for path in paths:
        bench = path["bench"]
        end_to_end_rows.append(
            [
                path["label"],
                fmt_int(bench["repeats"]),
                fmt_int(bench["completed_requests"]),
                fmt_int(bench["completed_responses"]),
                fmt_num(bench["total_wall_s"], 1.0, 2, "s"),
                fmt_num(bench["latency_s"]["p50"], 1000, 1, "ms"),
                fmt_num(bench["latency_s"]["p95"], 1000, 1, "ms"),
                fmt_num(bench["ttft_s"]["p50"], 1000, 1, "ms"),
                fmt_num(bench["ttft_s"]["p95"], 1000, 1, "ms"),
            ]
        )
        source_rows.append(
            [
                path["label"],
                fmt_int(path["request_events"]),
                (
                    f"{fmt_num(path['payload_bytes'].get('p50', float('nan')), 1, 0)} / "
                    f"{fmt_num(path['payload_bytes'].get('p95', float('nan')), 1, 0)} / "
                    f"{fmt_num(path['payload_bytes'].get('max', float('nan')), 1, 0)}"
                ),
                (
                    f"{fmt_num(path['source_ns'].get('p50', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['source_ns'].get('p95', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['source_ns'].get('max', float('nan')), 1/1000, 1, 'us')}"
                ),
                fmt_num(path["source_pct_wall"], 1.0, 2, "%"),
            ]
        )
        phase_rows.append(
            [
                path["label"],
                fmt_int(path["serde_ns"].get("count", 0)),
                (
                    f"{fmt_num(path['serde_ns'].get('p50', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['serde_ns'].get('p95', float('nan')), 1/1000, 1, 'us')}"
                ),
                (
                    f"{fmt_num(path['transport_ns'].get('p50', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['transport_ns'].get('p95', float('nan')), 1/1000, 1, 'us')}"
                ),
                (
                    f"{fmt_num(path['wait_ns'].get('p50', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['wait_ns'].get('p95', float('nan')), 1/1000, 1, 'us')}"
                    if path["wait_ns"]
                    else "n/a"
                ),
                (
                    f"{fmt_num(path['source_ns'].get('p50', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['source_ns'].get('p95', float('nan')), 1/1000, 1, 'us')}"
                ),
                (
                    f"serde {path['phase_mix']['serde_pct']:.1f}% | "
                    f"transport {path['phase_mix']['transport_pct']:.1f}%"
                    + (
                        f" | wait {path['phase_mix']['wait_pct']:.1f}%"
                        if path["phase_mix"]["wait_pct"] > 0
                        else ""
                    )
                ),
            ]
        )
        reader_rows.append(
            [
                path["label"],
                fmt_int(path["reader_transport_ns"].get("count", 0)),
                (
                    f"{fmt_num(path['reader_transport_ns'].get('p50', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['reader_transport_ns'].get('p95', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['reader_transport_ns'].get('max', float('nan')), 1/1000, 1, 'us')}"
                    if path["reader_transport_ns"]
                    else "n/a"
                ),
                (
                    f"{fmt_num(path['reader_unpickle_ns'].get('p50', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['reader_unpickle_ns'].get('p95', float('nan')), 1/1000, 1, 'us')} / "
                    f"{fmt_num(path['reader_unpickle_ns'].get('max', float('nan')), 1/1000, 1, 'us')}"
                    if path["reader_unpickle_ns"]
                    else "n/a"
                ),
            ]
        )

    lines = [
        "# TP=8 Actual-Inference Transport Comparison",
        "",
        "## End-to-end benchmark",
        "",
        "```text",
        render_table(
            [
                "Path",
                "Batches",
                "Requests",
                "Responses",
                "Wall",
                "Lat P50",
                "Lat P95",
                "TTFT P50",
                "TTFT P95",
            ],
            end_to_end_rows,
        ),
        "```",
        "",
        "## Request fanout source path",
        "",
        "```text",
        render_table(
            [
                "Path",
                "Req Msg n",
                "Payload B P50/P95/Max",
                "Src Total P50/P95/Max",
                "Src % Wall",
            ],
            source_rows,
        ),
        "```",
        "",
        "## Serde and transport breakdown",
        "",
        "```text",
        render_table(
            [
                "Path",
                "Samples",
                "Serde P50/P95",
                "Transport P50/P95",
                "Wait P50/P95",
                "Src Total P50/P95",
                "Phase Mix",
            ],
            phase_rows,
        ),
        "```",
        "",
        "## Reader-side receive path",
        "",
        "```text",
        render_table(
            [
                "Path",
                "Samples",
                "Recv P50/P95/Max",
                "Unpickle P50/P95/Max",
            ],
            reader_rows,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gloo-artifact", type=Path, required=True)
    parser.add_argument("--mq-inline-artifact", type=Path, required=True)
    parser.add_argument("--mq-overflow-artifact", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    summaries = [
        summarize_path("Gloo TCP", args.gloo_artifact),
        summarize_path("MQ SHM Inline", args.mq_inline_artifact),
        summarize_path("MQ ZMQ Overflow", args.mq_overflow_artifact),
    ]
    args.output_md.write_text(build_report(summaries), encoding="utf-8")
    args.output_json.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
