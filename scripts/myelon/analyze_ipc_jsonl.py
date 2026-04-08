#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_records(path: Path, tag: str | None) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("kind") != "myelon_ipc_stats":
            continue
        if tag and record.get("tag") != tag:
            continue
        records.append(record)
    return records


def aggregate(records: list[dict]) -> dict:
    writers = [r for r in records if r.get("queue_config", {}).get("role") == "writer"]
    readers = [r for r in records if r.get("queue_config", {}).get("role") != "writer"]

    payload_types: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "max_bytes": 0, "overflow_count": 0}
    )
    for record in writers:
        for row in record.get("payload_types", []):
            dst = payload_types[row["type"]]
            dst["count"] += int(row["count"])
            dst["max_bytes"] = max(dst["max_bytes"], int(row["max_bytes"]))
            dst["overflow_count"] += int(row["overflow_count"])

    top_payload_types = sorted(
        (
            {"type": key, **value}
            for key, value in payload_types.items()
        ),
        key=lambda row: (
            row["overflow_count"],
            row["max_bytes"],
            row["count"],
            row["type"],
        ),
        reverse=True,
    )[:10]

    writer_enqueue = {
        "count": sum(r["enqueue"]["count"] for r in writers),
        "inline_count": sum(r["enqueue"]["inline_count"] for r in writers),
        "overflow_count": sum(r["enqueue"]["overflow_count"] for r in writers),
        "remote_send_count": sum(r["enqueue"]["remote_send_count"] for r in writers),
        "payload_avg_bytes_weighted": 0,
        "payload_max_bytes": max(
            (r["enqueue"]["payload_max_bytes"] for r in writers),
            default=0,
        ),
    }
    total_enqueue_bytes = sum(
        r["enqueue"]["payload_avg_bytes"] * r["enqueue"]["count"] for r in writers
    )
    if writer_enqueue["count"] > 0:
        writer_enqueue["payload_avg_bytes_weighted"] = (
            total_enqueue_bytes / writer_enqueue["count"]
        )

    reader_dequeue = {
        "count": sum(r["dequeue"]["count"] for r in readers),
        "inline_count": sum(r["dequeue"]["inline_count"] for r in readers),
        "zmq_count": sum(r["dequeue"]["zmq_count"] for r in readers),
    }

    return {
        "record_count": len(records),
        "writer_processes": len(writers),
        "reader_processes": len(readers),
        "writer_enqueue": writer_enqueue,
        "reader_dequeue": reader_dequeue,
        "top_payload_types": top_payload_types,
        "records": records,
    }


def render(summary: dict) -> str:
    lines = []
    lines.append("Myelon IPC Summary")
    lines.append(
        f"- records: {summary['record_count']} "
        f"(writers={summary['writer_processes']}, readers={summary['reader_processes']})"
    )
    we = summary["writer_enqueue"]
    lines.append(
        f"- writer enqueue: count={we['count']} inline={we['inline_count']} "
        f"overflow={we['overflow_count']} remote_send={we['remote_send_count']}"
    )
    if we["count"] > 0:
        overflow_pct = 100.0 * we["overflow_count"] / we["count"]
        lines.append(
            f"- writer payloads: avg_bytes={we['payload_avg_bytes_weighted']:.1f} "
            f"max_bytes={we['payload_max_bytes']} overflow_rate={overflow_pct:.2f}%"
        )
    rd = summary["reader_dequeue"]
    lines.append(
        f"- reader dequeue: count={rd['count']} inline={rd['inline_count']} zmq={rd['zmq_count']}"
    )
    if summary["top_payload_types"]:
        lines.append("- top payload types:")
        for row in summary["top_payload_types"]:
            lines.append(
                f"  - {row['type']}: count={row['count']} "
                f"max_bytes={row['max_bytes']} overflow={row['overflow_count']}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.input, args.tag)
    summary = aggregate(records)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
