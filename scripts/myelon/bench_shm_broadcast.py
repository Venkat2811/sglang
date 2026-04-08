#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch.distributed as dist

from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--max-chunk-bytes", type=int, default=1 << 22)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--writer-rank", type=int, default=0)
    parser.add_argument("--mode", choices=["bytes", "dict"], default="bytes")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def build_payload(mode: str, payload_bytes: int):
    blob = b"x" * payload_bytes
    if mode == "bytes":
        return blob
    return {
        "kind": "bench_shm_broadcast",
        "payload": blob,
        "payload_bytes": payload_bytes,
    }


def recv_payload_len(obj) -> int:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, dict) and "payload" in obj:
        return len(obj["payload"])
    raise TypeError(f"unexpected payload type: {type(obj)}")


def main():
    args = parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("gloo")

    mq = MessageQueue.create_from_process_group(
        dist.group.WORLD,
        max_chunk_bytes=args.max_chunk_bytes,
        max_chunks=args.max_chunks,
        writer_rank=args.writer_rank,
    )
    payload = build_payload(args.mode, args.payload_bytes)

    for _ in range(args.warmup_iterations):
        if rank == args.writer_rank:
            mq.broadcast_object(payload)
        else:
            obj = mq.broadcast_object()
            assert recv_payload_len(obj) == args.payload_bytes

    dist.barrier()
    started_at = time.perf_counter()
    for _ in range(args.iterations):
        if rank == args.writer_rank:
            mq.broadcast_object(payload)
        else:
            obj = mq.broadcast_object()
            assert recv_payload_len(obj) == args.payload_bytes
    dist.barrier()
    elapsed = time.perf_counter() - started_at

    result = {
        "kind": "bench_shm_broadcast",
        "rank": rank,
        "world_size": world_size,
        "payload_bytes": args.payload_bytes,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "mode": args.mode,
        "max_chunk_bytes": args.max_chunk_bytes,
        "max_chunks": args.max_chunks,
        "elapsed_s": elapsed,
        "ops_per_sec": args.iterations / elapsed if elapsed > 0 else 0.0,
    }

    if rank == args.writer_rank:
        print(json.dumps(result, sort_keys=True))
        if args.output_json is not None:
            args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
