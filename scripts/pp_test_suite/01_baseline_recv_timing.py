#!/usr/bin/env python3
"""
PP Baseline: Measure per-rank recv blocking time.

This script measures the ACTUAL bottleneck: recv_object and recv_tensor_dict blocking time.
Based on investigation sessions 3 & 7, the key metrics are:
- R0 recv blocking: ~1.1-1.3ms (async_depth=0) → ~0.27ms (async_depth=2)
- R1 recv blocking: ~1.8-2.0ms (async_depth=0) → ~0.95ms (async_depth=2)

Usage:
    # Start server with PP_DEBUG enabled
    SGLANG_PP_DEBUG=1 python -m sglang.launch_server \
        --model-path Qwen/Qwen3-0.6B \
        --port 30000 \
        --pipeline-parallel-size 2 \
        --chunked-prefill-size 256 \
        2>&1 | tee /tmp/pp_debug.log &
    
    # Wait for server, then run workload
    sleep 60
    python scripts/pp_test_suite/01_baseline_recv_timing.py --url http://localhost:30000
    
    # Analyze log
    python scripts/pp_test_suite/01_baseline_recv_timing.py --analyze /tmp/pp_debug.log
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests


@dataclass
class RecvTimingStats:
    """Statistics for recv blocking time."""
    rank: int
    count: int = 0
    times_us: List[float] = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.times_us) if self.times_us else 0
    
    @property
    def median(self) -> float:
        return statistics.median(self.times_us) if self.times_us else 0
    
    @property
    def p95(self) -> float:
        if not self.times_us:
            return 0
        sorted_times = sorted(self.times_us)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    @property
    def min(self) -> float:
        return min(self.times_us) if self.times_us else 0
    
    @property
    def max(self) -> float:
        return max(self.times_us) if self.times_us else 0


def generate_decode_workload(url: str, num_requests: int = 10, output_len: int = 64) -> float:
    """
    Generate decode-heavy workload to trigger PP communication.
    Returns total time taken.
    """
    print(f"Generating {num_requests} requests with {output_len} output tokens each...")
    
    start_time = time.time()
    total_tokens = 0
    
    for i in range(num_requests):
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": f"Count from 1 to {i + 20}"}],
            "max_tokens": output_len,
            "temperature": 0,
        }
        
        try:
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            total_tokens += tokens
            print(f"  Request {i+1}/{num_requests}: {tokens} tokens")
        except Exception as e:
            print(f"  Request {i+1} failed: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal: {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.1f} tok/s)")
    return elapsed


def parse_pp_debug_log(log_path: str) -> Dict[str, RecvTimingStats]:
    """
    Parse SGLANG_PP_DEBUG=1 log to extract recv timing.
    
    Expected log lines:
    [PP RECV_OBJ R0] recv size_tensor time=1164.1us src=1
    [PP RECV_OBJ R1] recv size_tensor time=2864.1us src=0
    [PP RECV R0] recv_object (metadata) time=1312.8us src=1
    [PP RECV R1] recv_object (metadata) time=2936.1us src=0
    """
    stats = {
        "r0_recv_object": RecvTimingStats(rank=0),
        "r1_recv_object": RecvTimingStats(rank=1),
        "r0_recv_size_tensor": RecvTimingStats(rank=0),
        "r1_recv_size_tensor": RecvTimingStats(rank=1),
        "r0_recv_tensor_dict": RecvTimingStats(rank=0),
        "r1_recv_tensor_dict": RecvTimingStats(rank=1),
    }
    
    # Patterns to match
    patterns = {
        # recv_object (metadata) timing
        r"\[PP RECV R(\d)\] recv_object \(metadata\) time=([\d.]+)us": "recv_object",
        # recv_object internal - size tensor (this is the blocking part)
        r"\[PP RECV_OBJ R(\d)\] recv size_tensor time=([\d.]+)us": "recv_size_tensor",
        # recv_tensor_dict timing
        r"\[PP RECV R(\d)\] recv_tensor_dict.*time=([\d.]+)us": "recv_tensor_dict",
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                for pattern, metric_type in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        rank = int(match.group(1))
                        time_us = float(match.group(2))
                        key = f"r{rank}_{metric_type}"
                        if key in stats:
                            stats[key].times_us.append(time_us)
                            stats[key].count += 1
                        break
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return {}
    
    return stats


def print_timing_report(stats: Dict[str, RecvTimingStats]) -> None:
    """Print formatted timing report."""
    print("\n" + "=" * 70)
    print("PP RECV BLOCKING TIME ANALYSIS")
    print("=" * 70)
    
    for key, stat in sorted(stats.items()):
        if stat.count == 0:
            continue
        
        rank = "R0" if stat.rank == 0 else "R1"
        metric = key.split("_", 1)[1]
        
        print(f"\n{rank} {metric}:")
        print(f"  Count:  {stat.count}")
        print(f"  Mean:   {stat.mean:,.1f} us")
        print(f"  Median: {stat.median:,.1f} us")
        print(f"  P95:    {stat.p95:,.1f} us")
        print(f"  Min:    {stat.min:,.1f} us")
        print(f"  Max:    {stat.max:,.1f} us")
    
    # Key bottleneck summary
    print("\n" + "-" * 70)
    print("KEY BOTTLENECK SUMMARY")
    print("-" * 70)
    
    r0_recv = stats.get("r0_recv_object", RecvTimingStats(rank=0))
    r1_recv = stats.get("r1_recv_object", RecvTimingStats(rank=1))
    
    if r0_recv.count > 0 and r1_recv.count > 0:
        print(f"\nR0 recv blocking (waiting for R1): {r0_recv.mean:,.1f} us mean")
        print(f"R1 recv blocking (waiting for R0): {r1_recv.mean:,.1f} us mean")
        print(f"\nTotal per-token blocking overhead: {r0_recv.mean + r1_recv.mean:,.1f} us")
        
        # Assessment
        if r1_recv.mean > 1500:
            print("\n⚠️  R1 blocking is HIGH (>1.5ms) - optimization opportunity!")
        elif r1_recv.mean > 800:
            print("\n⚡ R1 blocking is MODERATE (0.8-1.5ms)")
        else:
            print("\n✅ R1 blocking is LOW (<0.8ms) - well optimized")
    
    print("=" * 70)


def export_results(stats: Dict[str, RecvTimingStats], output_path: str) -> None:
    """Export results to JSON for comparison."""
    results = {}
    for key, stat in stats.items():
        if stat.count > 0:
            results[key] = {
                "count": stat.count,
                "mean_us": stat.mean,
                "median_us": stat.median,
                "p95_us": stat.p95,
                "min_us": stat.min,
                "max_us": stat.max,
            }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PP recv timing analysis")
    parser.add_argument("--url", type=str, help="Server URL to generate workload")
    parser.add_argument("--analyze", type=str, help="Path to PP debug log to analyze")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests")
    parser.add_argument("--output-len", type=int, default=64, help="Output tokens per request")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    
    args = parser.parse_args()
    
    if args.url:
        # Generate workload
        generate_decode_workload(
            args.url,
            num_requests=args.num_requests,
            output_len=args.output_len,
        )
        print("\nNow analyze the server log with: --analyze /path/to/log")
    
    if args.analyze:
        # Analyze log
        stats = parse_pp_debug_log(args.analyze)
        if stats:
            print_timing_report(stats)
            if args.export:
                export_results(stats, args.export)
        else:
            print("No PP timing data found in log.")
            print("Make sure server was started with SGLANG_PP_DEBUG=1")


if __name__ == "__main__":
    main()
