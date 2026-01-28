#!/usr/bin/env python3
"""
PP Decode Micro-Benchmark: Isolate decode-phase latency.

This benchmark focuses on decode-heavy workloads where PP bubbles are most visible.
- Short prefill (minimize prefill overhead)
- Long decode (expose PP communication overhead)
- Sequential requests (no batching to hide latency)

Based on investigation session 5:
- Single-user sequential: async_depth=2 gives +12.2% improvement
- Multi-user parallel: async_depth=2 gives -12.6% regression

Usage:
    # Start PP server
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3-0.6B \
        --port 30000 \
        --pipeline-parallel-size 2 \
        --chunked-prefill-size 256
    
    # Run benchmark
    python scripts/pp_test_suite/02_decode_microbench.py \
        --url http://localhost:30000 \
        --mode sequential \
        --num-requests 20 \
        --output-len 128
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests


@dataclass
class RequestResult:
    """Result of a single request."""
    request_id: int
    input_tokens: int
    output_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    
    @property
    def inter_token_latency_ms(self) -> float:
        """Average time per token during decode."""
        if self.output_tokens > 1:
            return self.decode_time_ms / (self.output_tokens - 1)
        return self.decode_time_ms


@dataclass  
class BenchmarkResults:
    """Aggregated benchmark results."""
    mode: str
    num_requests: int
    output_len: int
    results: List[RequestResult] = field(default_factory=list)
    wall_time_s: float = 0
    
    @property
    def total_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)
    
    @property
    def throughput(self) -> float:
        return self.total_tokens / self.wall_time_s if self.wall_time_s > 0 else 0
    
    @property
    def mean_itl_ms(self) -> float:
        itls = [r.inter_token_latency_ms for r in self.results if r.output_tokens > 1]
        return statistics.mean(itls) if itls else 0
    
    @property
    def median_itl_ms(self) -> float:
        itls = [r.inter_token_latency_ms for r in self.results if r.output_tokens > 1]
        return statistics.median(itls) if itls else 0
    
    @property
    def p95_itl_ms(self) -> float:
        itls = sorted([r.inter_token_latency_ms for r in self.results if r.output_tokens > 1])
        if not itls:
            return 0
        idx = int(len(itls) * 0.95)
        return itls[min(idx, len(itls) - 1)]
    
    @property
    def mean_decode_time_ms(self) -> float:
        return statistics.mean(r.decode_time_ms for r in self.results) if self.results else 0


def send_request(
    url: str,
    request_id: int,
    prompt: str,
    max_tokens: int,
    stream: bool = True,
) -> RequestResult:
    """Send a single request and measure timing."""
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        if stream:
            # Streaming to measure time-to-first-token and ITL
            with requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    token_count += 1  # Approximate
                            except json.JSONDecodeError:
                                pass
        else:
            # Non-streaming
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            token_count = data.get("usage", {}).get("completion_tokens", 0)
            first_token_time = time.time()  # Can't distinguish
    
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return RequestResult(
            request_id=request_id,
            input_tokens=len(prompt.split()),
            output_tokens=0,
            prefill_time_ms=0,
            decode_time_ms=0,
            total_time_ms=0,
            tokens_per_second=0,
        )
    
    end_time = time.time()
    
    # Calculate timing
    total_time_ms = (end_time - start_time) * 1000
    
    if first_token_time and first_token_time > start_time:
        prefill_time_ms = (first_token_time - start_time) * 1000
        decode_time_ms = (end_time - first_token_time) * 1000
    else:
        # Can't distinguish, assume 10% prefill
        prefill_time_ms = total_time_ms * 0.1
        decode_time_ms = total_time_ms * 0.9
    
    return RequestResult(
        request_id=request_id,
        input_tokens=len(prompt.split()),
        output_tokens=token_count,
        prefill_time_ms=prefill_time_ms,
        decode_time_ms=decode_time_ms,
        total_time_ms=total_time_ms,
        tokens_per_second=token_count / (total_time_ms / 1000) if total_time_ms > 0 else 0,
    )


def run_sequential_benchmark(
    url: str,
    num_requests: int,
    output_len: int,
    warmup: int = 2,
) -> BenchmarkResults:
    """Run sequential (single-user) benchmark."""
    
    print(f"\n{'='*60}")
    print("SEQUENTIAL DECODE BENCHMARK")
    print(f"{'='*60}")
    print(f"Requests: {num_requests}, Output len: {output_len}, Warmup: {warmup}")
    
    results = BenchmarkResults(
        mode="sequential",
        num_requests=num_requests,
        output_len=output_len,
    )
    
    prompts = [
        f"Count from {i} to {i + 100} step by step" 
        for i in range(1, num_requests + warmup + 1)
    ]
    
    # Warmup
    print(f"\nWarmup ({warmup} requests)...")
    for i in range(warmup):
        result = send_request(url, -i, prompts[i], output_len, stream=True)
        print(f"  Warmup {i+1}: {result.output_tokens} tokens, {result.total_time_ms:.0f}ms")
    
    # Benchmark
    print(f"\nBenchmarking ({num_requests} requests)...")
    start_time = time.time()
    
    for i in range(num_requests):
        idx = warmup + i
        result = send_request(url, i, prompts[idx], output_len, stream=True)
        results.results.append(result)
        
        itl = result.inter_token_latency_ms
        print(f"  Request {i+1:3d}: {result.output_tokens:3d} tokens, "
              f"total={result.total_time_ms:6.0f}ms, "
              f"decode={result.decode_time_ms:6.0f}ms, "
              f"ITL={itl:5.2f}ms, "
              f"TPS={result.tokens_per_second:5.1f}")
    
    results.wall_time_s = time.time() - start_time
    
    return results


def run_parallel_benchmark(
    url: str,
    num_requests: int,
    output_len: int,
    concurrency: int = 4,
    warmup: int = 2,
) -> BenchmarkResults:
    """Run parallel (multi-user) benchmark."""
    
    print(f"\n{'='*60}")
    print(f"PARALLEL DECODE BENCHMARK (concurrency={concurrency})")
    print(f"{'='*60}")
    print(f"Requests: {num_requests}, Output len: {output_len}, Warmup: {warmup}")
    
    results = BenchmarkResults(
        mode=f"parallel-{concurrency}",
        num_requests=num_requests,
        output_len=output_len,
    )
    
    prompts = [
        f"Count from {i} to {i + 100} step by step"
        for i in range(1, num_requests + warmup + 1)
    ]
    
    # Warmup (sequential)
    print(f"\nWarmup ({warmup} requests)...")
    for i in range(warmup):
        result = send_request(url, -i, prompts[i], output_len, stream=False)
        print(f"  Warmup {i+1}: {result.output_tokens} tokens, {result.total_time_ms:.0f}ms")
    
    # Parallel benchmark
    print(f"\nBenchmarking ({num_requests} requests, {concurrency} concurrent)...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            idx = warmup + i
            future = executor.submit(
                send_request, url, i, prompts[idx], output_len, False
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.results.append(result)
    
    results.wall_time_s = time.time() - start_time
    
    # Sort by request_id for display
    results.results.sort(key=lambda r: r.request_id)
    
    return results


def print_results(results: BenchmarkResults) -> None:
    """Print benchmark results."""
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {results.mode.upper()}")
    print(f"{'='*60}")
    
    print(f"\nRequests: {results.num_requests}")
    print(f"Total tokens: {results.total_tokens}")
    print(f"Wall time: {results.wall_time_s:.2f}s")
    print(f"Throughput: {results.throughput:.1f} tok/s")
    
    print(f"\nInter-Token Latency (ITL):")
    print(f"  Mean:   {results.mean_itl_ms:.2f} ms")
    print(f"  Median: {results.median_itl_ms:.2f} ms")
    print(f"  P95:    {results.p95_itl_ms:.2f} ms")
    
    print(f"\nDecode Time per Request:")
    print(f"  Mean: {results.mean_decode_time_ms:.0f} ms")
    
    # PP efficiency estimate
    # With no PP overhead, ITL should be ~compute time per token
    # With PP overhead, ITL includes blocking recv time
    if results.mean_itl_ms > 0:
        # Assume ~2ms compute time for Qwen3-0.6B
        estimated_compute_ms = 2.0
        pp_overhead_ms = max(0, results.mean_itl_ms - estimated_compute_ms)
        overhead_pct = (pp_overhead_ms / results.mean_itl_ms) * 100
        print(f"\nEstimated PP overhead: {pp_overhead_ms:.2f}ms ({overhead_pct:.0f}% of ITL)")


def export_results(results: BenchmarkResults, output_path: str) -> None:
    """Export results to JSON."""
    data = {
        "mode": results.mode,
        "num_requests": results.num_requests,
        "output_len": results.output_len,
        "total_tokens": results.total_tokens,
        "wall_time_s": results.wall_time_s,
        "throughput_tps": results.throughput,
        "mean_itl_ms": results.mean_itl_ms,
        "median_itl_ms": results.median_itl_ms,
        "p95_itl_ms": results.p95_itl_ms,
        "mean_decode_time_ms": results.mean_decode_time_ms,
        "requests": [
            {
                "request_id": r.request_id,
                "output_tokens": r.output_tokens,
                "prefill_time_ms": r.prefill_time_ms,
                "decode_time_ms": r.decode_time_ms,
                "total_time_ms": r.total_time_ms,
                "itl_ms": r.inter_token_latency_ms,
            }
            for r in results.results
        ],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PP Decode Micro-Benchmark")
    parser.add_argument("--url", type=str, required=True, help="Server URL")
    parser.add_argument("--mode", choices=["sequential", "parallel", "both"], 
                        default="sequential", help="Benchmark mode")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of requests")
    parser.add_argument("--output-len", type=int, default=128, help="Output tokens per request")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests (parallel mode)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests")
    parser.add_argument("--export", type=str, help="Export results to JSON")
    
    args = parser.parse_args()
    
    # Check server health
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        resp.raise_for_status()
        print(f"Server is healthy: {args.url}")
    except Exception as e:
        print(f"Server not available: {e}")
        return
    
    all_results = []
    
    if args.mode in ["sequential", "both"]:
        results = run_sequential_benchmark(
            args.url,
            args.num_requests,
            args.output_len,
            args.warmup,
        )
        print_results(results)
        all_results.append(results)
        
        if args.export:
            export_results(results, args.export.replace(".json", "_sequential.json"))
    
    if args.mode in ["parallel", "both"]:
        results = run_parallel_benchmark(
            args.url,
            args.num_requests,
            args.output_len,
            args.concurrency,
            args.warmup,
        )
        print_results(results)
        all_results.append(results)
        
        if args.export:
            export_results(results, args.export.replace(".json", "_parallel.json"))


if __name__ == "__main__":
    main()
