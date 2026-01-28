#!/usr/bin/env python3
"""
Compare PP benchmark results between baseline and optimized runs.

Usage:
    python 05_compare_results.py --baseline results/baseline_xxx --optimized results/nonblocking_xxx
    python 05_compare_results.py --auto  # Auto-find latest results
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BenchmarkComparison:
    """Comparison between baseline and optimized runs."""
    metric: str
    baseline: float
    optimized: float
    
    @property
    def improvement(self) -> float:
        """Positive = improvement, negative = regression."""
        if self.baseline == 0:
            return 0
        return ((self.baseline - self.optimized) / self.baseline) * 100
    
    @property
    def improvement_str(self) -> str:
        imp = self.improvement
        if imp > 0:
            return f"↓{imp:.1f}% (better)"
        elif imp < 0:
            return f"↑{-imp:.1f}% (worse)"
        else:
            return "no change"


def load_json(path: str) -> Optional[Dict]:
    """Load JSON file if exists."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def find_latest_results(results_dir: str, prefix: str) -> Optional[str]:
    """Find latest results directory with given prefix."""
    try:
        dirs = sorted(
            [d for d in os.listdir(results_dir) if d.startswith(prefix)],
            reverse=True
        )
        if dirs:
            return os.path.join(results_dir, dirs[0])
    except FileNotFoundError:
        pass
    return None


def compare_decode_results(baseline_path: str, optimized_path: str) -> List[BenchmarkComparison]:
    """Compare decode benchmark results."""
    baseline = load_json(baseline_path)
    optimized = load_json(optimized_path)
    
    comparisons = []
    
    if baseline and optimized:
        metrics = [
            ("throughput_tps", "Throughput (tok/s)"),
            ("mean_itl_ms", "Mean ITL (ms)"),
            ("median_itl_ms", "Median ITL (ms)"),
            ("p95_itl_ms", "P95 ITL (ms)"),
            ("mean_decode_time_ms", "Mean Decode Time (ms)"),
        ]
        
        for key, name in metrics:
            if key in baseline and key in optimized:
                # For throughput, higher is better, so invert
                if "tps" in key:
                    comparisons.append(BenchmarkComparison(
                        metric=name,
                        baseline=baseline[key],
                        optimized=optimized[key],
                    ))
                else:
                    comparisons.append(BenchmarkComparison(
                        metric=name,
                        baseline=baseline[key],
                        optimized=optimized[key],
                    ))
    
    return comparisons


def compare_recv_timing(baseline_path: str, optimized_path: str) -> List[BenchmarkComparison]:
    """Compare recv timing results."""
    baseline = load_json(baseline_path)
    optimized = load_json(optimized_path)
    
    comparisons = []
    
    if baseline and optimized:
        for key in baseline.keys():
            if key in optimized:
                comparisons.append(BenchmarkComparison(
                    metric=f"{key} (mean us)",
                    baseline=baseline[key]["mean_us"],
                    optimized=optimized[key]["mean_us"],
                ))
    
    return comparisons


def print_comparison_table(
    title: str,
    comparisons: List[BenchmarkComparison],
    baseline_name: str = "Baseline",
    optimized_name: str = "Optimized",
) -> None:
    """Print comparison table."""
    if not comparisons:
        print(f"\n{title}: No data available")
        return
    
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Metric':<30} {baseline_name:>15} {optimized_name:>15} {'Change':>20}")
    print("-" * 80)
    
    for c in comparisons:
        # For latency metrics, lower is better
        # For throughput, higher is better
        is_latency = "ms" in c.metric or "us" in c.metric
        
        if is_latency:
            # Lower is better for latency
            improvement = c.improvement
        else:
            # Higher is better for throughput
            improvement = -c.improvement
        
        if improvement > 0:
            status = f"✅ {improvement:.1f}% better"
        elif improvement < 0:
            status = f"❌ {-improvement:.1f}% worse"
        else:
            status = "➡️  no change"
        
        print(f"{c.metric:<30} {c.baseline:>15.2f} {c.optimized:>15.2f} {status:>20}")


def main():
    parser = argparse.ArgumentParser(description="Compare PP benchmark results")
    parser.add_argument("--baseline", type=str, help="Path to baseline results directory")
    parser.add_argument("--optimized", type=str, help="Path to optimized results directory")
    parser.add_argument("--async-depth", type=str, help="Path to async_depth results directory")
    parser.add_argument("--auto", action="store_true", help="Auto-find latest results")
    parser.add_argument("--results-dir", type=str, 
                        default="/root/Documents/sglang/pp_test_results",
                        help="Results directory")
    
    args = parser.parse_args()
    
    # Auto-find latest results
    if args.auto or not (args.baseline or args.optimized):
        args.baseline = find_latest_results(args.results_dir, "baseline_")
        args.optimized = find_latest_results(args.results_dir, "nonblocking_")
        args.async_depth = find_latest_results(args.results_dir, "async_depth_2_")
        
        print("Auto-detected results:")
        print(f"  Baseline: {args.baseline}")
        print(f"  Nonblocking: {args.optimized}")
        print(f"  Async depth: {args.async_depth}")
    
    # Compare baseline vs nonblocking
    if args.baseline and args.optimized:
        decode_comps = compare_decode_results(
            os.path.join(args.baseline, "decode_results.json"),
            os.path.join(args.optimized, "decode_results.json"),
        )
        print_comparison_table(
            "DECODE BENCHMARK: Baseline vs Nonblocking Recv",
            decode_comps,
            "Baseline",
            "Nonblocking",
        )
        
        recv_comps = compare_recv_timing(
            os.path.join(args.baseline, "recv_timing.json"),
            os.path.join(args.optimized, "recv_timing.json"),
        )
        print_comparison_table(
            "RECV TIMING: Baseline vs Nonblocking Recv",
            recv_comps,
            "Baseline",
            "Nonblocking",
        )
    
    # Compare baseline vs async_depth
    if args.baseline and args.async_depth:
        decode_comps = compare_decode_results(
            os.path.join(args.baseline, "decode_results.json"),
            os.path.join(args.async_depth, "decode_results.json"),
        )
        print_comparison_table(
            "DECODE BENCHMARK: Baseline vs Async Depth 2",
            decode_comps,
            "Baseline",
            "Async=2",
        )
        
        recv_comps = compare_recv_timing(
            os.path.join(args.baseline, "recv_timing.json"),
            os.path.join(args.async_depth, "recv_timing.json"),
        )
        print_comparison_table(
            "RECV TIMING: Baseline vs Async Depth 2",
            recv_comps,
            "Baseline",
            "Async=2",
        )
    
    # Summary and recommendations
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if args.async_depth:
        async_decode = load_json(os.path.join(args.async_depth, "decode_results.json"))
        if async_decode:
            print(f"\n--pp-async-batch-depth 2 (existing optimization):")
            print(f"  Mean ITL: {async_decode.get('mean_itl_ms', 0):.2f} ms")
            print(f"  Throughput: {async_decode.get('throughput_tps', 0):.1f} tok/s")
    
    if args.optimized:
        opt_decode = load_json(os.path.join(args.optimized, "decode_results.json"))
        if opt_decode:
            print(f"\nSGLANG_PP_NONBLOCKING_RECV=1 (new optimization):")
            print(f"  Mean ITL: {opt_decode.get('mean_itl_ms', 0):.2f} ms")
            print(f"  Throughput: {opt_decode.get('throughput_tps', 0):.1f} tok/s")
    
    print("\nKey metrics to evaluate success:")
    print("  ✅ R1 recv blocking < 1000us (from ~1800us)")
    print("  ✅ Mean ITL improved by 10%+")
    print("  ✅ Throughput improved for sequential workloads")
    print("  ⚠️  No regression for parallel workloads")


if __name__ == "__main__":
    main()
