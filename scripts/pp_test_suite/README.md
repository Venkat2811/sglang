# SGLang Pipeline Parallelism (PP) Optimization Test Suite

This test suite is designed to measure the specific bottlenecks in PP decode and validate optimizations.

## TL;DR - Current Status

**Best optimization available**: `--pp-async-batch-depth 2`
- Reduces recv blocking by 60-80%
- Improves throughput by ~12% for single-user workloads
- ⚠️ Hurts throughput by ~13% for high-concurrency workloads

**Non-blocking recv**: DISABLED (needs architectural refactor)
- The PP loop has TWO recv operations that get out of sync
- Requires converting ALL recv ops to async, not just one

## Key Metrics

Based on investigation sessions 1-7 (2026-01-25), the critical metrics are:

| Metric | What it measures | Target |
|--------|------------------|--------|
| **R0 recv blocking** | Time R0 waits for R1 to send token | <300us (from ~1.2ms) |
| **R1 recv blocking** | Time R1 waits for R0 to send hidden_states | <1000us (from ~1.8ms) |
| **GPU idle time** | % of time GPU is idle (bubbles) | <20% (from ~60%) |
| **Per-token ITL** | Inter-token latency | Measure improvement |

## Why These Metrics

The previous investigation revealed:
- ❌ Pickle serialization is NOT the bottleneck (only 2% of time)
- ❌ Tensor transfers are NOT the bottleneck (only 6% of time)  
- ✅ **Blocking recv waiting for sender** is 90%+ of the overhead

## Test Workloads

### 1. Decode-Heavy Sequential (primary)
- Short prefill (32 tokens), long decode (128-512 tokens)
- Single request at a time (sequential)
- This is where PP bubbles are most visible
- Optimization SHOULD help here

### 2. Decode-Heavy Parallel (secondary)
- Same workload but 4-8 concurrent requests
- Tests if optimization has diminishing returns
- Optimization may have less impact here

### 3. High Concurrency (validation)
- 128 concurrent requests
- Tests for regression (optimization SHOULD NOT hurt here)

## Files

- `01_baseline_recv_timing.py` - Measure per-rank recv blocking time
- `02_decode_microbench.py` - Decode-only micro-benchmark
- `03_gpu_idle_profiler.py` - Measure GPU idle time (bubbles)
- `04_run_all.sh` - Run full test suite
- `05_compare_results.py` - Compare baseline vs optimized

## Usage

```bash
cd /root/Documents/sglang
source .venv/bin/activate

# Run baseline
./scripts/pp_test_suite/04_run_all.sh --mode baseline

# Run with optimization enabled
./scripts/pp_test_suite/04_run_all.sh --mode optimized

# Compare results
python scripts/pp_test_suite/05_compare_results.py
```

## Expected Results

With `SGLANG_PP_NONBLOCKING_RECV=1`:
- R0 recv blocking: Should stay ~300us (already low)
- R1 recv blocking: Should drop from ~1800us to <1000us
- GPU idle: Should drop from ~60% to <30%
- ITL: Should improve 10-20% for sequential workloads
